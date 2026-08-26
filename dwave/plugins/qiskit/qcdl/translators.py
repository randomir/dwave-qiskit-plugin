# Copyright 2020 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper methods for translating Qiskit QuantumCircuit instances into QCDL."""

from __future__ import annotations

import logging
import numbers
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from dwave.gate.results import Result
from dwave.gate.qcdl import QCDLModule, operations, procedure, qcdl
from dwave.gate.qcdl.qcdl_circuit import QCDLCircuit
from dwave.gate.results import format_memory
from dwave.cloud.exceptions import InvalidAPIResponseError

from qiskit import QuantumCircuit, qasm2
from qiskit.converters import circuit_to_dag
from qiskit.result.postprocess import _separate_bitstring

if TYPE_CHECKING:
    from dwave.gate.qcdl.components import Procedure
    from qiskit.circuit import ClassicalRegister, QuantumRegister

logger = logging.getLogger(__name__)


@dataclass
class QCDLWithMetadata:
    """A translated QCDL program along with metadata."""

    job_name: str
    qcdl: dict
    qasm: str | None
    clbit_to_tag: list[str | None] | None
    circuit_metadata: dict | list[QCDLWithMetadata]
    qiskit_header: dict[str, Any] | None
    next_tag: int | None


def _bit_register_layout(
    registers: Iterator[ClassicalRegister | QuantumRegister],
) -> tuple[list[list[str | int]], list[list[str | int]]]:
    """Describe a sequence of registers for the QCDL result header.

    Args:
        registers: The circuit's classical or quantum registers.

    Returns:
        The distinct ``[name, size]`` pairs, in first-seen order, and a
        ``[name, offset]`` label for every bit spanning all the registers.
    """
    register_sizes: dict[str, int] = {}
    bit_labels: list[list[str | int]] = []

    for register in registers:
        register_sizes.setdefault(register.name, register.size)
        bit_labels.extend([register.name, offset] for offset in range(register.size))

    return [[name, size] for name, size in register_sizes.items()], bit_labels


@dataclass(frozen=True)
class QiskitHeader:
    """A circuit's registers and metadata, in the format the QCDL result expects."""

    memory_slots: int
    global_phase: float
    num_qubits: int
    name: str
    creg_sizes: list[list[str | int]]
    clbit_labels: list[list[str | int]]
    qreg_sizes: list[list[str | int]]
    qubit_labels: list[list[str | int]]

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> QiskitHeader:
        """Build a header from a circuit's registers and metadata.

        Args:
            circuit: A Qiskit circuit.

        Returns:
            The header describing the circuit's registers and metadata.
        """
        creg_sizes, clbit_labels = _bit_register_layout(circuit.cregs)
        qreg_sizes, qubit_labels = _bit_register_layout(circuit.qregs)
        return cls(
            memory_slots=circuit.num_clbits,
            global_phase=circuit.global_phase,
            num_qubits=circuit.num_qubits,
            name=circuit.name,
            creg_sizes=creg_sizes,
            clbit_labels=clbit_labels,
            qreg_sizes=qreg_sizes,
            qubit_labels=qubit_labels,
        )


class InstructionMemoryEstimate:
    """Estimate for how much of a QCDL's instruction memory each operation will use."""

    def __init__(
        self,
        fixed_cost_per_qubit: float = 0.03282504,
        cz_estimate: float = 0.00066796,
        sx_estimate: float = 0.00015183,
        measure_estimate: float = 0.00286359,
    ):
        self._fixed_cost_per_qubit = fixed_cost_per_qubit
        self._estimate_per_op = dict(
            barrier=0,
            measure=measure_estimate,
            reset=measure_estimate,
            id=sx_estimate,
            swap=3 * cz_estimate,
        )
        # 2q gates
        for gate in ["cz", "cx", "cy", "cu", "rzz"]:
            self._estimate_per_op[gate] = cz_estimate

        # these are likely to end up as 2 sx gates plus rz
        for gate in ["x", "y", "u"]:
            self._estimate_per_op[gate] = 2 * sx_estimate

        # single 1q basis gates
        for gate in ["h", "rx", "ry", "sx", "sxdg"]:
            self._estimate_per_op[gate] = sx_estimate

        # virtual gates
        for gate in ["rz", "z", "p", "s", "sdg", "t", "tdg"]:
            self._estimate_per_op[gate] = 0

    @staticmethod
    def _make_key(op: str, qubits: int | list[int] | tuple[int, ...]) -> str:
        """Build the lookup key for a per-qubit-combination estimate override."""
        if isinstance(qubits, int):
            qubits = [qubits]
        qubits = tuple(qubits)
        return f"{op}, {qubits}"

    def estimate_op(self, op: str, qubits: list[int]) -> float:
        """Estimate the instruction memory cost of a single operation.

        Args:
            op: The operation name.
            qubits: The qubit indices the operation acts on.

        Returns:
            The estimated instruction memory cost.
        """
        # if you want to have a different estimate based on which qubit, it's
        # possible this way:
        key = self._make_key(op, qubits)
        if key in self._estimate_per_op:
            return self._estimate_per_op[key]

        unknown_estimate = (
            self._estimate_per_op["cz"]
            if len(qubits) == 2
            else self._estimate_per_op["sx"]
        )
        return self._estimate_per_op.get(op, unknown_estimate)

    def estimate_circuit(self, circuit: QuantumCircuit) -> dict[int, float]:
        """This is a very crude estimate for how many instructions (handled as a
        percentage of full capacity) this circuit would use.

        This estimate ignores:
        * transpilation and coupling maps
        * dynamical decoupling
        * compiler options
        * procedurization
        * synchronization
        * differences in QPU configurations

        The estimates were created with a simple linear fit of number operations
        vs percent full against the model 8 qubit configuration. Based on this,
        this is an underestimate that is intended to be accounted for by setting
        qcdl_pack_target somewhere less than 100%.

        .. note:: This approach is not likely long term.

        FIXME: A more dynamic retrieval of these parameters.

        Args:
            circuit: A Qiskit circuit.

        Returns:
            Dictionary for qubit to estimated percent full.
        """

        pct_per_qubit: dict[int, float] = defaultdict(
            lambda: self._fixed_cost_per_qubit
        )
        for inst in circuit.data:
            qubits = [circuit.qubits.index(q) for q in inst.qubits]
            estimate = self.estimate_op(op=inst.operation.name, qubits=qubits)
            for q in qubits:
                pct_per_qubit[q] += estimate
        return dict(pct_per_qubit)


_INSTRUCTION_MEMORY_ESTIMATOR = InstructionMemoryEstimate()


def group_circuits_by_instruction_estimates(
    circuits: list[QuantumCircuit], qcdl_pack_target: float
) -> list[list[QuantumCircuit]]:
    """Break apart a list of circuits into groups of circuits which, based on
    estimates, could be compiled into a single QCDL.

    This is just a simple greedy algorithm which makes no attempt to optimize.

    Args:
        circuits: List of Qiskit circuits.
        qcdl_pack_target: The cutoff for each group.

    Returns:
        Groups of :class:`QuantumCircuit`.
    """
    group_estimate: dict[int, float] = {}
    groups: list[list[QuantumCircuit]] = [[]]
    for circuit in circuits:
        logger.info(f"{circuit.name} has {circuit.count_ops()}")
        # estimate for this circuit:
        circuit_estimate = _INSTRUCTION_MEMORY_ESTIMATOR.estimate_circuit(circuit)
        # estimate if we include this circuit in the current group:
        included_group_estimate = {
            q: circuit_estimate.get(q, 0) + group_estimate.get(q, 0)
            for q in list(circuit_estimate) + list(group_estimate)
        }
        included_group_estimate_max = max(included_group_estimate.values())

        if included_group_estimate_max > qcdl_pack_target and len(groups[-1]) > 0:
            # put this circuit in the next group
            logger.info(f"group {len(groups)} has {group_estimate=} instructions")
            groups.append([])
            group_estimate = circuit_estimate
        else:
            group_estimate = included_group_estimate
        groups[-1].append(circuit)

    logger.info(f"group {len(groups)} has {group_estimate=} instructions")
    return groups


def _active_qubits(circuit: QuantumCircuit) -> list[int]:
    """Return the indices of qubits that are used by at least one instruction."""
    dag = circuit_to_dag(circuit)
    # NOTE: a barrier is not counted as idle!
    active_qubits = [qubit for qubit in circuit.qubits if qubit not in dag.idle_wires()]
    return [circuit.qubits.index(q) for q in active_qubits]


def circuit_to_qcdl(
    circuit: QuantumCircuit,
    procedure: Procedure | None = None,
    next_tag: int = 0,
    name_prefix: str = "",
) -> QCDLWithMetadata:
    """Build a circuit in QCDL's instruction format from Qiskit instructions.

    Args:
        circuit: A Qiskit circuit.
        procedure: If provided, instructions will be added
            to this procedure. Otherwise, a new QCDL will be created and
            instructions added to that.
        next_tag: If provided, the tags will start from here. This can
            help make the tags unique across multiple procedures.
        name_prefix: The job name will be the circuit name with this prefix.

    Raises:
        ValueError: If unsupported measurement is detected.

    Returns:
        The QCDL with its metadata.
    """
    # required for determining whether TODO
    is_top_level = procedure is None

    if is_top_level:
        if circuit.num_clbits == 0:
            raise ValueError(f"circuit {circuit.name} has no measurements")

        qcdl_program = QCDLCircuit()
        procedure = qcdl_program.main
        active_qubits = _active_qubits(circuit)
        if len(active_qubits) == 0:
            raise ValueError(f"no active qubits found in circuit {circuit.name}")
        operations.initialize(*[procedure.q(q) for q in active_qubits])

    def _numeric_params(instruction) -> list:
        # ParameterExpressions occasionally show up unresolved; pass such values
        # through as-is (with a warning) instead of failing the whole translation.
        values = []
        for value in getattr(instruction, "params", []):
            try:
                value = float(value)
            except (TypeError, ValueError):
                logger.warning(f"can't convert {value} to a float in op {instruction}")
            values.append(value)
        return values

    clbit_to_tag: list[str | None] = [None] * circuit.num_clbits

    for instruction, qubits, clbits in circuit.data:
        # check if `instruction.condition` exists and evaluates True
        if getattr(instruction, "condition", None):
            raise NotImplementedError("qiskit-aqumen does not support control-flow")

        qcdl_args = [procedure.q(circuit.qubits.index(q)) for q in qubits]

        qcdl_kwargs = {}
        if instruction.name == "measure":
            tag = str(next_tag)
            next_tag += 1
            clbit_to_tag[circuit.clbits.index(clbits[0])] = tag
            qcdl_kwargs["tag"] = tag

        qcdl_args.extend(_numeric_params(instruction))

        if hasattr(operations, instruction.name):
            getattr(operations, instruction.name)(*qcdl_args, **qcdl_kwargs)
        elif instruction.name == "barrier":
            target, barrier_qubits = qcdl_args[0], qcdl_args[1:]
            target.barrier(*barrier_qubits, **qcdl_kwargs)
        else:
            raise NotImplementedError(
                f"operation {instruction.name} is not implemented"
            )

    qiskit_header = QiskitHeader.from_circuit(circuit)

    if is_top_level:
        qdict = qcdl_program.to_model().model_dump()

        qdict.setdefault("metadata", {}).update(
            qiskit=asdict(qiskit_header),
            qasm=qasm2.dumps(circuit),
            circuit_metadata={**(circuit.metadata or {})},
            clbit_to_tag=clbit_to_tag,
            next_tag=next_tag,
        )
    else:
        # qdict is None when procedure was passed in (not the top-level circuit), in
        # which case there's no QCDL dict to attach metadata to.
        qdict = None

    job_name = f"{name_prefix}_{circuit.name}" if name_prefix else circuit.name
    return QCDLWithMetadata(
        job_name=job_name,
        qcdl=qdict,
        qasm=qasm2.dumps(circuit),
        clbit_to_tag=clbit_to_tag,
        circuit_metadata={**(circuit.metadata or {})},
        qiskit_header=asdict(qiskit_header),
        next_tag=next_tag,
    )


def circuit_to_procedure(
    circuit: QuantumCircuit,
    qubits: list[QCDLModule],
    proc_name: str,
    next_tag: int,
) -> QCDLWithMetadata:
    """Translate a circuit into a new named procedure on the given qubits.

    Args:
        circuit: A Qiskit circuit.
        qubits: The QCDL qubits the procedure will be defined on.
        proc_name: The name of the new procedure.
        next_tag: The tags will start from here. This can help make the tags
            unique across multiple procedures.

    Returns:
        The QCDL with its metadata.
    """
    def f(*proc_qubits: QCDLModule):
        proc_qubits[0].comment(f"circuit {circuit.name}")
        operations.initialize(*proc_qubits)
        return circuit_to_qcdl(
            circuit=circuit,
            procedure=proc_qubits[0].procedure,
            next_tag=next_tag,
        )

    return procedure(f=f, proc_name=proc_name)(*qubits)


def concatenate_circuits_to_qcdl(
    circuits: list[QuantumCircuit], name_prefix: str = ""
) -> QCDLWithMetadata:
    """Concatenate a list of circuits into one QCDL.

    Each circuit is placed into its own procedure, and the first instruction in
    that procedure will be initialize. Each measurement is given a tag that will
    be unique across all the circuits.

    Args:
        circuits: A list of Qiskit circuits.
        name_prefix: Prefix for the job name. Defaults to "".

    Returns:
        The QCDL with its metadata.
    """
    active_qubits = sorted(
        set().union(*[set(_active_qubits(circuit)) for circuit in circuits])
    )
    if len(active_qubits) == 0:
        raise ValueError("no active qubits found in any circuit")
    num_qubits = max(active_qubits) + 1
    circuit_metadata: list[QCDLWithMetadata] = []

    @qcdl(num_qubits)
    def main(**kwargs: QCDLModule):
        qubits = [kwargs[f"q{q}"] for q in active_qubits]
        next_tag = 0
        for idx, circuit in enumerate(circuits):
            qcdl_metadata = circuit_to_procedure(
                circuit=circuit,
                qubits=qubits,
                proc_name=f"circuit_{idx}",
                next_tag=next_tag,
            )

            circuit_metadata.append(qcdl_metadata)
            next_tag = qcdl_metadata.next_tag
        return circuit_metadata

    qcdl_input = main().model_dump()
    job_name = f"{name_prefix}_concatenated" if name_prefix else "concatenated"

    if not isinstance(qcdl_input, dict):
        raise TypeError("qcdl_input must be a dict")

    for idx, qwm in enumerate(circuit_metadata):
        # use the job_name if it exists otherwise make a unique name for reference later
        job_key = qwm.job_name or f"{job_name}_{idx}"  # ensures a valid key

        # check for existing metadata/create dict as needed
        if "metadata" not in qcdl_input:
            qcdl_input["metadata"] = {}

        # this moves all the needed metadata to the executed_qcdl per circuit
        md = qcdl_input["metadata"].get(job_key, {})
        md["qiskit"] = qwm.qiskit_header
        md["qasm"] = qwm.qasm
        md["next_tag"] = qwm.next_tag
        md["clbit_to_tag"] = qwm.clbit_to_tag
        md["circuit_metadata"] = qwm.circuit_metadata

        qcdl_input["metadata"][job_key] = md

    return QCDLWithMetadata(
        job_name=job_name,
        qcdl=qcdl_input,
        qasm=None,
        clbit_to_tag=None,
        circuit_metadata=circuit_metadata,
        qiskit_header=None,
        next_tag=None,
    )


def circuits_to_qcdls(
    circuits: list[QuantumCircuit],
    job_id: str = "job",
    qcdl_pack_target: numbers.Real | bool = True,
) -> Iterator[QCDLWithMetadata]:
    """Convert a list of :class:`QuantumCircuit` into a list of :class:`QCDLWithMetadata`.

    We're playing a game of blackjack: how close can we get to 100% full without
    going over. This depends on our ability to estimate how full the QCDL will
    be before we attempt to compile it. To account for the approximations in the
    estimation, we target less than 100% full.

    .. note:: The more instructions a QCDL has, the longer it will take to compile.

    Args:
        circuits: The Qiskit :class:`QuantumCircuit` instances.
        job_id: The Qiskit Job ID, used in naming the QCDLs. Defaults to "job".
        qcdl_pack_target: How full to pack the QCDLs. If False, each :class:`QuantumCircuit`
            will be in a separate QCDLs. If True, defaults to 0.4.

    Yields:
        The QCDL instances.
    """
    if qcdl_pack_target is True:
        qcdl_pack_target = 0.4
    elif qcdl_pack_target is False:
        qcdl_pack_target = 0.0

    if qcdl_pack_target > 0:
        groups = group_circuits_by_instruction_estimates(
            circuits, qcdl_pack_target=qcdl_pack_target
        )
        logger.info(f"broke {len(circuits)} circuits into {len(groups)} groups")
        for idx, group in enumerate(groups):
            name_prefix = job_id
            if len(groups) > 1:
                name_prefix += f"[{idx}]"
            yield concatenate_circuits_to_qcdl(circuits=group, name_prefix=name_prefix)
    else:
        # Eventually we could probably fold this case into the other
        for idx, circuit in enumerate(circuits):
            yield circuit_to_qcdl(circuit, name_prefix=f"{job_id}[{idx}]")


def make_qiskit_counts(
    result: Result, metadata: QCDLWithMetadata
) -> dict[str, int]:
    """Build a Qiskit-style counts dict from a QCDL result and its metadata.

    Args:
        result: The QCDL result.
        metadata: The metadata produced when the circuit was translated to QCDL.

    Returns:
        A mapping of bitstrings to the number of shots observed for them.
    """
    memory = [None] * len(metadata.clbit_to_tag)
    measurements = result.get_measurements()
    for clbit_idx, tag in enumerate(metadata.clbit_to_tag):
        mem_idx = len(metadata.clbit_to_tag) - clbit_idx - 1
        if tag is None:
            memory[mem_idx] = np.array(["0"] * result.num_shots)
            continue
        if not measurements:
            raise InvalidAPIResponseError("no tagged measurements are available")
        elif tag not in measurements:
            raise InvalidAPIResponseError(
                f"{tag=} for clbit={clbit_idx} is not found in tagged measurements"
            )

        tagged_measurements = measurements[tag]
        register = [
            f"q{idx}"
            for idx in range(len(tagged_measurements))
            if len(tagged_measurements[idx])
        ]
        # since each measurement was given a unique tag, only one qubit is
        # expected to have that tag.
        if len(register) != 1:
            raise InvalidAPIResponseError(
                f"invalid register {register} for clbit={clbit_idx} with {tag=}"
            )
        tag_mem: np.ndarray = format_memory(
            tagged_measurements,
            register=register,
            num_shots=result.num_shots,
        )
        # If there were multiple measurements into the same clbit from the same
        # qubit, then we take the last measurement.
        memory[mem_idx] = tag_mem[-1].reshape((result.num_shots,))

    memory = np.array(memory)
    counts = dict(Counter(map("".join, memory.transpose())))
    creg_sizes = metadata.qiskit_header["creg_sizes"]
    counts = {
        _separate_bitstring(k, creg_sizes=creg_sizes): v for k, v in counts.items()
    }
    return counts
