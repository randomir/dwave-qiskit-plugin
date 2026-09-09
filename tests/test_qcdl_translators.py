# Copyright 2026 D-Wave
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

"""Test the qcdl translators."""

import random
import re

import numpy as np
import pytest
from dwave.gate.qcdl import qcdl
from dwave.gate.utils import print_qcdl
from dwave.gate.results import Result
from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    instruction,
)

from dwave.plugins.qiskit.qcdl.translators import (
    InstructionMemoryEstimate,
    _active_qubits,
    circuit_to_procedure,
    circuit_to_qcdl,
    circuits_to_qcdls,
    concatenate_circuits_to_qcdl,
    group_circuits_by_instruction_estimates,
    make_qiskit_counts,
)


def test_measurement_only_circuit():
    """Test a valid circuit with only measurements."""
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0"]


def test_simple_circuit():
    """Test basic structure of a simple circuit"""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0"]


# pylint: disable=invalid-name
def test_circuit_with_entangling_ops():
    """Test structure of circuits with entangling ops."""
    qc = QuantumCircuit(2, 2)
    qc.cx(1, 0)
    qc.measure_all()
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, None, "0", "1"]


def test_barrier():
    """Barrier is translated via the sole operations.barrier code path."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.barrier(0, 1)
    qc.measure([0, 1], [0, 1])

    qcdl_metadata = circuit_to_qcdl(qc)
    qcdl_str = print_qcdl(qcdl_metadata.qcdl, to_Display=False)
    assert "barrier" in qcdl_str


def test_rotation_from_instruction_params():
    """Test that instruction parameters are used for rotation."""
    qc = QuantumCircuit(2)
    qc.append(instruction.Instruction("rx", 1, 0, [1.0]), [1])
    qc.measure_all()
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0", "1"]


def test_invalid_instruction():
    """test that the number of qubits for a gate is validated"""
    num_qubits = random.randint(3, 10)
    qc = QuantumCircuit(num_qubits)
    qc.append(instruction.Instruction("rx", num_qubits, 0, [1.0]), range(num_qubits))
    qc.measure_all()
    with pytest.raises(
        TypeError,
        match=f"takes 2 positional arguments but {num_qubits+1} were given",
    ):
        circuit_to_qcdl(qc)


def test_unknown_gate():
    qc = QuantumCircuit(2)
    gate_name = "not_a_gate"
    qc.append(instruction.Instruction(gate_name, 1, 0, [1.0]), [1])
    qc.measure_all()

    with pytest.raises(NotImplementedError, match=f"{gate_name} is not implemented"):
        circuit_to_qcdl(qc)


def test_circuit_measurement():
    """
    Test that putting an instruction on a qubit that has been measured is allowed
    """
    qc = QuantumCircuit(2, 2)
    qc.measure(1, 1)
    qc.x(1)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, "0"]


def test_unmeasured_clbit():
    """If only one qubit/clbit is not measured"""
    qc = QuantumCircuit(2, 2)
    qc.measure(1, 1)
    qc.x(0)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, "0"]


@pytest.mark.parametrize("q", [0, 1, 2, 3])
def test_circuit_with_multiple_registers(q):
    """Test multiple classical registers"""
    qr0 = QuantumRegister(2, "qr0")
    qr1 = QuantumRegister(2, "qr1")
    cr0 = ClassicalRegister(2, "cr0")
    cr1 = ClassicalRegister(2, "cr1")

    qc = QuantumCircuit(
        qr0,
        qr1,
        cr0,
        cr1,
    )

    qc.x(q)
    qc.measure([qr0[0], qr0[1], qr1[0], qr1[1]], [cr0[0], cr0[1], cr1[0], cr1[1]])

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0", "1", "2", "3"]


@pytest.mark.parametrize("q", [0, 1, None])
def test_measure_to_multiple_cbits(q):
    """Test that we are allowed to measure the same qubit into multiple classical registers"""
    qc = QuantumCircuit(2, 2)
    if q is not None:
        qc.x(q)
    qc.measure(1, 0)
    qc.measure(1, 1)

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0", "1"]


@pytest.mark.parametrize("qubits_used", [[0], [1], [0, 1], [3, 4, 5]])
def test_initialize(qubits_used):
    qc = QuantumCircuit(8, 8)
    for q in qubits_used:
        qc.x(q)
        qc.measure(q, q)
    assert _active_qubits(qc) == qubits_used

    qcdl = circuit_to_qcdl(qc)
    qcdl_str = print_qcdl(qcdl.qcdl, to_Display=False)
    expected_initialize = (
        f"q{qubits_used[0]}.initialize("
        + ", ".join(f"q{q}" for q in qubits_used[1:])
        + ")"
    )
    assert expected_initialize in qcdl_str


@pytest.mark.parametrize("q", [0, 1, None])
def test_measure_from_multiple_qubits(q):
    """Test that we are allowed to measure different qubits into the same cbit"""
    qc = QuantumCircuit(2, 2)
    if q is not None:
        qc.x(q)
    qc.measure(0, 1)
    qc.measure(1, 1)

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, "1"]


@pytest.mark.parametrize("q", [0, 1, None])
def test_multiple_measures_per_shot(q):
    """Measure multiple times same qubit to same cbit"""
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(qr, cr)
    if q is not None:
        qc.x(q)
    for _ in range(3):
        qc.measure(qr, cr)

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["4", "5"]


def test_if():
    bits = [Qubit(), Qubit(), Clbit()]
    qc = QuantumCircuit(bits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    with qc.if_test((bits[2], 0)) as else_:
        qc.h(0)
    with else_:
        qc.x(0)

    with pytest.raises(NotImplementedError, match="control-flow"):
        circuit_to_qcdl(qc)


def test_no_measurments():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    with pytest.raises(ValueError, match="no measurements"):
        circuit_to_qcdl(qc)


@pytest.mark.parametrize("qcdl_pack_target", [False, True])
def test_no_measurements_rejected_regardless_of_packing(qcdl_pack_target):
    """A measurement-free circuit must be rejected the same way whether or
    not it ends up packed together with other circuits."""
    qc_ok = QuantumCircuit(1, 1, name="ok")
    qc_ok.measure(0, 0)

    qc_no_measure = QuantumCircuit(1, name="no_measure")
    qc_no_measure.h(0)

    with pytest.raises(ValueError, match="no measurements"):
        list(
            circuits_to_qcdls(
                [qc_ok, qc_no_measure], qcdl_pack_target=qcdl_pack_target
            )
        )


def test_circuits_to_qcdls_with_empty_first_circuit():
    """A circuit with no instructions must not crash instruction-estimate
    grouping with an opaque max() error; it should surface the normal
    validation error instead."""
    with pytest.raises(ValueError, match="no active qubits"):
        list(circuits_to_qcdls([QuantumCircuit(1, 1)]))


def _check_circuit_concatenation(circuits: list[QuantumCircuit]) -> None:
    num_circuits = len(circuits)
    qcdl_metadata = concatenate_circuits_to_qcdl(circuits=circuits)
    assert len(qcdl_metadata.circuit_metadata) == num_circuits
    assert isinstance(qcdl_metadata.qcdl, dict)

    # metadata dict
    metadata = qcdl_metadata.qcdl.get("metadata", {})
    assert metadata is not None

    # check each circuit's qiskit header exists
    for qwm in qcdl_metadata.circuit_metadata:
        md_entry = metadata.get(qwm.job_name)
        assert md_entry is not None, f"Metadata missing for job_name {qwm.job_name}"
        assert (
            "qiskit" in md_entry
        ), f"qiskit header missing for job_name {qwm.job_name}"
        assert md_entry["qiskit"] == qwm.qiskit_header

    expected_job_names = {qwm.job_name for qwm in qcdl_metadata.circuit_metadata}
    assert set(metadata.keys()) == expected_job_names

    for qwm in qcdl_metadata.circuit_metadata:
        entry = metadata[qwm.job_name]
        assert "qiskit" in entry
        assert entry["qiskit"] == qwm.qiskit_header
        # circuit metadata (e.g. RB's "xval") must round-trip when packing
        assert "circuit_metadata" in entry
        assert entry["circuit_metadata"] == qwm.circuit_metadata

    # each has an initialize, and they're all the same (all qubits)
    qcdl_str = print_qcdl(qcdl_metadata.qcdl, to_Display=False)
    initializations = list(
        re.findall(r"^\s*q\d\.initialize\([q\d\,\s]*\)$", qcdl_str, flags=re.MULTILINE)
    )
    assert len(initializations) == num_circuits
    assert len(set(initializations)) == 1

    tags = set()
    for circuit_qcdl_metadata in qcdl_metadata.circuit_metadata:
        tags_used = [
            tag for tag in circuit_qcdl_metadata.clbit_to_tag if tag is not None
        ]
        assert len(set(tags_used)) == len(tags_used)
        assert tags.isdisjoint(tags_used)
        tags.update(tags_used)
        assert circuit_qcdl_metadata.qcdl is None


@pytest.mark.parametrize("measure_reps", range(1, 4))
def test_circuit_concatenation(measure_reps):
    circuits = []
    labels = ["00", "10", "01", "11", "0", "1"]
    random.shuffle(labels)
    for label in labels:
        qc = QuantumCircuit(len(label), name=f"circ{label}")
        for idx, val in enumerate(reversed(label)):
            if val == "1":
                qc.x(idx)
        for _ in range(measure_reps):
            qc.measure_all()
        circuits.append(qc)

    _check_circuit_concatenation(circuits=circuits)


@pytest.mark.parametrize("qubits_used", [[0], [1], [0, 1], [3, 4, 5]])
def test_concatenation_active_qubits(qubits_used):
    circuits = []
    # each circuit declares and uses a different number of qubits
    for q in qubits_used * 3:
        qc = QuantumCircuit(q + 1, q + 1, name=f"circ{q}")
        qc.measure(q, q)
        circuits.append(qc)

    _check_circuit_concatenation(circuits=circuits)
    qcdl_metadata = concatenate_circuits_to_qcdl(circuits=circuits)
    # make sure the final qubits_used is what we expect
    assert qcdl_metadata.qcdl["program"]["signature"]["qubits_used"] == [
        f"q{q}" for q in qubits_used
    ]


def test_group_circuits_by_instruction_estimates():
    circuits = []
    for idx in range(10):
        qc = QuantumCircuit(1, name=f"circ{idx}")
        for _ in range(10):
            qc.x(0)
        circuits.append(qc)

    estimator = InstructionMemoryEstimate()
    est_per_circuit = estimator.estimate_circuit(circuits[0])[0]
    assert est_per_circuit > 0.01

    # all separated
    groups = group_circuits_by_instruction_estimates(circuits, qcdl_pack_target=0)
    assert len(groups) == len(circuits)

    # all in one group
    groups = group_circuits_by_instruction_estimates(circuits, qcdl_pack_target=1000)
    assert len(groups) == 1

    # two groups
    groups = group_circuits_by_instruction_estimates(
        circuits, qcdl_pack_target=est_per_circuit * 5
    )
    assert len(groups) == 2


def test_qiskit_headers_are_per_circuit_not_combined():

    # add in two circuits
    qc1 = QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)

    qc2 = QuantumCircuit(1, 1)
    qc2.x(0)
    qc2.measure(0, 0)

    result = concatenate_circuits_to_qcdl([qc1, qc2])

    metadata = result.qcdl["metadata"]

    # should be 1 metadata per circuit
    assert len(metadata) == 2

    # this should not exist, should be split by {"circuit_name": "qiskit": {qiskit header content}}
    assert "qiskit" not in metadata

    # check each header
    for qwm in result.circuit_metadata:
        assert metadata[qwm.job_name]["qiskit"] == qwm.qiskit_header

    # headers are different, we gave it two different circuits
    headers = [v["qiskit"] for v in metadata.values()]
    assert headers[0] != headers[1]


@pytest.mark.parametrize("name", ["rb_seq", ""])
def test_concatenate_circuits_with_duplicate_names(name):
    """Circuits sharing a name (including an empty name) must not collide;
    each must get a unique, non-empty job_name and its own metadata entry."""
    qc1 = QuantumCircuit(1, 1, name=name)
    qc1.h(0)
    qc1.measure(0, 0)
    qc2 = QuantumCircuit(1, 1, name=name)
    qc2.x(0)
    qc2.measure(0, 0)

    result = concatenate_circuits_to_qcdl([qc1, qc2])
    job_names = [qwm.job_name for qwm in result.circuit_metadata]

    assert all(job_names)
    assert len(set(job_names)) == len(job_names)

    metadata = result.qcdl["metadata"]
    assert len(metadata) == 2
    for qwm in result.circuit_metadata:
        assert metadata[qwm.job_name]["qasm"] == qwm.qasm


def test_circuit_to_procedure():
    """A circuit translated via circuit_to_procedure is attached to a new,
    named procedure instead of producing its own standalone QCDL."""
    qc = QuantumCircuit(1, 1, name="my_circuit")
    qc.x(0)
    qc.measure(0, 0)

    captured = {}

    @qcdl(1)
    def main(**kwargs):
        qubits = [kwargs["q0"]]
        captured["metadata"] = circuit_to_procedure(
            circuit=qc, qubits=qubits, proc_name="my_proc", next_tag=0
        )

    qcdl_program = main().model_dump()
    metadata = captured["metadata"]

    assert metadata.qcdl is None
    assert metadata.clbit_to_tag == ["0"]

    qcdl_str = print_qcdl(qcdl_program, to_Display=False)
    assert "my_proc" in qcdl_str


def test_circuits_to_qcdls_without_packing():
    """With packing disabled, each circuit gets its own QCDL."""
    circuits = [QuantumCircuit(1, 1, name=f"circ{i}") for i in range(2)]
    for qc in circuits:
        qc.measure(0, 0)

    results = list(circuits_to_qcdls(circuits, job_id="job", qcdl_pack_target=False))

    assert len(results) == len(circuits)
    for idx, (qwm, qc) in enumerate(zip(results, circuits)):
        assert qwm.qcdl is not None
        assert qwm.job_name == f"job[{idx}]_{qc.name}"


def test_circuits_to_qcdls_with_packing():
    """With a generous pack target, circuits are combined into one QCDL."""
    circuits = [QuantumCircuit(1, 1, name=f"circ{i}") for i in range(3)]
    for qc in circuits:
        qc.measure(0, 0)

    results = list(circuits_to_qcdls(circuits, job_id="job", qcdl_pack_target=1000))

    assert len(results) == 1
    assert results[0].qcdl is not None
    assert len(results[0].circuit_metadata) == len(circuits)


def test_make_qiskit_counts():
    """make_qiskit_counts should convert tagged measurements into Qiskit counts."""
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qcdl_metadata = circuit_to_qcdl(qc)
    assert qcdl_metadata.clbit_to_tag == ["0"]

    result = Result(num_shots=3, measurements={"0": [np.array(["0", "1", "0"])]})
    counts = make_qiskit_counts(result, qcdl_metadata)
    assert counts == {"0": 2, "1": 1}


def test_make_qiskit_counts_rejects_aggregated_metadata():
    """make_qiskit_counts requires per-circuit metadata, not a concatenated
    QCDL's aggregated metadata (whose clbit_to_tag is None)."""
    qc1 = QuantumCircuit(1, 1)
    qc1.measure(0, 0)
    qc2 = QuantumCircuit(1, 1)
    qc2.measure(0, 0)
    aggregated = concatenate_circuits_to_qcdl([qc1, qc2])

    with pytest.raises(ValueError, match="clbit_to_tag"):
        make_qiskit_counts(Result(num_shots=1), aggregated)


def test_make_qiskit_counts_with_loose_clbits():
    """Circuits with loose clbits (no classical register, empty creg_sizes)
    must not have every bitstring collapsed into a single '' key."""
    qc = QuantumCircuit([Qubit()], [Clbit()])
    qc.measure(0, 0)
    qcdl_metadata = circuit_to_qcdl(qc)
    assert qcdl_metadata.qiskit_header["creg_sizes"] == []

    result = Result(num_shots=3, measurements={"0": [np.array(["0", "1", "0"])]})
    counts = make_qiskit_counts(result, qcdl_metadata)
    assert counts == {"0": 2, "1": 1}


def test_make_qiskit_counts_with_multiple_registers():
    """Bitstrings must be split into one space-joined key per shot, with a
    substring per register, ordered last-declared register first."""
    qr = QuantumRegister(2, "qr")
    cr0 = ClassicalRegister(1, "c")
    cr1 = ClassicalRegister(1, "d")
    qc = QuantumCircuit(qr, cr0, cr1)
    qc.measure(qr[0], cr0[0])
    qc.measure(qr[1], cr1[0])

    qcdl_metadata = circuit_to_qcdl(qc)
    assert qcdl_metadata.clbit_to_tag == ["0", "1"]

    result = Result(
        num_shots=2,
        measurements={"0": [np.array(["1", "0"])], "1": [np.array(["0", "1"])]},
    )
    counts = make_qiskit_counts(result, qcdl_metadata)
    assert counts == {"0 1": 1, "1 0": 1}

