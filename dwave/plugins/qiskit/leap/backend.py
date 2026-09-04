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

"""Qiskit backend for D-Wave Leap QCDL solvers."""

from __future__ import annotations

import copy
import uuid
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from dwave.plugins.qiskit.leap.job import QCDLJob
from dwave.plugins.qiskit.qcdl.translators import circuits_to_qcdls

if TYPE_CHECKING:
    from dwave.cloud.solver import QCDLSolver

    from dwave.plugins.qiskit.leap.provider import DWaveProvider

__all__ = ["QCDLBackend"]

# gates the QCDL translators support that are also Qiskit standard gates
# (the translator-only sy/sydg/mced have no Qiskit standard counterpart)
_QCDL_STANDARD_GATE_NAMES = (
    "x", "sx", "y", "z", "s", "sdg", "t", "tdg", "h",
    "rx", "ry", "rz", "p", "u",
    "swap", "cx", "cy", "cz", "crx", "cry", "crz", "cp", "cu",
    "rxx", "ryy", "rzz",
)


class QCDLBackend(BackendV2):
    """A Qiskit backend running circuits on a D-Wave Leap QCDL solver.

    Args:
        solver: The Leap QCDL solver the backend submits problems to.
        provider: The provider the backend was obtained from.

    Raises:
        ValueError: If the solver is not a QCDL solver.
    """

    def __init__(self, solver: QCDLSolver, provider: DWaveProvider | None = None):
        if 'qcdl' not in solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'qcdl' problem type.")

        super().__init__(
            provider=provider,
            name=solver.name,
            description=f"D-Wave Leap QCDL solver {solver.name}",
            backend_version=solver.properties.get("version"),
        )
        self._solver = solver
        self._target: Target | None = None

        max_shots = solver.properties.get("max_shots")
        if max_shots:
            self.options.set_validator("shots", (1, int(max_shots)))

    @property
    def solver(self) -> QCDLSolver:
        """The Leap QCDL solver the backend submits problems to."""
        return self._solver

    @property
    def target(self) -> Target:
        if self._target is None:
            self._target = self._build_target()
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024, time_limit=None, label=None, qcdl_pack_target=True)

    def _build_target(self) -> Target:
        target = Target(
            description=f"Target for {self.name}",
            # None means unconstrained (no qubit limit advertised by the solver)
            num_qubits=self._solver.properties.get("num_qubits"),
        )
        gate_mapping = get_standard_gate_name_mapping()
        for gate_name in _QCDL_STANDARD_GATE_NAMES:
            # properties=None makes the instruction global (all-to-all)
            target.add_instruction(gate_mapping[gate_name], properties=None, name=gate_name)
        target.add_instruction(Measure(), properties=None, name="measure")
        return target

    def run(
        self, run_input: QuantumCircuit | list[QuantumCircuit], **options
    ) -> QCDLJob:
        """Translate circuits to QCDL and submit them to the solver.

        Args:
            run_input: A circuit, or list of circuits, to run.
            **options: Overrides of the backend's :attr:`options` for this run
                (``shots``, ``time_limit``, ``label``, ``qcdl_pack_target``).

        Returns:
            The job wrapping the submitted QCDL problems.
        """
        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, (list, tuple)) and all(
            isinstance(circuit, QuantumCircuit) for circuit in run_input
        ):
            circuits = list(run_input)
        else:
            raise TypeError(
                "run() accepts a QuantumCircuit or a list of QuantumCircuit, "
                f"not {type(run_input).__name__}"
            )

        unknown = set(options) - set(self.options)
        if unknown:
            raise AttributeError(
                f"options not valid for this backend: {', '.join(sorted(unknown))}"
            )
        opts = copy.deepcopy(self.options)
        opts.update_options(**options)

        job_id = uuid.uuid4().hex

        params = {"shots": opts.shots}
        if opts.time_limit is not None:
            params["time_limit"] = opts.time_limit

        qcdls = list(
            circuits_to_qcdls(
                circuits, job_id=job_id, qcdl_pack_target=opts.qcdl_pack_target
            )
        )
        futures = [
            self._solver.sample_qcdl(
                qcdl.qcdl,
                label=opts.label or f"qiskit:{job_id}:{qcdl.job_name}",
                **params,
            )
            for qcdl in qcdls
        ]

        return QCDLJob(backend=self, job_id=job_id, futures=futures, qcdls=qcdls)
