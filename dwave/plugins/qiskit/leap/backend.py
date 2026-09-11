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
from collections.abc import Iterable
from functools import cached_property
from typing import Any, TYPE_CHECKING

from dwave.gate.results import YieldHandling

from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers import BackendV2, Options
from qiskit.result import MeasLevel
from qiskit.transpiler import Target

from dwave.plugins.qiskit.leap.job import QCDLJob
from dwave.plugins.qiskit.qcdl.translators import circuits_to_qcdls

if TYPE_CHECKING:
    from dwave.cloud.solver import QCDLSolver
    from dwave.plugins.qiskit.leap.provider import DWaveProvider

__all__ = ["QCDLSimulatorBackend"]

# gates the QCDL translators support that are also Qiskit standard gates
# (the translator-only sy/sydg/mced have no Qiskit standard counterpart)
_QCDL_STANDARD_GATE_NAMES = (
    "x", "sx", "y", "z", "s", "sdg", "t", "tdg", "h",
    "rx", "ry", "rz", "p", "u",
    "swap", "cx", "cy", "cz", "crx", "cry", "crz", "cp", "cu",
    "rxx", "ryy", "rzz",
)


class QCDLSimulatorBackend(BackendV2):
    """A Qiskit backend running circuits on a D-Wave Leap QCDL simulator solver.

    Args:
        solver:
            The Leap QCDL simulator solver the backend submits problems to.
        provider:
            The provider the backend was obtained from.

    Raises:
        ValueError: If the solver is not a QCDL simulator solver.
    """

    def __init__(self, solver: QCDLSolver, provider: DWaveProvider | None = None):
        if solver.properties.get("category") != "software-gate":
            raise ValueError("selected solver is not a gate-model simulator")
        if 'qcdl' not in solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'qcdl' problem type.")

        super().__init__(
            provider=provider,
            name=solver.name,
            description=f"D-Wave Leap QCDL simulator solver {solver.name}",
            backend_version=solver.properties.get("version"),
        )
        self._solver = solver
        self._target: Target | None = None

        # update options with actual solver defaults
        self._options.update_options(**self._solver_defaults)

        # configure solver parameter validators
        self.options.set_validator("noise_model", bool)
        self.options.set_validator("repeat_until_shots_requested", bool)
        self.options.set_validator("transpile", bool)

        # a list validator accepts both YieldHandling members and their names
        self.options.set_validator("yield_handling", list(YieldHandling))

        if supported_qpu_strings := solver.properties.get("supported_qpu_strings"):
            self.options.set_validator("qpu", list(supported_qpu_strings))

        min_shots = solver.properties.get("minimum_shots", 1)
        max_shots = solver.properties.get("maximum_shots", 1_000_000)
        self.options.set_validator("shots", (int(min_shots), int(max_shots)))

        min_time_limit = solver.properties.get("minimum_time_limit_s", 1)
        max_time_limit = solver.properties.get("maximum_time_limit_s", 2700)
        self.options.set_validator("time_limit", (int(min_time_limit), int(max_time_limit)))

    @property
    def solver(self) -> QCDLSolver:
        """The Leap QCDL simulator solver the backend submits problems to."""
        return self._solver

    @property
    def target(self) -> Target:
        if self._target is None:
            self._target = self._build_target()
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @cached_property
    def _solver_defaults(self) -> dict[str, Any]:
        """Default values of solver parameters."""
        properties = self.solver.properties.copy()
        defaults = {}
        for param in self.solver.parameters:
            default = properties.get(f"default_{param}", properties.get(f"default_{param}_s"))
            if default is not None:
                defaults[param] = default
        return defaults

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            noise_model=False,
            qpu=None,
            repeat_until_shots_requested=False,
            shots=1,
            time_limit=1,
            transpile=True,
            label=None,
            pack_qcdls=True,
            qcdl_pack_target=0.4,
            yield_handling=YieldHandling.only_post_selected_counts,
        )

    def _build_target(self) -> Target:
        target = Target(
            description=f"Target for {self.name}",
            # None means unconstrained (no qubit limit advertised by the solver)
            num_qubits=self._solver.properties.get("maximum_num_qubits"),
        )
        gate_mapping = get_standard_gate_name_mapping()
        for gate_name in _QCDL_STANDARD_GATE_NAMES:
            # properties=None makes the instruction global (all-to-all)
            target.add_instruction(gate_mapping[gate_name], properties=None, name=gate_name)
        target.add_instruction(Measure(), properties=None, name="measure")
        return target

    def run(
        self, run_input: QuantumCircuit | Iterable[QuantumCircuit], **options
    ) -> QCDLJob:
        """Translate circuits to QCDL and submit them to the solver.

        Args:
            run_input:
                A circuit, or an iterable of circuits, to run.
            **options:
                Overrides of the backend's :attr:`options` for this run
                (``shots``, ``time_limit``, ``repeat_until_shots_requested``,
                ``transpile``, ``qpu``, ``noise_model``, ``label``,
                ``pack_qcdls``, ``qcdl_pack_target``, ``yield_handling``).
                ``yield_handling``, a :class:`~dwave.gate.results.YieldHandling`
                member or its name, selects how splats reported when running
                with ``noise_model=True`` are resolved in the result counts.

        Returns:
            The job wrapping the submitted QCDL problems.
        """
        # TODO: link to param docs once published.

        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, Iterable) and all(
            isinstance(circuit, QuantumCircuit) for circuit in run_input
        ):
            circuits = list(run_input)
        else:
            raise TypeError(
                "run() accepts a QuantumCircuit or an iterable of QuantumCircuit items, "
                f"not {type(run_input).__name__}"
            )

        # we only support MeasLevel.CLASSIFIED, so don't fail if explicitly specified
        meas_level = options.pop('meas_level', None)
        if meas_level is not None and meas_level != MeasLevel.CLASSIFIED:
            raise RuntimeError(f"{meas_level=} is not supported by this backend")

        # clarify the failure
        if memory := options.pop('memory', None):
            raise RuntimeError(f"{memory=} is not supported by this backend")

        unknown = set(options) - set(self.options)
        if unknown:
            raise AttributeError(
                f"options not valid for this backend: {', '.join(sorted(unknown))}"
            )
        opts = copy.deepcopy(self.options)
        opts.update_options(**options)

        job_id = uuid.uuid4().hex

        params = {name: opts[name] for name in self.solver.parameters}

        qcdls = list(
            circuits_to_qcdls(
                circuits, job_id=job_id, pack_qcdls=opts.pack_qcdls,
                qcdl_pack_target=opts.qcdl_pack_target
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

        return QCDLJob(
            backend=self,
            job_id=job_id,
            futures=futures,
            qcdls=qcdls,
            yield_handling=YieldHandling.from_name(opts.yield_handling),
        )
