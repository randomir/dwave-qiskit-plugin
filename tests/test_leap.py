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

import io
import json
import unittest.mock as mock

import pytest

from dwave.cloud import Client
from dwave.cloud.exceptions import (
    CanceledFutureError,
    ConfigFileError,
    SolverError,
    SolverFailureError,
    SolverNotFoundError,
)
from dwave.cloud.solver import QCDLSolver
from dwave.cloud.testing.mocks import qcdl_solver_data
from dwave.gate.results import YieldHandling

from qiskit import QuantumCircuit
from qiskit.providers import JobError, JobStatus, JobTimeoutError
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.result import MeasLevel

from dwave.plugins.qiskit import DWaveProvider
from dwave.plugins.qiskit.leap import QCDLJob, QCDLResult, QCDLSimulatorBackend
from dwave.plugins.qiskit.leap.backend import _QCDL_STANDARD_GATE_NAMES
from dwave.plugins.qiskit.leap.job import _future_status


def make_solver(**kwargs) -> QCDLSolver:
    # TODO: update the `qcdl_solver_data` mock solver data generator
    kwargs.setdefault("category", "software-gate")
    kwargs.setdefault("default_noise_model", False)
    kwargs.setdefault("default_qpu", "DRsim_21qubits")
    kwargs.setdefault("default_repeat_until_shots_requested", False)
    kwargs.setdefault("default_shots", 1000)
    kwargs.setdefault("default_time_limit_s", 2700)
    kwargs.setdefault("default_transpile", True)
    kwargs.setdefault("maximum_num_qubits", 21)
    kwargs.setdefault("maximum_shots", 1000000)
    kwargs.setdefault("maximum_time_limit_s", 2700)
    kwargs.setdefault("minimum_shots", 1)
    kwargs.setdefault("minimum_time_limit_s", 1)
    kwargs.setdefault("supported_qpu_strings", ["DRsim_17qubits", "DRsim_21qubits"])
    kwargs.setdefault("parameters", dict(noise_model='', qpu='', repeat_until_shots_requested='',
                                         shots='', time_limit='', transpile=''))

    return QCDLSolver(client=mock.Mock(), data=qcdl_solver_data(**kwargs))


def make_answer(num_shots: int, tag_to_bits: dict[str, tuple[int, list[str]]],
                num_qubits: int) -> dict:
    """Fabricate a decoded QCDL solver answer.

    Args:
        num_shots: Number of shots.
        tag_to_bits: Maps a measurement tag to the measured qubit's index and
            its per-shot values.
        num_qubits: Total number of qubits in the QCDL program.

    Returns:
        The answer dict, serializable to the solver answer JSON.
    """
    measurements = {}
    for tag, (qubit, bits) in tag_to_bits.items():
        per_qubit: list[list[str]] = [[] for _ in range(num_qubits)]
        per_qubit[qubit] = bits
        measurements[tag] = per_qubit
    return {"num_shots": num_shots, "measurements": measurements}


class StubFuture:
    """Minimal stand-in for :class:`dwave.cloud.computation.Future`."""

    def __init__(self, *, answer: dict | None = None, remote_status: str | None = None,
                 done: bool = False, exc: Exception | None = None, id: str = "problem-1"):
        self.id = id
        self.label = None
        self.remote_status = remote_status
        self._done = done
        self._exc = exc
        self._answer = answer
        self.cancel_called = False

    def done(self) -> bool:
        return self._done

    def wait(self, timeout: float | None = None) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancel_called = True

    def exception(self) -> None:
        if self._exc is not None:
            raise self._exc

    @property
    def answer_data(self) -> io.BytesIO:
        return io.BytesIO(json.dumps(self._answer).encode())


def done_future(answer: dict, **kwargs) -> StubFuture:
    return StubFuture(answer=answer, done=True, remote_status="COMPLETED", **kwargs)


def patch_sample_qcdl(monkeypatch, solver: QCDLSolver, futures: list[StubFuture]) -> list:
    """Replace ``solver.sample_qcdl`` with a recorder returning canned futures."""
    calls = []
    futures_iter = iter(futures)

    def sample_qcdl(qcdl, label=None, **params):
        calls.append({"qcdl": qcdl, "label": label, "params": params})
        return next(futures_iter)

    monkeypatch.setattr(solver, "sample_qcdl", sample_qcdl)
    return calls


def bell_circuit(name: str = "bell") -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2, name=name)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


# provider


def test_provider_lists_qcdl_backends():
    client = mock.Mock()
    client.get_solvers.return_value = [make_solver()]

    with mock.patch("dwave.plugins.qiskit.leap.provider.Client") as client_cls:
        client_cls.from_config.return_value = client
        provider = DWaveProvider(token="secret")

        client_cls.from_config.assert_not_called()  # client is created lazily

        backends = provider.backends()

    client_cls.from_config.assert_called_once()
    call_kwargs = client_cls.from_config.call_args.kwargs
    assert call_kwargs["client"] == "base"
    assert call_kwargs["connection_close"] is True
    assert call_kwargs["token"] == "secret"
    solver_filters = client.get_solvers.call_args.kwargs
    assert solver_filters["supported_problem_types__contains"] == "qcdl"
    assert solver_filters["category"] == "software-gate"
    assert solver_filters["order_by"] == "-properties.version"

    assert len(backends) == 1
    assert isinstance(backends[0], QCDLSimulatorBackend)
    assert backends[0].name == "qcdl_mock_solver"
    assert backends[0].provider is provider


def test_provider_backends_name_filter():
    client = mock.Mock()
    client.get_solvers.return_value = []

    with mock.patch("dwave.plugins.qiskit.leap.provider.Client") as client_cls:
        client_cls.from_config.return_value = client
        DWaveProvider().backends(name="some_solver")

    assert client.get_solvers.call_args.kwargs["name"] == "some_solver"


def test_provider_backends_property_filtering_works():
    solvers = [
        make_solver(name="dr17", maximum_num_qubits=17),
        make_solver(name="dr21", maximum_num_qubits=21),
    ]
    client = Client(endpoint='mock', token='mock')

    def _fetch_solvers(**kwargs):
        if kwargs.pop('name', None) not in ['dr17', 'dr21', None]:
            raise SolverNotFoundError
        return solvers
    client._fetch_solvers = _fetch_solvers

    with mock.patch("dwave.plugins.qiskit.leap.provider.Client") as client_cls:
        client_cls.from_config.return_value = client

        # name filter
        backends = DWaveProvider().backends(name="dr17")
        assert len(backends) == 1
        assert backends[0].num_qubits == 17

        # custom property filter
        backends = DWaveProvider().backends(maximum_num_qubits=21)
        assert len(backends) == 1
        assert backends[0].name == "dr21"

        backend = DWaveProvider().get_backend(maximum_num_qubits=21)
        assert backend.name == "dr21"

        # solver/backend not found
        backends = DWaveProvider().backends(name="non-existing")
        assert len(backends) == 0

        with pytest.raises(QiskitBackendNotFoundError):
            DWaveProvider().get_backend(name="non-existing")

        with pytest.raises(QiskitBackendNotFoundError):
            DWaveProvider().get_backend(maximum_num_qubits=100)


def test_get_backend_raises_when_none_match():
    client = mock.Mock()
    client.get_solvers.return_value = []

    with mock.patch("dwave.plugins.qiskit.leap.provider.Client") as client_cls:
        client_cls.from_config.return_value = client
        with pytest.raises(QiskitBackendNotFoundError):
            DWaveProvider().get_backend()


def test_get_backend_returns_first_of_many():
    solvers = [
        make_solver(name="qcdl_solver_old", version="0.1"),
        make_solver(name="qcdl_solver_new", version="0.2"),
    ]
    client = Client(endpoint='mock', token='mock')
    client._fetch_solvers = lambda **kwargs: solvers

    with mock.patch("dwave.plugins.qiskit.leap.provider.Client") as client_cls:
        client_cls.from_config.return_value = client
        backend = DWaveProvider().get_backend()

    assert backend.name == "qcdl_solver_new"


def test_provider_context_manager_closes_client():
    client = mock.Mock()
    client.get_solvers.return_value = []

    with mock.patch("dwave.plugins.qiskit.leap.provider.Client") as client_cls:
        client_cls.from_config.return_value = client
        with DWaveProvider() as provider:
            provider.backends()
        client.close.assert_called_once()

        # a closed provider creates a fresh client on next use
        provider.backends()
        assert client_cls.from_config.call_count == 2


# backend


def test_backend_rejects_non_simulator_solver():
    with pytest.raises(ValueError, match="not a gate-model simulator"):
        QCDLSimulatorBackend(make_solver(category="hybrid"))


def test_target_gates_and_connectivity():
    backend = QCDLSimulatorBackend(make_solver())
    target = backend.target

    assert set(target.operation_names) == set(_QCDL_STANDARD_GATE_NAMES) | {"measure"}
    assert target.num_qubits == 21
    # properties=None instructions are global, i.e. all-to-all
    assert target.qargs is None


def test_target_num_qubits_from_solver():
    backend = QCDLSimulatorBackend(make_solver(maximum_num_qubits=8))
    assert backend.target.num_qubits == 8


def _get_default_solver_options():
    return {
        "shots": 1000, "time_limit": 2700,
        "noise_model": False, "qpu": "DRsim_21qubits",
        "transpile": True, "repeat_until_shots_requested": False,
    }

def _get_default_options():
    return _get_default_solver_options() | {
        "label": None, "pack_qcdls": True, "qcdl_pack_target": 0.4,
        "yield_handling": YieldHandling.only_post_selected_counts,
    }

def test_default_options():
    backend = QCDLSimulatorBackend(make_solver())
    assert dict(backend.options) == _get_default_options()


def test_shots_validated_against_max_shots(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())  # mock solver has max_shots=1M
    patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    with pytest.raises(ValueError, match="shots"):
        backend.run(bell_circuit(), shots=2000000)


def test_unknown_run_option_rejected():
    backend = QCDLSimulatorBackend(make_solver())
    with pytest.raises(AttributeError, match="num_reads"):
        backend.run(bell_circuit(), num_reads=100)


def test_extra_run_options(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    with pytest.raises(RuntimeError, match="memory"):
        backend.run(bell_circuit(), memory=1)

    with pytest.raises(RuntimeError, match="meas_level"):
        backend.run(bell_circuit(), meas_level=MeasLevel.RAW)

    job = backend.run(bell_circuit(), meas_level=MeasLevel.CLASSIFIED)
    assert isinstance(job, QCDLJob)


def test_run_rejects_non_circuit_input():
    backend = QCDLSimulatorBackend(make_solver())
    with pytest.raises(TypeError):
        backend.run("not a circuit")


def test_run_single_circuit(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    job = backend.run(bell_circuit(), shots=100)

    assert isinstance(job, QCDLJob)
    assert job.backend() is backend
    assert len(calls) == 1
    assert calls[0]["params"] == _get_default_solver_options() | {"shots": 100}
    assert calls[0]["label"].startswith(f"qiskit:{job.job_id()}:")
    assert isinstance(calls[0]["qcdl"], dict)  # the submittable QCDL program


def test_run_passes_time_limit_and_label(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    backend.run(bell_circuit(), time_limit=5, label="my-label")

    assert calls[0]["params"] == _get_default_solver_options() | {"time_limit": 5}
    assert calls[0]["label"] == "my-label"


def test_run_packs_circuits_by_default(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    backend.run([bell_circuit("bell1"), bell_circuit("bell2")])

    assert len(calls) == 1  # both circuits packed into one QCDL


def test_run_without_packing(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture(), StubFuture()])

    backend.run([bell_circuit("bell1"), bell_circuit("bell2")], pack_qcdls=False)

    assert len(calls) == 2  # one QCDL per circuit


# job status


def test_submit_raises():
    job = QCDLJob(backend=None, job_id="id", futures=[], qcdls=[])
    with pytest.raises(JobError):
        job.submit()


@pytest.mark.parametrize("future,expected", [
    (StubFuture(remote_status=None), JobStatus.INITIALIZING),
    (StubFuture(remote_status="PENDING"), JobStatus.QUEUED),
    (StubFuture(remote_status="IN_PROGRESS"), JobStatus.RUNNING),
    (StubFuture(done=True, remote_status="COMPLETED"), JobStatus.DONE),
    (StubFuture(done=True, remote_status="FAILED",
                exc=SolverFailureError("failed")), JobStatus.ERROR),
    (StubFuture(done=True, remote_status="CANCELLED",
                exc=CanceledFutureError()), JobStatus.CANCELLED),
    (StubFuture(done=True, remote_status="CANCELLED"), JobStatus.CANCELLED),
])
def test_future_status_mapping(future, expected):
    assert _future_status(future) is expected


@pytest.mark.parametrize("statuses,expected", [
    (["COMPLETED", "COMPLETED"], JobStatus.DONE),
    (["COMPLETED", "IN_PROGRESS"], JobStatus.RUNNING),
    (["PENDING", "IN_PROGRESS"], JobStatus.RUNNING),
    (["PENDING", None], JobStatus.QUEUED),
    ([None, None], JobStatus.INITIALIZING),
])
def test_job_status_aggregation(statuses, expected):
    futures = [
        StubFuture(remote_status=status, done=(status == "COMPLETED"))
        for status in statuses
    ]
    job = QCDLJob(backend=None, job_id="id", futures=futures, qcdls=[])
    assert job.status() is expected


def test_job_status_error_takes_precedence():
    futures = [
        StubFuture(done=True, remote_status="COMPLETED"),
        StubFuture(done=True, remote_status="FAILED", exc=SolverFailureError("failed")),
    ]
    job = QCDLJob(backend=None, job_id="id", futures=futures, qcdls=[])
    assert job.status() is JobStatus.ERROR


def test_cancel_fans_out():
    futures = [StubFuture(), StubFuture()]
    job = QCDLJob(backend=None, job_id="id", futures=futures, qcdls=[])
    job.cancel()
    assert all(future.cancel_called for future in futures)


# job result


def test_result_single_circuit(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    answer = make_answer(4, {"0": (0, ["0", "1", "0", "1"]),
                             "1": (1, ["0", "1", "0", "1"])}, num_qubits=2)
    patch_sample_qcdl(monkeypatch, backend.solver, [done_future(answer)])

    circuit = bell_circuit()
    result = backend.run(circuit).result()

    assert result.success
    assert result.backend_name == "qcdl_mock_solver"
    assert result.get_counts(circuit) == {"00": 2, "11": 2}
    assert result.results[0].shots == 4
    assert result.results[0].header["name"] == "bell"


def test_result_circuits_packed_in_one_qcdl(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    # tags are globally unique across the packed circuits: bell1 -> 0/1, bell2 -> 2/3
    answer = make_answer(4, {"0": (0, ["0", "1", "0", "1"]),
                             "1": (1, ["0", "1", "0", "1"]),
                             "2": (0, ["1", "1", "1", "1"]),
                             "3": (1, ["1", "1", "1", "1"])}, num_qubits=2)
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [done_future(answer)])

    result = backend.run([bell_circuit("bell1"), bell_circuit("bell2")]).result()

    assert len(calls) == 1
    assert [res.header["name"] for res in result.results] == ["bell1", "bell2"]
    assert result.get_counts("bell1") == {"00": 2, "11": 2}
    assert result.get_counts("bell2") == {"11": 4}


def test_result_header_carries_circuit_metadata(monkeypatch):
    # qiskit-experiments reads each experiment's `header["metadata"]`, along
    # with `shots` and `meas_level`
    backend = QCDLSimulatorBackend(make_solver())
    answer = make_answer(4, {"0": (0, ["0", "1", "0", "1"]),
                             "1": (1, ["0", "1", "0", "1"]),
                             "2": (0, ["1", "1", "1", "1"]),
                             "3": (1, ["1", "1", "1", "1"])}, num_qubits=2)
    patch_sample_qcdl(monkeypatch, backend.solver, [done_future(answer)])

    bell1 = bell_circuit("bell1")
    bell1.metadata = {"xval": 0.1, "composite_index": [0]}
    bell2 = bell_circuit("bell2")   # no metadata set

    result = backend.run([bell1, bell2]).result()

    header = result.results[0].header
    assert header["name"] == "bell1"
    assert header["memory_slots"] == 2
    assert header["creg_sizes"] == [["c", 2]]
    assert header["metadata"] == {"xval": 0.1, "composite_index": [0]}
    assert result.results[1].header["metadata"] == {}
    assert result.results[0].meas_level == MeasLevel.CLASSIFIED

    # circuit metadata lives in the header only, not in the experiment data
    assert "metadata" not in result.data(0)


def test_result_multiple_qcdls(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    answer1 = make_answer(2, {"0": (0, ["0", "0"]), "1": (1, ["0", "0"])}, num_qubits=2)
    answer2 = make_answer(2, {"0": (0, ["1", "1"]), "1": (1, ["1", "1"])}, num_qubits=2)
    patch_sample_qcdl(
        monkeypatch, backend.solver, [done_future(answer1), done_future(answer2)]
    )

    job = backend.run(
        [bell_circuit("bell1"), bell_circuit("bell2")], pack_qcdls=False
    )
    result = job.result()

    # experiments follow circuit submission order across QCDLs
    assert [res.header["name"] for res in result.results] == ["bell1", "bell2"]
    assert result.get_counts("bell1") == {"00": 2}
    assert result.get_counts("bell2") == {"11": 2}


def test_result_cached(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    answer = make_answer(2, {"0": (0, ["0", "0"]), "1": (1, ["0", "0"])}, num_qubits=2)
    patch_sample_qcdl(monkeypatch, backend.solver, [done_future(answer)])

    job = backend.run(bell_circuit())
    assert job.result() is job.result()


def test_result_raises_on_failed_problem(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    future = StubFuture(done=True, remote_status="FAILED",
                        exc=SolverFailureError("solver blew up"))
    patch_sample_qcdl(monkeypatch, backend.solver, [future])

    job = backend.run(bell_circuit())
    assert job.status() is JobStatus.ERROR
    with pytest.raises(JobError, match="solver blew up"):
        job.result()


def test_result_timeout(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture(done=False)])

    job = backend.run(bell_circuit())
    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01)


def make_splat_answer() -> dict:
    """An answer where noise_model=True marked one measurement with a splat."""
    return make_answer(4, {"0": (0, ["0", "1", "*", "1"]),
                           "1": (1, ["0", "1", "0", "1"])}, num_qubits=2)


def test_result_post_selects_splats_by_default(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    patch_sample_qcdl(monkeypatch, backend.solver, [done_future(make_splat_answer())])

    result = backend.run(bell_circuit(), noise_model=True).result()

    assert isinstance(result, QCDLResult)
    assert result.get_counts() == {"00": 1, "11": 2}
    # the unresolved counts and the yield stay available in the experiment data
    assert result.data(0)["raw_counts"] == {"00": 1, "11": 2, "0*": 1}
    assert result.data(0)["post_selection_yield"] == 0.75


def test_result_keeps_splats_when_ignored(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    patch_sample_qcdl(monkeypatch, backend.solver, [done_future(make_splat_answer())])

    job = backend.run(bell_circuit(), noise_model=True,
                      yield_handling=YieldHandling.ignore_splats)
    counts = job.result().get_counts()

    assert counts == {"00": 1, "11": 2, "0*": 1}


def test_get_counts_yield_handling_override(monkeypatch):
    backend = QCDLSimulatorBackend(make_solver())
    patch_sample_qcdl(monkeypatch, backend.solver, [done_future(make_splat_answer())])

    # yield_handling is also accepted by name
    result = backend.run(bell_circuit(), noise_model=True,
                         yield_handling="ignore_splats").result()

    assert result.get_counts() == {"00": 1, "11": 2, "0*": 1}
    renormalized = result.get_counts(
        yield_handling=YieldHandling.renormalize_distribution)
    assert renormalized == {"00": pytest.approx(4 / 3), "11": pytest.approx(8 / 3)}


def test_run_rejects_invalid_yield_handling():
    backend = QCDLSimulatorBackend(make_solver())
    with pytest.raises(ValueError, match="yield_handling"):
        backend.run(bell_circuit(), yield_handling="not_a_strategy")


# live tests (skipped unless Leap access is configured)


def test_live_bell_circuit():
    try:
        with DWaveProvider() as provider:
            try:
                backend = provider.get_backend()
            except QiskitBackendNotFoundError:
                pytest.skip("no QCDL solver available")

            job = backend.run(bell_circuit(), shots=100)
            counts = job.result().get_counts()
    except (ValueError, ConfigFileError, SolverError) as exc:
        pytest.skip(f"no Leap access configured: {exc}")

    assert sum(counts.values()) == 100
    assert set(counts) <= {"00", "01", "10", "11"}
