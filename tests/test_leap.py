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
)
from dwave.cloud.solver import QCDLSolver
from dwave.cloud.testing.mocks import qcdl_solver_data

from qiskit import QuantumCircuit
from qiskit.providers import JobError, JobStatus, JobTimeoutError
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from dwave.plugins.qiskit import DWaveProvider
from dwave.plugins.qiskit.leap import QCDLBackend, QCDLJob
from dwave.plugins.qiskit.leap.backend import _QCDL_STANDARD_GATE_NAMES
from dwave.plugins.qiskit.leap.job import _future_status


def make_solver(**kwargs) -> QCDLSolver:
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
    assert call_kwargs["token"] == "secret"
    assert client.get_solvers.call_args.kwargs["supported_problem_types__contains"] == "qcdl"

    assert len(backends) == 1
    assert isinstance(backends[0], QCDLBackend)
    assert backends[0].name == "qcdl_mock_solver"
    assert backends[0].provider is provider


def test_provider_backends_name_filter():
    client = mock.Mock()
    client.get_solvers.return_value = []

    with mock.patch("dwave.plugins.qiskit.leap.provider.Client") as client_cls:
        client_cls.from_config.return_value = client
        DWaveProvider().backends(name="some_solver")

    assert client.get_solvers.call_args.kwargs["name"] == "some_solver"


@pytest.mark.parametrize("num_solvers", [0, 2])
def test_get_backend_requires_single_match(num_solvers):
    client = mock.Mock()
    client.get_solvers.return_value = [make_solver() for _ in range(num_solvers)]

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


def test_target_gates_and_connectivity():
    backend = QCDLBackend(make_solver())
    target = backend.target

    assert set(target.operation_names) == set(_QCDL_STANDARD_GATE_NAMES) | {"measure"}
    assert target.num_qubits is None
    # properties=None instructions are global, i.e. all-to-all
    assert target.qargs is None


def test_target_num_qubits_from_solver():
    backend = QCDLBackend(make_solver(num_qubits=8))
    assert backend.target.num_qubits == 8


def test_default_options():
    backend = QCDLBackend(make_solver())
    assert dict(backend.options) == {
        "shots": 1024, "time_limit": None, "label": None, "qcdl_pack_target": True,
    }


def test_shots_validated_against_max_shots(monkeypatch):
    backend = QCDLBackend(make_solver())  # mock solver has max_shots=10000
    patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    with pytest.raises(ValueError, match="shots"):
        backend.run(bell_circuit(), shots=20000)


def test_unknown_run_option_rejected():
    backend = QCDLBackend(make_solver())
    with pytest.raises(AttributeError, match="num_reads"):
        backend.run(bell_circuit(), num_reads=100)


def test_run_rejects_non_circuit_input():
    backend = QCDLBackend(make_solver())
    with pytest.raises(TypeError):
        backend.run("not a circuit")


def test_run_single_circuit(monkeypatch):
    backend = QCDLBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    job = backend.run(bell_circuit(), shots=100)

    assert isinstance(job, QCDLJob)
    assert job.backend() is backend
    assert len(calls) == 1
    assert calls[0]["params"] == {"shots": 100}
    assert calls[0]["label"].startswith(f"qiskit:{job.job_id()}:")
    assert isinstance(calls[0]["qcdl"], dict)  # the submittable QCDL program


def test_run_passes_time_limit_and_label(monkeypatch):
    backend = QCDLBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    backend.run(bell_circuit(), time_limit=5, label="my-label")

    assert calls[0]["params"] == {"shots": 1024, "time_limit": 5}
    assert calls[0]["label"] == "my-label"


def test_run_packs_circuits_by_default(monkeypatch):
    backend = QCDLBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture()])

    backend.run([bell_circuit("bell1"), bell_circuit("bell2")])

    assert len(calls) == 1  # both circuits packed into one QCDL


def test_run_without_packing(monkeypatch):
    backend = QCDLBackend(make_solver())
    calls = patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture(), StubFuture()])

    backend.run([bell_circuit("bell1"), bell_circuit("bell2")], qcdl_pack_target=False)

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
    backend = QCDLBackend(make_solver())
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
    backend = QCDLBackend(make_solver())
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


def test_result_multiple_qcdls(monkeypatch):
    backend = QCDLBackend(make_solver())
    answer1 = make_answer(2, {"0": (0, ["0", "0"]), "1": (1, ["0", "0"])}, num_qubits=2)
    answer2 = make_answer(2, {"0": (0, ["1", "1"]), "1": (1, ["1", "1"])}, num_qubits=2)
    patch_sample_qcdl(
        monkeypatch, backend.solver, [done_future(answer1), done_future(answer2)]
    )

    job = backend.run(
        [bell_circuit("bell1"), bell_circuit("bell2")], qcdl_pack_target=False
    )
    result = job.result()

    # experiments follow circuit submission order across QCDLs
    assert [res.header["name"] for res in result.results] == ["bell1", "bell2"]
    assert result.get_counts("bell1") == {"00": 2}
    assert result.get_counts("bell2") == {"11": 2}


def test_result_cached(monkeypatch):
    backend = QCDLBackend(make_solver())
    answer = make_answer(2, {"0": (0, ["0", "0"]), "1": (1, ["0", "0"])}, num_qubits=2)
    patch_sample_qcdl(monkeypatch, backend.solver, [done_future(answer)])

    job = backend.run(bell_circuit())
    assert job.result() is job.result()


def test_result_raises_on_failed_problem(monkeypatch):
    backend = QCDLBackend(make_solver())
    future = StubFuture(done=True, remote_status="FAILED",
                        exc=SolverFailureError("solver blew up"))
    patch_sample_qcdl(monkeypatch, backend.solver, [future])

    job = backend.run(bell_circuit())
    assert job.status() is JobStatus.ERROR
    with pytest.raises(JobError, match="solver blew up"):
        job.result()


def test_result_timeout(monkeypatch):
    backend = QCDLBackend(make_solver())
    patch_sample_qcdl(monkeypatch, backend.solver, [StubFuture(done=False)])

    job = backend.run(bell_circuit())
    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01)


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
