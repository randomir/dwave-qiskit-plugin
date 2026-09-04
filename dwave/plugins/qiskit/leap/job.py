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

"""Qiskit job wrapping problems submitted to a D-Wave Leap QCDL solver."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dwave.cloud.api.constants import ProblemStatus
from dwave.cloud.exceptions import CanceledFutureError
from dwave.gate.results import Result as GateResult

from qiskit.providers import JobError, JobStatus, JobTimeoutError, JobV1
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from dwave.plugins.qiskit.qcdl.translators import QCDLWithMetadata, make_qiskit_counts

if TYPE_CHECKING:
    from dwave.cloud.computation import Future

    from dwave.plugins.qiskit.leap.backend import QCDLSimulatorBackend

__all__ = ["QCDLJob"]

_STATUS_MAP = {
    ProblemStatus.PENDING.value: JobStatus.QUEUED,
    ProblemStatus.IN_PROGRESS.value: JobStatus.RUNNING,
    ProblemStatus.COMPLETED.value: JobStatus.DONE,
    ProblemStatus.FAILED.value: JobStatus.ERROR,
    ProblemStatus.CANCELLED.value: JobStatus.CANCELLED,
}


def _future_status(future: Future) -> JobStatus:
    """Map a cloud-client future's state onto a Qiskit job status."""
    if future.done():
        try:
            # Future.exception() raises the stored exception, if any
            future.exception()
        except CanceledFutureError:
            return JobStatus.CANCELLED
        except Exception:
            return JobStatus.ERROR
        if future.remote_status == ProblemStatus.CANCELLED.value:
            return JobStatus.CANCELLED
        return JobStatus.DONE

    # remote_status is None until the first status poll comes back
    return _STATUS_MAP.get(future.remote_status, JobStatus.INITIALIZING)


def _per_circuit_metadata(qcdl: QCDLWithMetadata) -> list[QCDLWithMetadata]:
    """List the per-circuit metadata entries of a translated QCDL, in circuit order."""
    if qcdl.clbit_to_tag is not None:
        # a single-circuit QCDL carries its own (per-circuit) metadata
        return [qcdl]
    return qcdl.circuit_metadata


class QCDLJob(JobV1):
    """A Qiskit job for circuits submitted to a D-Wave Leap QCDL solver.

    A single job may span multiple QCDL problems when the circuits given to
    :meth:`.QCDLSimulatorBackend.run` are translated into more than one QCDL
    program.

    Args:
        backend: The backend the job was submitted through.
        job_id: A unique identifier for the job.
        futures: The in-flight cloud-client problems, one per QCDL program.
        qcdls: The translated QCDL programs, positionally matching ``futures``.
    """

    _async = True

    def __init__(
        self,
        backend: QCDLSimulatorBackend,
        job_id: str,
        futures: list[Future],
        qcdls: list[QCDLWithMetadata],
    ):
        super().__init__(backend, job_id)
        self._futures = futures
        self._qcdls = qcdls
        self._result: Result | None = None

    def submit(self) -> None:
        """Unsupported; problems are submitted when the job is created by ``run()``."""
        raise JobError("job has already been submitted")

    def status(self) -> JobStatus:
        """Return the aggregate status of all the job's QCDL problems."""
        statuses = [_future_status(future) for future in self._futures]
        for status in (JobStatus.ERROR, JobStatus.CANCELLED):
            if status in statuses:
                return status
        if all(status is JobStatus.DONE for status in statuses):
            return JobStatus.DONE
        for status in (JobStatus.RUNNING, JobStatus.QUEUED):
            if status in statuses:
                return status
        return JobStatus.INITIALIZING

    def cancel(self) -> None:
        """Request cancellation of all the job's QCDL problems.

        Cancellation is best-effort; check :meth:`status` for the outcome.
        """
        for future in self._futures:
            future.cancel()

    def result(self, timeout: float | None = None) -> Result:
        """Wait for all the job's QCDL problems and assemble their results.

        Args:
            timeout: Total number of seconds to wait for the results.

        Returns:
            A :class:`~qiskit.result.Result` with one experiment per submitted
            circuit, in submission order.

        Raises:
            JobTimeoutError: If the results don't arrive within ``timeout``.
            JobError: If any of the QCDL problems failed or was cancelled.
        """
        if self._result is None:
            self._result = self._build_result(timeout=timeout)
        return self._result

    def _build_result(self, timeout: float | None) -> Result:
        deadline = None if timeout is None else time.monotonic() + timeout
        experiments = []

        for future, qcdl in zip(self._futures, self._qcdls):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not future.wait(timeout=remaining):
                raise JobTimeoutError(
                    f"timed out waiting for results of QCDL problem {future.id!r}"
                )
            try:
                future.exception()
            except Exception as exc:
                raise JobError(f"QCDL problem {future.id!r} failed: {exc}") from exc

            answer = future.answer_data
            answer.seek(0)
            gate_result = GateResult.model_validate_json(answer.read())

            for metadata in _per_circuit_metadata(qcdl):
                experiments.append(
                    ExperimentResult(
                        shots=gate_result.num_shots,
                        success=True,
                        data=ExperimentResultData(
                            counts=make_qiskit_counts(gate_result, metadata)
                        ),
                        header=metadata.qiskit_header,
                    )
                )

        return Result(
            backend_name=self.backend().name,
            backend_version=self.backend().backend_version,
            job_id=self.job_id(),
            success=True,
            results=experiments,
            date=datetime.now(timezone.utc).isoformat(),
            status="COMPLETED",
        )
