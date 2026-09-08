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

"""Qiskit result for jobs run on a D-Wave Leap QCDL solver."""

from __future__ import annotations

from dwave.gate.results import YieldHandling

from qiskit.exceptions import QiskitError
from qiskit.result import Counts, Result

__all__ = ["QCDLResult"]


class QCDLResult(Result):
    """A Qiskit result aware of splats in QCDL solver measurements.

    When a circuit runs with ``noise_model=True``, a measurement in which an
    error was detected is reported as a ``"*"`` (splat) instead of a bit value.
    Qiskit cannot represent splats in counts, so they are resolved with a
    :class:`~dwave.gate.results.YieldHandling` strategy: the counts stored in
    each experiment's ``data`` were resolved with the strategy the job ran
    with, while the unresolved counts are kept in ``data`` as ``raw_counts``
    (with the observed yield as ``post_selection_yield``).
    """

    def get_counts(
        self, experiment=None, yield_handling: YieldHandling | str | None = None
    ) -> Counts | list[Counts]:
        """Get the histogram data of an experiment, with splats resolved.

        Args:
            experiment: The experiment, as in
                :meth:`qiskit.result.Result.get_counts`.
            yield_handling: Strategy (or its name) used to re-resolve splats
                in the raw counts. Defaults to the counts already resolved
                with the strategy the job ran with.

        Returns:
            One counts dict per selected experiment. Note that counts are
            floats for the ``renormalize_distribution*`` strategies, and that
            keys contain splats for ``ignore_splats``.
        """
        if experiment is None:
            exp_keys = range(len(self.results))
        else:
            exp_keys = [experiment]

        counts_list = []
        for key in exp_keys:
            exp = self._get_experiment(key)
            try:
                header = exp.header
            except (AttributeError, QiskitError):
                header = None

            data = self.data(key)
            if yield_handling is None:
                counts = data["counts"]
            else:
                counts, _ = YieldHandling.from_name(yield_handling).apply(
                    data["raw_counts"]
                )

            counts_header = {}
            if header:
                counts_header = {
                    k: v
                    for k, v in header.items()
                    if k in {"time_taken", "creg_sizes", "memory_slots"}
                }
            if any("*" in bitstring for bitstring in counts):
                # splat keys can't be reformatted via creg_sizes/memory_slots,
                # but they are already register-formatted, so keep them as-is
                counts_header.pop("creg_sizes", None)
                counts_header.pop("memory_slots", None)

            counts_list.append(Counts(counts, **counts_header))

        if len(counts_list) == 1:
            return counts_list[0]
        return counts_list
