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

"""Qiskit provider exposing D-Wave Leap QCDL solvers as backends."""

from __future__ import annotations

from dwave.cloud import Client

from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends

from dwave.plugins.qiskit.leap.backend import QCDLBackend

__all__ = ["DWaveProvider"]


class DWaveProvider:
    """A Qiskit provider for D-Wave Leap quantum gate model (QCDL) solvers.

    Credentials and connection parameters follow the standard
    ``dwave-cloud-client`` configuration: values given here take precedence
    over environment variables, which take precedence over the configuration
    file.

    Args:
        **config:
            :class:`~dwave.cloud.Client` configuration options passed to
            :meth:`~dwave.cloud.client.Client.from_config`, e.g. ``config_file``
            or ``profile``.

    Examples:
        >>> from dwave.plugins.qiskit import DWaveProvider
        >>> with DWaveProvider() as provider:      # doctest: +SKIP
        ...     backend = provider.get_backend()
    """

    def __init__(self, **config):
        # default to the base client, but allow override
        config.setdefault("client", "base")
        # default to short-lived session to prevent resets on slow uploads
        config.setdefault("connection_close", True)

        self._config = config
        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client.from_config(**self._config)
        return self._client

    def backends(self, name: str | None = None, **kwargs) -> list[QCDLBackend]:
        """List QCDL backends available on Leap.

        Backends are listed newest solver first.

        Args:
            name: If given, only the backend (solver) with this name.
            **kwargs: Backend attribute filters, matched against backends'
                configuration and status.

        Returns:
            The matching backends.
        """
        filters = dict(
            supported_problem_types__contains="qcdl",
            order_by="-properties.version",
        )
        if name is not None:
            filters["name"] = name
        solvers = self._get_client().get_solvers(**filters)
        backends = [QCDLBackend(solver, provider=self) for solver in solvers]
        return filter_backends(backends, **kwargs)

    def get_backend(self, name: str | None = None, **kwargs) -> QCDLBackend:
        """Return a single QCDL backend matching the specified filtering.

        When more than one backend matches, the one with the newest solver
        version is returned.

        Args:
            name: Name of the backend (solver).
            **kwargs: Backend attribute filters, as for :meth:`backends`.

        Returns:
            The matching backend.

        Raises:
            QiskitBackendNotFoundError: If no backend matches the filtering.
        """
        backends = self.backends(name, **kwargs)
        if not backends:
            raise QiskitBackendNotFoundError("no backend matches the criteria")
        return backends[0]

    def close(self) -> None:
        """Release the cloud client's resources.

        Backends and jobs obtained from this provider stop working once the
        provider is closed.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> DWaveProvider:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
