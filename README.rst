==================================
D-Wave Ocean plugin for IBM Qiskit
==================================

Enables `Qiskit <https://www.ibm.com/quantum/qiskit>`_ users to work with `D-Wave <https://www.dwavesys.com/>`_'s
quantum resources, available via `Leap <https://cloud.dwavesys.com/>`_.

DWaveMinimumEigensolver
=======================

The package provides an implementation of Qiskit Optimization's
`SamplingMinimumEigensolver <https://qiskit-community.github.io/qiskit-optimization/apidocs/qiskit_optimization.minimum_eigensolvers.html>`_
interface (available as ``DWaveMinimumEigensolver``) which can be used directly on qubit operators, or via
``qiskit_optimization``'s `MinimumEigenOptimizer <https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.MinimumEigenOptimizer.html>`_.


Examples
--------

Solve a `QuadraticProgram <https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.QuadraticProgram.html>`_
with `MinimumEigenOptimizer <https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.MinimumEigenOptimizer.html>`_
using ``DWaveMinimumEigensolver``:

.. code-block:: python

    >>> from qiskit_optimization import QuadraticProgram
    >>> from qiskit_optimization.algorithms import MinimumEigenOptimizer
    >>> from dwave.plugins.qiskit import DWaveMinimumEigensolver
    ...
    >>> # Construct a simple quadratic program
    >>> qp = QuadraticProgram()
    >>> qp.binary_var('x')
    >>> qp.binary_var('y')
    >>> qp.minimize(quadratic={'xy': 1})
    ...
    >>> # Solve using Qiskit's MinimumEigenOptimizer on D-Wave QPU as a minimum eigen solver
    >>> dwave_mes = DWaveMinimumEigensolver()
    >>> optimizer = MinimumEigenOptimizer(dwave_mes)
    >>> result = optimizer.solve(qp)
    ...
    >>> print(result)
    fval=0.0, x=0.0, y=0.0, status=SUCCESS
    >>> [(''.join(str(int(v)) for v in s.x), s.fval, s.probability) for s in result.samples]
    [('00', 0.0, 0.33), ('10', 0.0, 0.33), ('01', 0.0, 0.33)]

Solve a 6-city TSP (or some other
`optimization application <https://qiskit-community.github.io/qiskit-optimization/apidocs/qiskit_optimization.applications.html>`_),
a 36-qubit Ising Hamiltonian:

.. code-block:: python

    >>> from qiskit_optimization.applications import Tsp
    >>> from qiskit_optimization.algorithms import MinimumEigenOptimizer
    >>> from dwave.plugins.qiskit import DWaveMinimumEigensolver
    ...
    >>> tsp = Tsp.create_random_instance(6, seed=123)
    >>> qp = tsp.to_quadratic_program()
    ...
    >>> dwave_mes = DWaveMinimumEigensolver(num_reads=1000)
    >>> result = MinimumEigenOptimizer(dwave_mes).solve(qp)
    ...
    >>> tsp.interpret(result)
    [3, 4, 2, 1, 5, 0]

For comparison, trying this on ``NumPyMinimumEigensolver`` (which constructs the
full 2^36 state space) produces:

.. code-block:: python

    >>> from qiskit_optimization.minimum_eigensolvers import NumPyMinimumEigensolver
    >>> result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    # snipped for brevity
    memory allocation of 1818775484491218187754844912 bytes failed
    Aborted (core dumped)

and trying with ``QAOA`` backed by the reference ``StatevectorSampler`` primitive
produces:

.. code-block:: python

    >>> import numpy as np
    >>> from qiskit.primitives import StatevectorSampler
    >>> from qiskit_optimization.minimum_eigensolvers import QAOA
    >>> from qiskit_optimization.optimizers import COBYLA
    ...
    >>> qaoa_mes = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(),
    ...                 initial_point=np.array([0.0, 0.0]))
    >>> result = MinimumEigenOptimizer(qaoa_mes).solve(qp)
    # snipped for brevity
    MemoryError: Unable to allocate 1.00 TiB for an array with shape (68719476736,) and data type complex128

QCDL Translators
================

``dwave.plugins.qiskit.qcdl.translators`` converts Qiskit ``QuantumCircuit``
objects into D-Wave's QCDL program format, for running gate-model circuits on
D-Wave's gate-model hardware/simulator via ``dwave-gate``.

Examples
--------

Translate a Bell state Qiskit circuit into a QCDL program:

.. code-block:: python

    >>> from qiskit import QuantumCircuit
    >>> from dwave.gate.qcdl import print_qcdl
    >>> from dwave.plugins.qiskit.qcdl.translators import circuit_to_qcdl
    ...
    >>> qc = QuantumCircuit(2, 2, name="bell")
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.measure([0, 1], [0, 1])
    ...
    >>> result = circuit_to_qcdl(qc)
    ...
    >>> print_qcdl(result.qcdl)
    begin quantum
       q0.initialize(q1)
       h([q0], q0)
       cx([q0, q1], q0, q1)
       measure([q0], q0, log=True, tag="0")
       measure([q1], q1, log=True, tag="1")
    end quantum

DWaveProvider
=============

``DWaveProvider`` exposes D-Wave's Leap QCDL simulator solvers
through the standard Qiskit provider/backend interface: circuits passed to
``QCDLSimulatorBackend.run()`` are translated to QCDL, submitted to a Leap solver,
and the answers are returned as a ``qiskit.result.Result``. Leap credentials are
picked up from the standard `dwave-cloud-client configuration
<https://docs.dwavequantum.com/en/latest/ocean/api_ref_cloud/config.html>`_
(configuration file or environment variables), or can be passed to the
provider directly.

Examples
--------

Run a Bell state circuit on a Leap QCDL solver:

.. code-block:: python

    >>> from qiskit import QuantumCircuit
    >>> from dwave.plugins.qiskit import DWaveProvider
    ...
    >>> qc = QuantumCircuit(2, 2, name="bell")
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.measure([0, 1], [0, 1])
    ...
    >>> with DWaveProvider() as provider:
    ...     backend = provider.get_backend()
    ...     job = backend.run(qc, shots=1000)
    ...     counts = job.result().get_counts()
    >>> counts                                      # doctest: +SKIP
    {'00': 512, '11': 488}

``run()`` also accepts a list of circuits; by default they are packed into as
few QCDL programs as estimated to fit (disable with ``qcdl_pack_target=False``
to submit one QCDL program per circuit).

Installation
============

Compatible with Python 3.11+, `Qiskit <https://github.com/Qiskit/qiskit>`_ 1.0+,
`qiskit-optimization <https://github.com/qiskit-community/qiskit-optimization>`_ 0.7+,
and `Ocean <https://github.com/dwavesystems/dwave-ocean-sdk>`_'s dwave-system 1.20+.

.. code-block:: bash

    pip install dwave-qiskit-plugin

To install from source:

.. code-block:: bash

    pip install --group dev
    pip install --editable .

Test dependencies are defined in the ``test`` dependency group in
``pyproject.toml``, and can be installed with:

.. code-block:: bash

    pip install --group test .
    python -m pytest

License
=======

Released under the Apache License 2.0. See `LICENSE <./LICENSE>`_ file.

Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
-------------

We use `reno <https://docs.openstack.org/reno/>`_ to manage release notes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
