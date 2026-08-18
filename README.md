# D-Wave Ocean plugin for IBM Qiskit

Enables [Qiskit](https://www.ibm.com/quantum/qiskit) users to obtain ground state(s) of Ising Hamiltonians using [D-Wave](https://www.dwavesys.com/)'s QPU available via [Leap](https://cloud.dwavesys.com/).

The package provides an implementation of Qiskit Optimization's
[`SamplingMinimumEigensolver`](https://qiskit-community.github.io/qiskit-optimization/apidocs/qiskit_optimization.minimum_eigensolvers.html)
interface (available as `DWaveMinimumEigensolver`) which can be used directly on qubit operators, or via
`qiskit_optimization`'s [`MinimumEigenOptimizer`](https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.MinimumEigenOptimizer.html).


## Examples

Solve a [`QuadraticProgram`](https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.QuadraticProgram.html)
with [`MinimumEigenOptimizer`](https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.MinimumEigenOptimizer.html)
using `DWaveMinimumEigensolver`:

```python
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
```

Solve a 6-city TSP (or some other
[optimization application](https://qiskit-community.github.io/qiskit-optimization/apidocs/qiskit_optimization.applications.html)),
a 36-qubit Ising Hamiltonian:

```python
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
```

For comparison, trying this on `NumPyMinimumEigensolver` (which constructs the
full 2^36 state space) produces:

```python
>>> from qiskit_optimization.minimum_eigensolvers import NumPyMinimumEigensolver
>>> result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
# snipped for brevity
memory allocation of 1818775484491218187754844912 bytes failed
Aborted (core dumped)
```

and trying with `QAOA` backed by the reference `StatevectorSampler` primitive
produces:

```python
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
```

## Installation

Compatible with Python 3.10+, [Qiskit](https://github.com/Qiskit/qiskit) 1.0+,
[qiskit-optimization](https://github.com/qiskit-community/qiskit-optimization) 0.7+,
and [Ocean](https://github.com/dwavesystems/dwave-ocean-sdk)'s dwave-system 1.20+.

```bash
pip install dwave-qiskit-plugin
```

To install from source:
```bash
pip install .
```

Test requirements are in `tests/requirements.txt`.

Note: [Configured access to D-Wave API](https://docs.dwavequantum.com/en/latest/ocean/sapi_access_basic.html) is required.


## License

Released under the Apache License 2.0. See [LICENSE](./LICENSE) file.
