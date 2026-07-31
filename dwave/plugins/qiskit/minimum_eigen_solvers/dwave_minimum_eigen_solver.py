# Copyright 2020 D-Wave Systems Inc.
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

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dimod
from dwave.system import DWaveSampler, AutoEmbeddingComposite

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import QuasiDistribution
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.minimum_eigensolvers import (
    SamplingMinimumEigensolver, SamplingMinimumEigensolverResult)

__all__ = ['DWaveMinimumEigensolver']

logger = logging.getLogger(__name__)


class DWaveMinimumEigensolver(SamplingMinimumEigensolver):
    """Obtain ground state(s) of an Ising Hamiltonian using D-Wave's QPU.

    Args:
        operator:
            Ising Hamiltonian qubit operator, with at most 2 Pauli Zs in
            any Pauli term.
        aux_operators:
            Auxiliary operators to be evaluated at each eigenvalue.
        sampler:
            Instantiated D-Wave sampler. Defaults to
            ~dwave.system.AutoEmbeddingComposite`-wrapped
            `~dwave.system.DWaveSampler` over a QPU solver.
        num_reads:
            Number of QPU reads.

    Note:
        Configured access to D-Wave API/Leap is a prerequisite.

    Example:
        # define a simple quadratic program
        qp = QuadraticProgram()
        qp.binary_var('x')
        qp.binary_var('y')
        qp.minimize(linear=[1,-2], quadratic={('x', 'y'): 1})

        # solve it with a minimum eigen optimizer that uses D-Wave QPU
        dwave_mes = DWaveMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(dwave_mes)
        result = optimizer.solve(qp)

    """

    def __init__(self,
                 operator: Optional[BaseOperator] = None,
                 aux_operators: Optional[List[Optional[BaseOperator]]] = None,
                 sampler: dimod.Sampler = None,
                 num_reads: int = 100,
                 ) -> None:
        super().__init__()
        self.operator = operator
        self.aux_operators = aux_operators

        self._sampler = sampler
        self._num_reads = num_reads

    @classmethod
    def supports_aux_operators(cls) -> bool:
        # NOTE: MinimumEigenOptimizer refuses to work with solvers that do not
        # claim aux operator support (it uses the flag as a proxy for "returns
        # an eigenstate"), so this has to return True
        return True

    def _operator_to_bqm(self, operator: BaseOperator) -> dimod.BinaryQuadraticModel:
        """Convert an Ising Hamiltonian operator (with at most 2 Pauli Zs in
        any Pauli term) to a `~dimod.BinaryQuadraticModel` suitable for
        submission to a D-Wave sampler.
        """
        # convert `operator` to QUBO, failing with `QiskitOptimizationError`
        # for unsupported operators
        qp = QuadraticProgram()
        qp.from_ising(operator)

        # sanity check
        assert qp.objective.sense is qp.objective.Sense.MINIMIZE

        # construct a BQM
        # (use to_array for linear coefficients to make sure implied, but not
        # used, variables are included; diagonal quadratic terms are folded
        # into linear biases by dimod, as x**2 == x for binary variables)
        return dimod.BinaryQuadraticModel(qp.objective.linear.to_array(),
                                          qp.objective.quadratic.to_dict(),
                                          qp.objective.constant,
                                          vartype=dimod.BINARY)

    @property
    def operator(self) -> Optional[BaseOperator]:
        return self._operator

    @operator.setter
    def operator(self, operator: Optional[BaseOperator]) -> None:
        """Convert an Ising Hamiltonian operator to a binary quadratic model
        suitable for submission to a D-Wave sampler.

        Args:
            operator:
                Ising Hamiltonian qubit operator, with at most 2 Pauli Zs in
                any Pauli term.

        Raises:
            QiskitOptimizationError:
                If there are Pauli Xs in any Pauli term, or if there are more
                than 2 Pauli Zs in any Pauli term

        """
        self._operator = operator
        logger.debug('operator set to %r', operator)

        if operator is not None:
            self._bqm = self._operator_to_bqm(operator)
            logger.debug('BQM set to %s', self._bqm)

    @property
    def aux_operators(self) -> Optional[List[Optional[BaseOperator]]]:
        return self._aux_operators

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators: Optional[
                          Union[BaseOperator,
                                List[Optional[BaseOperator]]]]) -> None:
        if aux_operators is None:
            aux_operators = []
        if not isinstance(aux_operators, list):
            aux_operators = [aux_operators]

        self._aux_operators = aux_operators
        self._aux_bqms = None

    @property
    def bqm(self) -> Optional[dimod.BinaryQuadraticModel]:
        """Binary quadratic model representation of Ising Hamiltonian operator.
        """
        bqm = getattr(self, '_bqm', None)
        if bqm is None:
            raise ValueError('operator not yet set, so bqm not yet available')
        return bqm

    @property
    def aux_bqms(self) -> Optional[List[dimod.BinaryQuadraticModel]]:
        """Binary quadratic model representations of auxiliary Ising Hamiltonian
        operators.
        """
        bqms = getattr(self, '_aux_bqms', None)
        if bqms is None:
            bqms = self._aux_bqms = [self._operator_to_bqm(aux_op) for aux_op in self.aux_operators]
        return bqms

    @property
    def sampler(self) -> dimod.Sampler:
        """Configured D-Wave sampler to use."""
        _sampler = getattr(self, '_sampler', None)
        if _sampler is None:
            _sampler = self._sampler = AutoEmbeddingComposite(DWaveSampler())
        return _sampler

    def compute_minimum_eigenvalue(
            self,
            operator: Optional[BaseOperator] = None,
            aux_operators: Optional[List[Optional[BaseOperator]]] = None
    ) -> SamplingMinimumEigensolverResult:
        if operator is not None:
            self.operator = operator
        if aux_operators is not None:
            self.aux_operators = aux_operators
        return self._run()

    def _sample(self) -> dimod.SampleSet:
        params = {}
        if 'num_reads' in self.sampler.parameters:
            params['num_reads'] = self._num_reads
        return self.sampler.sample(self.bqm, **params)

    @staticmethod
    def _stringify(sample: Dict, variables: Sequence) -> str:
        """Convert a sample (mapping of variable to 0/1 value) to a bit string
        in Qiskit's little-endian convention (variable/qubit 0 rightmost).
        """
        return ''.join(str(sample[v]) for v in reversed(variables))

    def _eval_aux_operators(
            self, samples: List[Tuple[Dict, float]]) -> List[Tuple[float, dict]]:
        """Evaluate all aux_operators as expectation values over the
        probability-weighted (ground state) samples.
        """
        return [(sum(p * bqm.energy(sample) for sample, p in samples), {})
                for bqm in self.aux_bqms]

    def _run(self) -> SamplingMinimumEigensolverResult:
        """Sample the Ising Hamiltonian provided on a D-Wave QPU to obtain the
        ground state(s).

        Returns:
            SamplingMinimumEigensolverResult:
                Results, namely a quasi-distribution over ground state samples
                in `eigenstate` and the lowest energy in `eigenvalue`.

        Raises:
            ValueError:
                if no operator has been provided
        """

        sampleset = self._sample()

        logger.debug('sampleset: %r', sampleset)

        if sampleset.vartype is not dimod.BINARY:   # pragma: no cover
            logger.critical("Unexpected result: sampleset=%r", sampleset)
            raise TypeError('expected binary vartype of result sampleset')

        # approximate ground state(s) with lowest-energy samples returned
        ground = sampleset.lowest(rtol=0)
        logger.debug('ground states (%d): %r', len(ground), ground)

        variables = sorted(ground.variables)
        total = int(ground.record.num_occurrences.sum())

        # probability-weighted ground state samples
        samples = [(datum.sample, datum.num_occurrences / total)
                   for datum in ground.data(fields=['sample', 'num_occurrences'])]

        result = SamplingMinimumEigensolverResult()
        result.eigenvalue = ground.first.energy

        # NOTE: `MinimumEigenOptimizer` interprets eigenstate bitstrings in
        # Qiskit's little-endian convention, i.e. variable/qubit 0 rightmost
        result.eigenstate = QuasiDistribution(
            {self._stringify(sample, variables): p for sample, p in samples})

        best = self._stringify(ground.first.sample, variables)
        result.best_measurement = {
            'state': int(best, 2),
            'bitstring': best,
            'value': ground.first.energy,
            'probability': ground.first.num_occurrences / total,
        }

        # optionally, evaluate aux_operators
        if self._aux_operators:
            result.aux_operators_evaluated = self._eval_aux_operators(samples)

        # include all samples for inspection
        result.sampleset = sampleset

        logger.debug('run result: %r', result)

        return result

    def run(self,
            operator: Optional[BaseOperator] = None,
            aux_operators: Optional[List[Optional[BaseOperator]]] = None
    ) -> SamplingMinimumEigensolverResult:
        """Obtain ground state(s) of an Ising Hamiltonian using D-Wave's QPU.
        """
        return self.compute_minimum_eigenvalue(operator, aux_operators)
