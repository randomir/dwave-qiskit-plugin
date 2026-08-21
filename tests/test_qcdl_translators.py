# Copyright 2020 D-Wave
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

"""Test the qcdl translators."""

import random
import re

import pytest
from dwave.gate.qcdl import print_qcdl
from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    instruction,
)
from qiskit_aer import QasmSimulator

from dwave.plugins.qiskit.qcdl.translators import (
    InstructionMemoryEstimate,
    _active_qubits,
    circuit_to_qcdl,
    concatenate_circuits_to_qcdl,
    group_circuits_by_instruction_estimates,
)


def circuit_to_counts(circuit: QuantumCircuit, shots=10000) -> dict:
    backend = QasmSimulator()
    result = backend.run(circuit, shots=shots).result()
    return result.get_counts()


def test_measurement_only_circuit():
    """Test a valid circuit with only measurements."""
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0"]


def test_simple_circuit():
    """Test basic structure of a simple circuit"""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0"]


# pylint: disable=invalid-name
def test_circuit_with_entangling_ops():
    """Test structure of circuits with entangling ops."""
    qc = QuantumCircuit(2, 2)
    qc.cx(1, 0)
    qc.measure_all()
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, None, "0", "1"]


def test_rotation_from_instruction_params():
    """Test that instruction parameters are used for rotation."""
    qc = QuantumCircuit(2)
    qc.append(instruction.Instruction("rx", 1, 0, [1.0]), [1])
    qc.measure_all()
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0", "1"]


def test_invalid_instruction():
    """test that the number of qubits for a gate is validated"""
    num_qubits = random.randint(3, 10)
    qc = QuantumCircuit(num_qubits)
    qc.append(instruction.Instruction("rx", num_qubits, 0, [1.0]), range(num_qubits))
    qc.measure_all()
    with pytest.raises(
        TypeError,
        match=f"takes 2 positional arguments but {num_qubits+1} were given",
    ):
        circuit_to_qcdl(qc)


def test_unknown_gate():
    qc = QuantumCircuit(2)
    gate_name = "not_a_gate"
    qc.append(instruction.Instruction(gate_name, 1, 0, [1.0]), [1])
    qc.measure_all()

    with pytest.raises(NotImplementedError, match=f"{gate_name} is not implemented"):
        circuit_to_qcdl(qc)


def test_circuit_measurement():
    """
    Test that putting an instruction on a qubit that has been measured is allowed
    """
    qc = QuantumCircuit(2, 2)
    qc.measure(1, 1)
    qc.x(1)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, "0"]


def test_unmeasured_clbit():
    """If only one qubit/clbit is not measured"""
    qc = QuantumCircuit(2, 2)
    qc.measure(1, 1)
    qc.x(0)
    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, "0"]


@pytest.mark.parametrize("q", [0, 1, 2, 3])
def test_circuit_with_multiple_registers(q):
    """Test multiple classical registers"""
    qr0 = QuantumRegister(2, "qr0")
    qr1 = QuantumRegister(2, "qr1")
    cr0 = ClassicalRegister(2, "cr0")
    cr1 = ClassicalRegister(2, "cr1")

    qc = QuantumCircuit(
        qr0,
        qr1,
        cr0,
        cr1,
    )

    qc.x(q)
    qc.measure([qr0[0], qr0[1], qr1[0], qr1[1]], [cr0[0], cr0[1], cr1[0], cr1[1]])

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0", "1", "2", "3"]


@pytest.mark.parametrize("q", [0, 1, None])
def test_measure_to_multiple_cbits(q):
    """Test that we are allowed to measure the same qubit into multiple classical registers"""
    qc = QuantumCircuit(2, 2)
    if q is not None:
        qc.x(q)
    qc.measure(1, 0)
    qc.measure(1, 1)

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["0", "1"]


@pytest.mark.parametrize("qubits_used", [[0], [1], [0, 1], [3, 4, 5]])
def test_initialize(qubits_used):
    qc = QuantumCircuit(8, 8)
    for q in qubits_used:
        qc.x(q)
        qc.measure(q, q)
    assert _active_qubits(qc) == qubits_used

    qcdl = circuit_to_qcdl(qc)
    qcdl_str = print_qcdl(qcdl.qcdl, to_Display=False)
    expected_initialize = (
        f"q{qubits_used[0]}.initialize("
        + ", ".join(f"q{q}" for q in qubits_used[1:])
        + ")"
    )
    assert expected_initialize in qcdl_str


@pytest.mark.parametrize("q", [0, 1, None])
def test_measure_from_multiple_qubits(q):
    """Test that we are allowed to measure different qubits into the same cbit"""
    qc = QuantumCircuit(2, 2)
    if q is not None:
        qc.x(q)
    qc.measure(0, 1)
    qc.measure(1, 1)

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == [None, "1"]


@pytest.mark.parametrize("q", [0, 1, None])
def test_multiple_measures_per_shot(q):
    """Measure multiple times same qubit to same cbit"""
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(qr, cr)
    if q is not None:
        qc.x(q)
    for _ in range(3):
        qc.measure(qr, cr)

    qcdl = circuit_to_qcdl(qc)
    assert qcdl.clbit_to_tag == ["4", "5"]


def test_if():
    bits = [Qubit(), Qubit(), Clbit()]
    qc = QuantumCircuit(bits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    with qc.if_test((bits[2], 0)) as else_:
        qc.h(0)
    with else_:
        qc.x(0)

    with pytest.raises(NotImplementedError, match="control-flow"):
        circuit_to_qcdl(qc)


def test_no_measurments():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    with pytest.raises(ValueError, match="no measurements"):
        circuit_to_qcdl(qc)


def _check_circuit_concatenation(
    circuits: list[QuantumCircuit],
    shots: int = 123,
) -> None:
    num_circuits = len(circuits)
    qcdl_metadata = concatenate_circuits_to_qcdl(circuits=circuits)
    assert len(qcdl_metadata.circuit_metadata) == num_circuits
    assert isinstance(qcdl_metadata.qcdl, dict)

    # metadata dict
    metadata = qcdl_metadata.qcdl.get("metadata", {})
    assert metadata is not None

    # check each circuit's qiskit header exists
    for qwm in qcdl_metadata.circuit_metadata:
        md_entry = metadata.get(qwm.job_name)
        assert md_entry is not None, f"Metadata missing for job_name {qwm.job_name}"
        assert (
            "qiskit" in md_entry
        ), f"qiskit header missing for job_name {qwm.job_name}"
        assert md_entry["qiskit"] == qwm.qiskit_header

    expected_job_names = {qwm.job_name for qwm in qcdl_metadata.circuit_metadata}
    assert set(metadata.keys()) == expected_job_names

    for qwm in qcdl_metadata.circuit_metadata:
        entry = metadata[qwm.job_name]
        assert "qiskit" in entry
        assert entry["qiskit"] == qwm.qiskit_header
        # circuit metadata (e.g. RB's "xval") must round-trip when packing
        assert "circuit_metadata" in entry
        assert entry["circuit_metadata"] == qwm.circuit_metadata

    # each has an initialize, and they're all the same (all qubits)
    qcdl_str = print_qcdl(qcdl_metadata.qcdl, to_Display=False)
    initializations = list(
        re.findall(r"^\s*q\d\.initialize\([q\d\,\s]*\)$", qcdl_str, flags=re.MULTILINE)
    )
    assert len(initializations) == num_circuits
    assert len(set(initializations)) == 1

    tags = set()
    for circuit_qcdl_metadata in qcdl_metadata.circuit_metadata:
        tags_used = [
            tag for tag in circuit_qcdl_metadata.clbit_to_tag if tag is not None
        ]
        assert len(set(tags_used)) == len(tags_used)
        assert tags.isdisjoint(tags_used)
        tags.update(tags_used)
        assert circuit_qcdl_metadata.qcdl is None


@pytest.mark.parametrize("measure_reps", range(1, 4))
def test_circuit_concatenation(measure_reps):
    circuits = []
    labels = ["00", "10", "01", "11", "0", "1"]
    random.shuffle(labels)
    for label in labels:
        qc = QuantumCircuit(len(label), name=f"circ{label}")
        for idx, val in enumerate(reversed(label)):
            if val == "1":
                qc.x(idx)
        for _ in range(measure_reps):
            qc.measure_all()
        circuits.append(qc)

    shots = random.randint(10, 100)

    _check_circuit_concatenation(circuits=circuits, shots=shots)


@pytest.mark.parametrize("qubits_used", [[0], [1], [0, 1], [3, 4, 5]])
def test_concatenation_active_qubits(qubits_used):
    circuits = []
    # each circuit declares and uses a different number of qubits
    for q in qubits_used * 3:
        qc = QuantumCircuit(q + 1, q + 1, name=f"circ{q}")
        qc.measure(q, q)
        circuits.append(qc)

    _check_circuit_concatenation(circuits=circuits)
    qcdl_metadata = concatenate_circuits_to_qcdl(circuits=circuits)
    # make sure the final qubits_used is what we expect
    assert qcdl_metadata.qcdl["program"]["signature"]["qubits_used"] == [
        f"q{q}" for q in qubits_used
    ]


def test_group_circuits_by_instruction_estimates():
    circuits = []
    for idx in range(10):
        qc = QuantumCircuit(1, name=f"circ{idx}")
        for _ in range(10):
            qc.x(0)
        circuits.append(qc)

    estimator = InstructionMemoryEstimate()
    est_per_circuit = estimator.estimate_circuit(circuits[0])[0]
    assert est_per_circuit > 0.01

    # all separated
    groups = group_circuits_by_instruction_estimates(circuits, qcdl_pack_target=0)
    assert len(groups) == len(circuits)

    # all in one group
    groups = group_circuits_by_instruction_estimates(circuits, qcdl_pack_target=1000)
    assert len(groups) == 1

    # two groups
    groups = group_circuits_by_instruction_estimates(
        circuits, qcdl_pack_target=est_per_circuit * 5
    )
    assert len(groups) == 2


def test_qiskit_headers_are_per_circuit_not_combined():

    # add in two circuits
    qc1 = QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)

    qc2 = QuantumCircuit(1, 1)
    qc2.x(0)
    qc2.measure(0, 0)

    result = concatenate_circuits_to_qcdl([qc1, qc2])

    metadata = result.qcdl["metadata"]

    # should be 1 metadata per circuit
    assert len(metadata) == 2

    # this should not exist, should be split by {"circuit_name": "qiskit": {qiskit header content}}
    assert "qiskit" not in metadata

    # check each header
    for qwm in result.circuit_metadata:
        assert metadata[qwm.job_name]["qiskit"] == qwm.qiskit_header

    # headers are different, we gave it two different circuits
    headers = [v["qiskit"] for v in metadata.values()]
    assert headers[0] != headers[1]

