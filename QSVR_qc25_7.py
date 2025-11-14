import warnings
warnings.filterwarnings("ignore", message="No gradient function provided")

""""
QSVR
""""

"""
| Paket                       | Verzija |
| --------------------------- | ------- |
| **python**                  | 3.11.13 |
| **qiskit**                  | 1.4.4   |
| **qiskit-machine-learning** | 0.8.3   |
| **qiskit-ibm-runtime**      | 0.43.0  |
| **macOS**                   | Tahos   |
| **Apple**                   | M1      |
"""

"""
https://github.com/forsing
https://github.com/forsing?tab=repositories
"""

"""
Loto Skraceni Sistemi
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""

"""
svih 4510 izvlacenja
30.07.1985.- 11.11.2025.
"""


import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile 
from qiskit_aer import Aer

from qiskit.visualization import plot_histogram

from qiskit_machine_learning.utils import algorithm_globals
import random

from tqdm import tqdm

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# =========================
# Učitaj CSV
# =========================
df = pd.read_csv("/data/loto7_4510_k89.csv", header=None)
min_val = [1,2,3,4,5,6,7]
max_val = [33,34,35,36,37,38,39]

def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        if not df_indexed[i].between(0, max_val[i]-min_val[i]).all():
            raise ValueError(f"Kolona {i} izvan validnog opsega")
    return df_indexed

df_indexed = map_to_indexed_range(df, min_val, max_val)

# =========================
# QSVR Parametri
# =========================
num_qubits = 5
num_layers = 2
num_positions = 5   # 5+2=7
shots = 1024
simulator = Aer.get_backend('qasm_simulator')

def encode_position(value):
    v = int(value)
    bin_full = format(v,'b').zfill(num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i, bit in enumerate(reversed(bin_full)):
        if bit=='1':
            qc.x(i)
    return qc

def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    return qc

def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer*num_qubits
        end = (layer+1)*num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc

def build_qcbm_circuit(params_list, values):
    total_qubits = num_qubits*num_positions
    qc = QuantumCircuit(total_qubits)
    for pos in range(num_positions):
        start_q = pos*num_qubits
        end_q = start_q + num_qubits
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q,end_q), inplace=True)
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q,end_q), inplace=True)
    qc.measure_all()
    return qc

# =========================
# Predikcija QSVR
# =========================
predicted_combination = []
last_value = df_indexed.iloc[-1].tolist()

for pos in range(7):
    print(f"\n--- QSVR pozicija {pos+1} ---")
    # Nasumični parametri QSVR (odvojeni od QKR)
    params_list = [np.random.uniform(0,2*np.pi, num_qubits*num_layers) for _ in range(num_positions)]
    
    qc = build_qcbm_circuit(params_list, last_value)
    
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    
    # Najverovatniji izlaz
    most_probable = max(counts, key=counts.get)
    pred_val = int(most_probable[-num_qubits:],2)
    pred_val = max(min_val[pos], min(pred_val + min_val[pos], max_val[pos]))
    predicted_combination.append(pred_val)

print()
print("\n=== QSVR Predviđena loto kombinacija (7) ===")
print(" ".join(str(x) for x in predicted_combination))

# opcionalno: vizualizacija poslednje pozicije
plot_histogram(counts)

print()
"""
=== QSVR Predviđena loto kombinacija (7) ===
22 25 x x x 34 30
"""
