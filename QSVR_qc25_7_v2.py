import warnings

warnings.filterwarnings("ignore", message="No gradient function provided")

"""
QSVR (Quantum Support Vector Regressor)
"""


"""
svih 4584 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 20.03.2026.
"""


import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None  # type: ignore[misc, assignment]

from qiskit.visualization import plot_histogram

from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
# QSVR odvojeno od QKR (isti SEED baza, drugi offset u RNG za params)
SEED_QSVR_PARAMS = SEED + 137
# v2: isti ishod pri ponovnom pokretanju (Aer + transpiler; tie-break na counts)
SEED_SIMULATOR = SEED + 903
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# =========================
# Učitaj CSV
# =========================
df = pd.read_csv("/Users/4c/Desktop/GHQ/data/loto7_4584_k23.csv", header=None).iloc[:, :7]
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]


def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Kolona {i} izvan validnog opsega")
    return df_indexed


df_indexed = map_to_indexed_range(df, min_val, max_val)

# =========================
# QSVR Parametri (qc25: 5 blokova × 5 qubit-a)
# =========================
num_qubits = 5
num_layers = 2
num_positions = 5  # 5 u kolu + 2 izvedena → 7 (kao QCBM_qc25_7_v2)
shots = 1024

if AerSimulator is not None:
    simulator = AerSimulator()
else:
    simulator = Aer.get_backend("qasm_simulator")


def encode_position(value):
    v = int(value)
    bin_full = format(v, "b")
    # v2: kao QKR_v2 — ako treba više bitova, poslednjih num_qubits (LSB)
    if len(bin_full) > num_qubits:
        bin_repr = bin_full[-num_qubits:]
    else:
        bin_repr = bin_full.zfill(num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == "1":
            qc.x(i)
    return qc


def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer * num_qubits
        end = (layer + 1) * num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc


def build_qcbm_circuit(params_list, values):
    total_qubits = num_qubits * num_positions
    qc = QuantumCircuit(total_qubits)
    for pos in range(num_positions):
        start_q = pos * num_qubits
        end_q = start_q + num_qubits
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)
    qc.measure_all()
    return qc


def bitstring_to_loto_with_7(
    bitstring_int: int,
    n_qubits: int = 5,
    num_pos: int = 5,
) -> list[int]:
    """Ista logika kao u QCBM_qc25_7_v2 (5 izmerenih blokova + 6. i 7.)."""
    num_bits = n_qubits * num_pos
    bitstring = format(int(bitstring_int), "b").zfill(num_bits)
    main_numbers: list[int] = []
    for pos in range(num_pos):
        start = pos * n_qubits
        chunk = bitstring[start : start + n_qubits]
        val = int(chunk, 2)
        mv = min_val[pos]
        Mv = max_val[pos]
        rng = Mv - mv + 1
        mapped = (val % rng) + mv
        main_numbers.append(int(mapped))

    def find_unique(start_val: int, used_set: set[int], idx: int) -> int:
        mv = min_val[idx]
        Mv = max_val[idx]
        rng = Mv - mv + 1
        v = ((start_val - mv) % rng) + mv
        tries = 0
        while v in used_set and tries < rng:
            v = mv + ((v - mv + 1) % rng)
            tries += 1
        if v in used_set:
            for cand in range(mv, Mv + 1):
                if cand not in used_set:
                    v = cand
                    break
        return int(v)

    sum_main = sum(main_numbers)
    start6 = (sum_main) % (max_val[5] - min_val[5] + 1) + min_val[5]
    sixth = find_unique(start6, set(main_numbers), 5)
    used = set(main_numbers) | {sixth}
    start7 = (sum_main + sixth) % (max_val[6] - min_val[6] + 1) + min_val[6]
    seventh = find_unique(start7, used, 6)
    return main_numbers + [sixth, seventh]


def _counts_key_to_int(key: str | int) -> int:
    if isinstance(key, int):
        return int(key)
    s = key.replace(" ", "")
    return int(s, 2)


# =========================
# Predikcija QSVR (jedno kolo qc25 → 7 brojeva)
# =========================
last_value = df_indexed.iloc[-1].tolist()
values_5 = [int(last_value[i]) for i in range(num_positions)]

print()
print("=== QSVR qc25: jedna simulacija (5×5 qubit-a), zatim 7 brojeva (QCBM obrazac) ===")

_rng = np.random.default_rng(SEED_QSVR_PARAMS)
params_list = [_rng.uniform(0, 2 * np.pi, num_qubits * num_layers) for _ in range(num_positions)]

qc = build_qcbm_circuit(params_list, values_5)
compiled_circuit = transpile(qc, simulator, seed_transpiler=SEED)

if AerSimulator is not None:
    result = simulator.run(
        compiled_circuit, shots=shots, seed_simulator=SEED_SIMULATOR
    ).result()
else:
    result = simulator.run(compiled_circuit, shots=shots).result()
counts = result.get_counts()

# pri istom broju shots — deterministički izbor ako ima izjednačenje
most_probable = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
predicted_combination = bitstring_to_loto_with_7(
    _counts_key_to_int(most_probable),
    n_qubits=num_qubits,
    num_pos=num_positions,
)

for pos in range(7):
    print(f"\n--- QSVR pozicija {pos + 1} ---")
    print(predicted_combination[pos])

print()
print("\n=== QSVR Predviđena loto kombinacija (7) ===")
print(" ".join(str(x) for x in predicted_combination))

# opcionalno: vizualizacija poslednje pozicije
plot_histogram(counts)

print()



"""
python3 QSVR_qc25_7_v2.py
"""

"""
=== QSVR qc25: jedna simulacija (5×5 qubit-a), zatim 7 brojeva (QCBM obrazac) ===

--- QSVR pozicija 1 ---
20

--- QSVR pozicija 2 ---
19

--- QSVR pozicija 3 ---
17

--- QSVR pozicija 4 ---
21

--- QSVR pozicija 5 ---
15

--- QSVR pozicija 6 ---
32

--- QSVR pozicija 7 ---
33


=== QSVR Predviđena loto kombinacija (7) ===
20 x y z 15 32 33
"""

