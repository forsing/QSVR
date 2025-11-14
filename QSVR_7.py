# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html
import warnings
warnings.filterwarnings("ignore", message="No gradient function provided")


"""
QSVR (Quantum Support Vector Regressor)
"""

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

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# =========================
# Učitavanje CSV fajla
# =========================
csv_path = '/data/loto7_4510_k89.csv'
df = pd.read_csv(csv_path, header=None)  # bez headera

# =========================
# Koristimo samo zadnjih N=1000 za test
# =========================
N = 100 # zadnjih 100 kombinacija
# N = 4510 # sve kombinacije 

df = df.tail(N).reset_index(drop=True)

# 3. Priprema podataka
X = df.iloc[:, :-1].values   # prvih 6 brojeva
y_full = df.values           # svih 7 brojeva (6+1)

# Skaliranje X
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X).astype(np.float64)

# =========================
# Funkcija za kernel matricu sa progress bar-om
# =========================
def compute_kernel_matrix(X_scaled):
    n_features = X_scaled.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1, entanglement='linear')
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    backend = AerSimulator()
    kernel._quantum_instance = backend

    def compute_row(i):
        return [kernel.evaluate([X_scaled[i]], [X_scaled[j]])[0][0] for j in range(len(X_scaled))]

    kernel_matrix = np.zeros((len(X_scaled), len(X_scaled)))
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(compute_row, range(len(X_scaled))),
                            total=len(X_scaled),
                            desc="Računanje kernel matrice"))
    for i, row in enumerate(results):
        kernel_matrix[i, :] = row
    return kernel_matrix, feature_map, kernel

# =========================
# Treniranje i predikcija po brojevima koristeći QSVR
# =========================
predicted_combination = []
print()
for i in range(7):  # 6 brojeva + dodatni broj
    print(f"\n--- Treniranje QSVR modela za broj {i+1} ---")

    # y za trenutni model (float64)
    y = y_full[:, i].astype(np.float64)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()  # 1D float64

    # kernel matrica, feature_map i kernel (progress bar)
    kernel_matrix, feature_map, kernel = compute_kernel_matrix(X_scaled)

    # Kreiranje QSVR modela
    qsvr = QSVR(
        quantum_kernel=kernel,
        C=1.0,
        epsilon=0.1
    )


    total_iters = len(X_scaled)
    pbar = tqdm(total=total_iters, desc=f"Broj {i+1}")

    # Treniranje: QSVR očekuje X_scaled i 1D y_scaled
    qsvr.fit(X_scaled, y_scaled)

    pbar.update(1)
    pbar.close()

    # Predikcija sledeće kombinacije
    last_comb_scaled = scaler_X.transform([X[-1]]).astype(np.float64)
    pred_scaled = qsvr.predict(last_comb_scaled)
    pred = scaler_y.inverse_transform(np.array(pred_scaled).reshape(-1,1)).round().astype(int)[0][0]

    predicted_combination.append(int(pred))
    print(f"Predikcija za broj {i+1}: {pred}")
print()

"""
--- Treniranje QSVR modela za broj 1 ---
Računanje kernel matrice: 100%|██████████████| 100/100 [00:19<00:00,  5.05it/s]
Broj 1:   1%|▎                                 | 1/100 [00:06<11:15,  6.82s/it]
Predikcija za broj 1: 4

--- Treniranje QSVR modela za broj 2 ---
Računanje kernel matrice: 100%|██████████████| 100/100 [00:18<00:00,  5.29it/s]
Broj 2:   1%|▎                                 | 1/100 [00:06<11:19,  6.86s/it]
Predikcija za broj 2: 9

--- Treniranje QSVR modela za broj 3 ---
Računanje kernel matrice: 100%|██████████████| 100/100 [00:18<00:00,  5.40it/s]
Broj 3:   1%|▎                                 | 1/100 [00:06<11:22,  6.90s/it]
Predikcija za broj 3: x

--- Treniranje QSVR modela za broj 4 ---
Računanje kernel matrice: 100%|██████████████| 100/100 [00:18<00:00,  5.35it/s]
Broj 4:   1%|▎                                 | 1/100 [00:06<11:20,  6.87s/it]
Predikcija za broj 4: x

--- Treniranje QSVR modela za broj 5 ---
Računanje kernel matrice: 100%|██████████████| 100/100 [00:18<00:00,  5.33it/s]
Broj 5:   1%|▎                                 | 1/100 [00:06<11:24,  6.91s/it]
Predikcija za broj 5: x

--- Treniranje QSVR modela za broj 6 ---
Računanje kernel matrice: 100%|██████████████| 100/100 [00:18<00:00,  5.29it/s]
Broj 6:   1%|▎                                 | 1/100 [00:07<11:39,  7.07s/it]
Predikcija za broj 6: 35

--- Treniranje QSVR modela za broj 7 ---
Računanje kernel matrice: 100%|██████████████| 100/100 [00:19<00:00,  5.23it/s]
Broj 7:   1%|▎                                 | 1/100 [00:06<11:21,  6.89s/it]
Predikcija za broj 7: 37
"""


# =========================
# Rezultata
# =========================
print()
print("\n=== Predviđena sledeća loto kombinacija (7/39) ===")
print(" ".join(str(num) for num in predicted_combination))
print()
"""
N = 100

=== Predviđena sledeća loto kombinacija (7/39) ===
4 9 x x x 35 37
"""
