import numpy as np

from PSO import run_pso
from SA import run_sa
from plots import plot_boxplot_objective
from Rodrigo_Hamilton_and_matrix import (
    to_Rodrigo_Hamilton,
    to_transition_matrix,
)


# === Начальные условия
phi0 = -0.2
psi0 = 0.0
theta0 = 0.1
Omega_x0 = 0.3
Omega_y0 = 1.1
Omega_z0 = 0.5

# --- Желаемая ориетация
phi1 = 0.2
psi1 = 0.4
theta1 = 0.6

# === Предварительные вычисления
# начальные значения параметров Родрига-Гамильтона
lambd_vec_0 = to_Rodrigo_Hamilton(phi0, psi0, theta0)
# желаемые значения параметров Родрига-Гамильтона
lambd_vec_1 = to_Rodrigo_Hamilton(phi1, psi1, theta1)

# матрица перехода (желаемой ориентаций)
matrix_1 = to_transition_matrix(lambd_vec_1)

# === Параметры! PSO + SA
s = 30  # число частиц (PSO)
n = 50  # число интервалов для коэффициентов усиления
max_iter_pso = 100
max_iter_sa = 2000
h = 0.02
u_max = 25.0
param_min = np.array([0.0, 1.0e6, 500.0, 0.5e9])
param_max = np.array([50.0, 7.0e6, 3500.0, 3.5e9])

N_runs = 20

J_pso_all = []
J_sa_all = []

for i in range(N_runs):
    print(f"Запуск {i + 1}/{N_runs}")

    # === Запускаем PSO
    gbest, value_at_gbest, convergence_pso = run_pso(
        s,
        n,
        max_iter_pso,
        h,
        u_max,
        Omega_x0,
        Omega_y0,
        Omega_z0,
        param_min,
        param_max,
        lambd_vec_0,
        lambd_vec_1,
        matrix_1,
    )

    # === Запускаем SA
    sabest, value_at_sabest = run_sa(
        n,
        max_iter_sa,
        h,
        u_max,
        Omega_x0,
        Omega_y0,
        Omega_z0,
        param_min,
        param_max,
        lambd_vec_0,
        lambd_vec_1,
        matrix_1,
    )

    # === Сохраняем значения целевой функции
    J_pso_all.append(value_at_gbest)
    J_sa_all.append(value_at_sabest)


plot_boxplot_objective(J_pso_all, J_sa_all)
