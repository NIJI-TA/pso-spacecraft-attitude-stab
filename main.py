import numpy as np

from dynamics import simulate_dinamics
from PSO import run_pso
from SA import run_sa
from plots import (
    plot_angles_two_methods,
    plot_kl_two_methods,
    plot_km_two_methods,
    plot_hl_two_methods,
    plot_hm_two_methods,
)
from Rodrigo_Hamilton_and_matrix import (
    to_Rodrigo_Hamilton,
    from_Rodrigo_Hamilton,
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
s = 20  # число частиц (PSO)
n = 1  # число интервалов для коэффициентов усиления
max_iter_pso = 100
max_iter_sa = 1500
h = 0.02
u_max = 25.0
param_min = np.array([0.0, 1.0e6, 500.0, 0.5e9])
param_max = np.array([50.0, 7.0e6, 3500.0, 3.5e9])

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


# === Результаты
print()
print("=== === === Результаты PSO === === ===")
print()
print()
print("Параметры управления:")
print("kl", gbest[0])
print("km", gbest[1])
print("hl", gbest[2])
print("hm", gbest[3])
print()
print("Значение целевой функции в gbest:", value_at_gbest)
print("=== === === === === === === === ===")
print()
print("=== === === Результаты SA === === ===")
print()
print()
print("Параметры управления:")
print("kl", sabest[0])
print("km", sabest[1])
print("hl", sabest[2])
print("hm", sabest[3])
print()
print(f"Значение целевой функции в best_energy (лучшая энергия): {value_at_sabest}")
print("=== === === === === === === === ===")

print()
print("=== === === ИТОГ ПО ЦЕЛЕВЫМ ФУНКЦИЯМ === === ===")
print(value_at_gbest)
print(value_at_sabest)
print()
print()

# === Строим графики
y_pso, u_vec_pso = simulate_dinamics(
    Omega_x0, Omega_y0, Omega_z0, lambd_vec_0, matrix_1, gbest, n, u_max, h
)

lambd_vec = y_pso[0:4, :].T
Omega_x = y_pso[4, :]
Omega_y = y_pso[5, :]
Omega_z = y_pso[6, :]

phi_values_pso = np.array([from_Rodrigo_Hamilton(lam)[0] for lam in lambd_vec])
psi_values_pso = np.array([from_Rodrigo_Hamilton(lam)[1] for lam in lambd_vec])
theta_values_pso = np.array([from_Rodrigo_Hamilton(lam)[2] for lam in lambd_vec])

kl_pso = gbest[0, :]
km_pso = gbest[1, :]
hl_pso = gbest[2, :]
hm_pso = gbest[3, :]


y_sa, u_vec_sa = simulate_dinamics(
    Omega_x0, Omega_y0, Omega_z0, lambd_vec_0, matrix_1, sabest, n, u_max, h
)

lambd_vec = y_sa[0:4, :].T
Omega_x = y_sa[4, :]
Omega_y = y_sa[5, :]
Omega_z = y_sa[6, :]

phi_values_sa = np.array([from_Rodrigo_Hamilton(lam)[0] for lam in lambd_vec])
psi_values_sa = np.array([from_Rodrigo_Hamilton(lam)[1] for lam in lambd_vec])
theta_values_sa = np.array([from_Rodrigo_Hamilton(lam)[2] for lam in lambd_vec])

kl_sa = sabest[0, :]
km_sa = sabest[1, :]
hl_sa = sabest[2, :]
hm_sa = sabest[3, :]

plot_angles_two_methods(
    phi_values_pso,
    psi_values_pso,
    theta_values_pso,
    phi_values_sa,
    psi_values_sa,
    theta_values_sa,
    u_vec_pso,
    u_vec_sa,
)

plot_kl_two_methods(kl_pso, kl_sa, n, u_max, param_min, param_max)
plot_km_two_methods(km_pso, km_sa, n, u_max, param_min, param_max)
plot_hl_two_methods(hl_pso, hl_sa, n, u_max, param_min, param_max)
plot_hm_two_methods(hm_pso, hm_sa, n, u_max, param_min, param_max)
