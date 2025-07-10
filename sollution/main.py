import numpy as np


from dynamics import simulate_dinamics, calc_domega_vec
from PSO import run_pso
from plots import (
    get_angles,
    get_domega,
    get_lambd,
    get_convergence,
    get_kl,
    get_km,
    get_hl,
    get_hm,
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

# === PSO
s = 20
n = 1
max_iter = 600
h = 0.02
u_max = 60.0
param_min = np.array([0.0, 1.0e6, 500.0, 0.5e9])
param_max = np.array([50.0, 7.0e6, 3500.0, 3.5e9])

gbest, value_at_gbest, convergence = run_pso(
    s,
    n,
    max_iter,
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
print("=== === === Результаты === === ===")
print()
print("Параметры управления:")
print("kl", gbest[0])
print("km", gbest[1])
print("hl", gbest[2])
print("hm", gbest[3])
print()
print("Значение целевой функции в gbest:", value_at_gbest)
print("=== === === === === === === === ===")

# === Построение графиков
y, u_vec = simulate_dinamics(
    Omega_x0, Omega_y0, Omega_z0, lambd_vec_0, matrix_1, gbest, n, u_max, h
)

lambd_vec = y[0:4, :].T
Omega_x = y[4, :]
Omega_y = y[5, :]
Omega_z = y[6, :]

phi_values = np.array([from_Rodrigo_Hamilton(lam)[0] for lam in lambd_vec])
psi_values = np.array([from_Rodrigo_Hamilton(lam)[1] for lam in lambd_vec])
theta_values = np.array([from_Rodrigo_Hamilton(lam)[2] for lam in lambd_vec])


l2 = lambd_vec[:, 2]
l3 = lambd_vec[:, 3]

beta_1 = 2 * (lambd_vec[:, 1] * lambd_vec[:, 2] + lambd_vec[:, 0] * lambd_vec[:, 3])
beta_2 = (
    lambd_vec[:, 0] * lambd_vec[:, 0]
    - lambd_vec[:, 1] * lambd_vec[:, 1]
    + lambd_vec[:, 2] * lambd_vec[:, 2]
    - lambd_vec[:, 3] * lambd_vec[:, 3]
)
beta_3 = 2 * (lambd_vec[:, 2] * lambd_vec[:, 3] - lambd_vec[:, 0] * lambd_vec[:, 1])
domega_1, domega_2, domega_3 = calc_domega_vec(
    beta_1, beta_2, beta_3, Omega_x, Omega_y, Omega_z
)

kl = gbest[0, :]
km = gbest[1, :]
hl = gbest[2, :]
hm = gbest[3, :]

get_angles(phi_values, psi_values, theta_values, u_vec)
get_domega(domega_1, domega_2, domega_3, u_vec)
get_lambd(lambd_vec, u_vec)
get_convergence(convergence, max_iter)
get_kl(kl, n, u_max)
get_km(km, n, u_max)
get_hl(hl, n, u_max)
get_hm(hm, n, u_max)
