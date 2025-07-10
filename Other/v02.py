import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Характеристики КА и другие начальные данные
Q = 5e-10  # заряд экрана
# -- Компоненты тензора инерции инерции
J_A = 1000
DELTA = 1.2  # B/J_A
EPSILON = 0.8  # J_C/J_A

# J_B = delta * J_A
# J_C = epsilon * J_A

# J = np.diag([J_A, J_A, J_C])  # тензор инерции

# -- Орбитальные параметры
R = 7.0e6  # радиус орбиты в метрах
I_ANGLE = np.radians(30)  # наклонение орбиты
F = 3.98603e14  # гравитационный параметр Земли
OMEGA_0 = np.sqrt(F / R**3)  # угловая скорость орбитальной системы
OMEGA_E = 7.292132e-5  # угловая скорость вращения земли
R_E = 6.3712e6  # радиус Земли

# -- Магнитное поле Земли (модель - "Прямой диполь")
G10 = -2.93763e-5  # гауссов коэффициент
B_XI = 0.0
B_ETA = 0.0
B_ZETA = -((R_E**3) * G10) / (R**3)
B = np.array([B_XI, B_ETA, B_ZETA])  # Вектор магнитной индукции

# -- Прочее
K0 = 1  # коэффициент для автоматической коррекции нормы кватерниона


# Вычисление значений параметров Родрига-Гамильтона
def to_Rodrigo_Hamilton(phi, psi, theta):
    cos_phi, cos_psi, cos_theta = np.cos(phi / 2), np.cos(psi / 2), np.cos(theta / 2)
    sin_phi, sin_psi, sin_theta = np.sin(phi / 2), np.sin(psi / 2), np.sin(theta / 2)

    lambd_0 = cos_phi * cos_psi * cos_theta + sin_phi * sin_psi * sin_theta
    lambd_1 = sin_phi * cos_psi * cos_theta - cos_phi * sin_psi * sin_theta
    lambd_2 = cos_phi * cos_psi * sin_theta + sin_phi * sin_psi * cos_theta
    lambd_3 = cos_phi * sin_psi * cos_theta - sin_phi * cos_psi * sin_theta

    lambd_vec = np.array([lambd_0, lambd_1, lambd_2, lambd_3])

    return lambd_vec


def from_Rodrigo_Hamilton(lambd_vec):
    matrix = to_transition_matrix(lambd_vec)

    gam1, gam2, gam3 = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    alph1, beta1 = matrix[0, 0], matrix[1, 0]

    phi = np.arctan(gam2 / gam3)
    psi = np.arctan(beta1 / alph1)
    theta = -np.arcsin(gam1)

    angles = np.array([phi, psi, theta])

    return angles


# Вычисл. матрицы перехода от системы главн. центр. осей инерции к орбитальной сист. координат
def to_transition_matrix(lambd_vec):
    lambd_0, lambd_1, lambd_2, lambd_3 = lambd_vec

    alph_1 = (
        lambd_0 * lambd_0 + lambd_1 * lambd_1 - lambd_2 * lambd_2 - lambd_3 * lambd_3
    )
    alph_2 = 2 * (lambd_1 * lambd_2 - lambd_0 * lambd_3)
    alph_3 = 2 * (lambd_1 * lambd_3 + lambd_0 * lambd_2)

    beta_1 = 2 * (lambd_1 * lambd_2 + lambd_0 * lambd_3)
    beta_2 = (
        lambd_0 * lambd_0 - lambd_1 * lambd_1 + lambd_2 * lambd_2 - lambd_3 * lambd_3
    )
    beta_3 = 2 * (lambd_2 * lambd_3 - lambd_0 * lambd_1)

    gam_1 = 2 * (lambd_1 * lambd_3 - lambd_0 * lambd_2)
    gam_2 = 2 * (lambd_2 * lambd_3 + lambd_0 * lambd_1)
    gam_3 = (
        lambd_0 * lambd_0 - lambd_1 * lambd_1 - lambd_2 * lambd_2 + lambd_3 * lambd_3
    )

    matrix = np.array(
        [[alph_1, alph_2, alph_3], [beta_1, beta_2, beta_3], [gam_1, gam_2, gam_3]]
    )

    return matrix


def calc_vC(u):
    vC_1 = R * (OMEGA_0 - OMEGA_E * np.cos(I_ANGLE))
    vC_2 = R * (OMEGA_E * np.sin(I_ANGLE) * np.cos(u))
    vC_3 = 0

    vC = np.array([vC_1, vC_2, vC_3])

    return vC


def calc_M_L(matrix, matrix_1, vC, kl):
    vC_x_B = np.cross(vC, B)
    t_0 = np.matmul(matrix_1.T, vC_x_B)
    p = Q * kl * t_0
    tt = np.matmul(matrix.T, vC_x_B)

    m_L = np.cross(p, tt)
    return m_L


def calc_domega(beta_1, beta_2, beta_3, Omega_x, Omega_y, Omega_z):
    domega_1 = OMEGA_0 * (Omega_x - beta_1)
    domega_2 = OMEGA_0 * (Omega_y - beta_2)
    domega_3 = OMEGA_0 * (Omega_z - beta_3)

    domega = np.array([domega_1, domega_2, domega_3])

    return domega


def calc_M_M(matrix, matrix_1, km):
    I_vec = km * np.matmul(matrix_1.T, B)

    m_m = np.cross(I_vec, np.matmul(matrix.T, B))

    return m_m


def calc_M_DL(matrix, vC, domega, hl):
    tt = np.matmul(matrix.T, np.cross(vC, B))

    m_dl = Q * hl * (np.cross(np.cross(domega, tt), tt))
    return m_dl


def calc_M_DM(matrix, domega, hm):
    m_dm = hm * (
        np.cross(np.cross(domega, np.matmul(matrix.T, B)), np.matmul(matrix.T, B))
    )

    return m_dm


def simulate_dinamics(
    Omega_x0,
    Omega_y0,
    Omega_z0,
    lambd_vec_0,
    matrix_1,
    Agent,
    n,
    u_max,
    h=0.05,
    method="RK45",
):
    times = np.arange(0, u_max + h, h)
    interval = u_max / n

    def dynamics(u, state):
        lambd_0, lambd_1, lambd_2, lambd_3, Omega_x, Omega_y, Omega_z = state

        index = int(u // interval)
        index = min(index, n - 1)

        kl = Agent[0, index]
        km = Agent[1, index]
        hl = Agent[2, index]
        hm = Agent[3, index]

        matrix = to_transition_matrix([lambd_0, lambd_1, lambd_2, lambd_3])
        beta_1, beta_2, beta_3 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        vC = calc_vC(u)
        domega = calc_domega(beta_1, beta_2, beta_3, Omega_x, Omega_y, Omega_z)

        m_L = calc_M_L(matrix, matrix_1, vC, kl)
        m_M = calc_M_M(matrix, matrix_1, km)
        m_DL = calc_M_DL(matrix, vC, domega, hl)
        m_DM = calc_M_DM(matrix, domega, hm)

        # Параметры из системы
        dlambd_0 = 0.5 * (
            -lambd_1 * Omega_x
            - lambd_2 * Omega_y
            - lambd_3 * Omega_z
            + lambd_2
            - K0 * lambd_0 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dlambd_1 = 0.5 * (
            lambd_0 * Omega_x
            + lambd_2 * Omega_z
            - lambd_3 * Omega_y
            - lambd_3
            - K0 * lambd_1 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dlambd_2 = 0.5 * (
            lambd_0 * Omega_y
            + lambd_3 * Omega_x
            - lambd_1 * Omega_z
            - lambd_0
            - K0 * lambd_2 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dlambd_3 = 0.5 * (
            lambd_0 * Omega_z
            + lambd_1 * Omega_y
            - lambd_2 * Omega_x
            + lambd_1
            - K0 * lambd_3 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dOmega_x = -(EPSILON - DELTA) * (Omega_y * Omega_z - beta_2 * beta_3) + (
            m_L[0] + m_M[0] + m_DL[0] + m_DM[0]
        ) / (J_A * OMEGA_0**2)
        dOmega_y = -((1 - EPSILON) / DELTA) * (Omega_z * Omega_x - beta_1 * beta_3) + (
            m_L[1] + m_M[1] + m_DL[1] + m_DM[1]
        ) / (DELTA * J_A * OMEGA_0**2)
        dOmega_z = -((DELTA - 1) / EPSILON) * (Omega_x * Omega_y - beta_2 * beta_1) + (
            m_L[2] + m_M[2] + m_DL[2] + m_DM[2]
        ) / (EPSILON * J_A * OMEGA_0**2)

        return [dlambd_0, dlambd_1, dlambd_2, dlambd_3, dOmega_x, dOmega_y, dOmega_z]

    initial_state = np.concatenate([lambd_vec_0, [Omega_x0, Omega_y0, Omega_z0]])
    # Решаем систему используя функцию solve_ivp
    result = solve_ivp(dynamics, [0, u_max], initial_state, t_eval=times, method=method)

    # Возвращаем массив решений result.y и массив времени result.t
    return result.y, result.t


# Считаем целевую функцию L
def function_phi(
    Omega_x0,
    Omega_y0,
    Omega_z0,
    lambd_vec_0,
    lambd_vec_1,
    matrix_1,
    Agent,
    n,
    u_max,
    h,
):
    # y - массив решений, u - массив времени
    y, u = simulate_dinamics(
        Omega_x0,
        Omega_y0,
        Omega_z0,
        lambd_vec_0,
        matrix_1,
        Agent,
        n,
        u_max,
        h,
    )

    lambd_vec = y[0:4, :].T
    omega_vec = y[4:7, :].T

    # Вычисление ошибок
    LambdaDif = np.sum((lambd_vec - lambd_vec_1) ** 2, axis=1)
    OmegaX = OMEGA_0 * (
        omega_vec[:, 0]
        - 2 * (lambd_vec[:, 1] * lambd_vec[:, 2] - lambd_vec[:, 0] * lambd_vec[:, 3])
    )
    OmegaY = OMEGA_0 * (
        omega_vec[:, 1]
        - 2 * (lambd_vec[:, 0] * lambd_vec[:, 1] + lambd_vec[:, 2] * lambd_vec[:, 3])
    )
    OmegaZ = OMEGA_0 * (
        omega_vec[:, 2]
        - 2 * (lambd_vec[:, 2] * lambd_vec[:, 3] - lambd_vec[:, 0] * lambd_vec[:, 1])
    )
    omegaShDif = OmegaX**2 + OmegaY**2 + OmegaZ**2

    total_error = LambdaDif + omegaShDif

    phi = (h / 2) * np.sum(total_error[:-1] + total_error[1:])

    return phi


# === === == == == == == = = = = = = = = - - - - - - - - - - - - - - - -
# === Начальные условия
phi0 = 0.1
psi0 = -0.2
theta0 = 0.0
Omega_x0 = 0.3
Omega_y0 = 1.1
Omega_z0 = 0.5

# === Желаемая ориетация
phi1 = 0.6
psi1 = 0.2
theta1 = 0.4

# === Предварительные вычисления
# начальные значения параметров Родрига-Гамильтона
lambd_vec_0 = to_Rodrigo_Hamilton(phi0, psi0, theta0)
# значения параметров Родрига-Гамильтона
lambd_vec_1 = to_Rodrigo_Hamilton(phi1, psi1, theta1)

# матрицы перехода (желаемой ориентаций)
matrix_1 = to_transition_matrix(lambd_vec_1)

s = 20  # Число частиц
h = 0.05
n = 1  # Число интервалов
max_iter = 800
w, c1, c2 = 0.7, 1.4, 1.4
param_min = np.array([0.1, 1.0e6, 500.0, 0.5e9])
param_max = np.array([50.0, 7.0e6, 3500.0, 3.5e9])
u_max = 25.0

particles = np.zeros((s, 4, n))
velocities = np.zeros((s, 4, n))

for i in range(4):
    particles[:, i, :] = np.random.uniform(param_min[i], param_max[i], (s, n))
    velocities[:, i, :] = np.random.uniform(
        0.15 * param_min[i], 0.15 * param_max[i], (s, n)
    )

pbest = particles.copy()
value_at_pbest = np.full(s, np.inf)
# глобальное лучшее на первой итерации сразу перезапишется
gbest = particles[0].copy()
# значение целевой функции в gbest
value_at_gbest = np.inf

# PSO
# Запускаем главный цикл, используем tqdm для вывода шкалы процесса
for i in tqdm(range(max_iter), desc="Итерации", unit="итерация"):
    # Идем по каждой частице
    for j in range(s):
        # Моделируем
        current_function = function_phi(
            Omega_x0,
            Omega_y0,
            Omega_z0,
            lambd_vec_0,
            lambd_vec_1,
            matrix_1,
            particles[j],
            n,
            u_max,
            h,
        )
        # Ищем индивидуально лучшую минимизацию phi
        if current_function < value_at_pbest[j]:
            # Eсли текущее значение меньше индивидуально лучшего, то запоминаем значение и управление
            value_at_pbest[j] = current_function
            pbest[j] = particles[j].copy()
        # Ищем глобально лучшую минимизацию phi
        if current_function < value_at_gbest:
            # Eсли текущее значение меньше глобально лучшего, то запоминаем значение и управление
            value_at_gbest = current_function
            gbest = particles[j].copy()

    # Обновление скоростей и позиций согласно методу
    r1 = np.random.rand(s, 4, n)
    r2 = np.random.rand(s, 4, n)
    velocities = (
        w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
    )
    particles += velocities
    # Ограничение управления
    for k in range(4):
        particles[:, k, :] = np.clip(particles[:, k, :], param_min[k], param_max[k])

print("Лучшее управление:", gbest)
print()

y, u = simulate_dinamics(
    Omega_x0,
    Omega_y0,
    Omega_z0,
    lambd_vec_0,
    matrix_1,
    gbest,
    n,
    u_max,
    h,
)

lambd_vec = y[0:4, :].T
phi_values = np.array([from_Rodrigo_Hamilton(lam)[0] for lam in lambd_vec])
psi_values = np.array([from_Rodrigo_Hamilton(lam)[1] for lam in lambd_vec])
theta_values = np.array([from_Rodrigo_Hamilton(lam)[2] for lam in lambd_vec])


# Строим график
plt.figure(figsize=(10, 5))
plt.plot(u, phi_values, label="φ (phi)", color="blue")
plt.plot(u, psi_values, label="ψ (psi)", color="green")
plt.plot(u, theta_values, label="θ (theta)", color="red")
plt.xlabel("Время, сек")
plt.ylabel("рад.")
plt.title("Эволюция углов φ, ψ, θ во времени")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
