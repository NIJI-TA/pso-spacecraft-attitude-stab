import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


# Вычисление значений параметров Родрига-Гамильтона
def to_Rodrigo_Hamilton(phi, psi, theta):
    cos_phi, cos_psi, cos_theta = np.cos(phi / 2), np.cos(psi / 2), np.cos(theta / 2)
    sin_phi, sin_psi, sin_theta = np.sin(phi / 2), np.sin(psi / 2), np.sin(theta / 2)

    lambd_0 = cos_phi * cos_psi * cos_theta + sin_phi * sin_psi * sin_theta
    lambd_1 = sin_phi * cos_psi * cos_theta - cos_phi * sin_psi * sin_theta
    lambd_2 = cos_phi * cos_psi * sin_theta + sin_phi * sin_psi * cos_theta
    lambd_3 = cos_phi * sin_psi * cos_theta - sin_phi * cos_psi * sin_theta

    return np.array([lambd_0, lambd_1, lambd_2, lambd_3])


# Вычисл. матрицы перехода от сист. главн. центр. осей инерции к орбит. сист. координат
def to_transition_matrix(lambd):
    lambd_0, lambd_1, lambd_2, lambd_3 = lambd

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

    return np.array(
        [[alph_1, alph_2, alph_3], [beta_1, beta_2, beta_3], [gam_1, gam_2, gam_3]]
    )


def calc_vC(R, omega_0, omega_E, i_angle, u):
    vC_1 = R * (omega_0 - omega_E * np.cos(i_angle))
    vC_2 = R * (omega_E * np.sin(i_angle) * np.cos(u))
    vC_3 = 0

    return np.array([vC_1, vC_2, vC_3])


def calc_M_L(Matrix, Matrix_0, vC, B, kl, Q):
    vC_x_B = np.cross(vC, B)
    T_0 = np.matmul(Matrix_0.T, vC_x_B)
    P = Q * kl * T_0
    TT = np.matmul(Matrix.T, vC_x_B)
    return np.cross(P, TT)


def calc_domega(omega_0, beta_1, beta_2, beta_3, Omega_x, Omega_y, Omega_z):
    domega_1 = omega_0 * (Omega_x - beta_1)
    domega_2 = omega_0 * (Omega_y - beta_2)
    domega_3 = omega_0 * (Omega_z - beta_3)
    return np.array([domega_1, domega_2, domega_3])


def calc_M_M(Matrix, Matrix_0, B, km):
    I_vec = km * np.matmul(Matrix_0.T, B)
    return np.cross(I_vec, np.matmul(Matrix.T, B))


def calc_M_DL(Matrix, vC, B, hl, domega, Q):
    TT = np.matmul(Matrix.T, np.cross(vC, B))
    return Q * hl * (np.cross(np.cross(domega, TT), TT))


def calc_M_DM(Matrix, B, hm, domega):
    return hm * np.cross(
        np.cross(domega, np.matmul(Matrix.T, B)), np.matmul(Matrix.T, B)
    )


def simulate_dinamics(
    Agent,
    lambda_0,
    Omega_x0,
    Omega_y0,
    Omega_z0,
    Matrix_0,
    A,
    k0,
    Q,
    B,
    R,
    epsilon,
    delta,
    omega_0,
    omega_E,
    i_angle,
    N,
    t_max,
    h=0.05,
    method="RK45",
):
    times = np.arange(0, t_max + h, h)
    interval = t_max / N

    def dynamics(u, state):
        lambd_0, lambd_1, lambd_2, lambd_3, Omega_x, Omega_y, Omega_z = state

        index = int(u // interval)
        index = min(index, N - 1)

        kl = Agent[0, index]
        km = Agent[1, index]
        hl = Agent[2, index]
        hm = Agent[3, index]

        Matrix = to_transition_matrix([lambd_0, lambd_1, lambd_2, lambd_3])

        beta_1, beta_2, beta_3 = Matrix[1, 0], Matrix[1, 1], Matrix[1, 2]

        vC = calc_vC(R, omega_0, omega_E, i_angle, u)
        domega = calc_domega(omega_0, beta_1, beta_2, beta_3, Omega_x, Omega_y, Omega_z)

        M_L = calc_M_L(Matrix, Matrix_0, vC, B, kl, Q)
        M_M = calc_M_M(Matrix, Matrix_0, B, km)
        M_DL = calc_M_DL(Matrix, vC, B, hl, domega, Q)
        M_DM = calc_M_DM(Matrix, B, hm, domega)

        # Параметры из системы
        dlambd_0 = 0.5 * (
            -lambd_1 * Omega_x
            - lambd_2 * Omega_y
            - lambd_3 * Omega_z
            + lambd_2
            - k0 * lambd_0 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dlambd_1 = 0.5 * (
            lambd_0 * Omega_x
            + lambd_2 * Omega_z
            - lambd_3 * Omega_y
            - lambd_3
            - k0 * lambd_1 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dlambd_2 = 0.5 * (
            lambd_0 * Omega_y
            + lambd_3 * Omega_x
            - lambd_1 * Omega_z
            - lambd_0
            - k0 * lambd_2 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dlambd_3 = 0.5 * (
            lambd_0 * Omega_z
            + lambd_1 * Omega_y
            - lambd_2 * Omega_x
            + lambd_1
            - k0 * lambd_3 * (lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2 - 1)
        )
        dOmega_x = -(epsilon - delta) * (Omega_y * Omega_z - beta_2 * beta_3) + (
            M_L[0] + M_M[0] + M_DL[0] + M_DM[0]
        ) / (A * omega_0**2)
        dOmega_y = -((1 - epsilon) / delta) * (Omega_z * Omega_x - beta_1 * beta_3) + (
            M_L[1] + M_M[1] + M_DL[1] + M_DM[1]
        ) / (delta * A * omega_0**2)
        dOmega_z = -((delta - 1) / epsilon) * (Omega_x * Omega_y - beta_2 * beta_1) + (
            M_L[2] + M_M[2] + M_DL[2] + M_DM[2]
        ) / (epsilon * A * omega_0**2)

        return [dlambd_0, dlambd_1, dlambd_2, dlambd_3, dOmega_x, dOmega_y, dOmega_z]

    initial_state = np.concatenate([lambda_0, [Omega_x0, Omega_y0, Omega_z0]])

    # Решаем систему используя функцию solve_ivp
    result = solve_ivp(dynamics, [0, t_max], initial_state, t_eval=times, method=method)
    # Возвращаем массив решений result.y и массив времени result.t
    return result.y, result.t


# Считаем целевую функцию L
def function(
    Agent,
    lambda_1,
    lambda_0,
    Omega_x0,
    Omega_y0,
    Omega_z0,
    Matrix_0,
    R,
    epsilon,
    delta,
    omega_0,
    omega_E,
    i_angle,
    t_max,
    N,
    h,
):
    # y - массив решений, t - массив времени
    y, t = simulate_dinamics(
        Agent,
        lambda_0,
        Omega_x0,
        Omega_y0,
        Omega_z0,
        Matrix_0,
        A,
        k0,
        Q,
        B,
        R,
        epsilon,
        delta,
        omega_0,
        omega_E,
        i_angle,
        N,
        t_max,
        h,
    )

    lambda_vec = y[0:4, :].T
    omega_vec = y[4:7, :].T

    # Вычисление ошибок
    LambdaDif = np.sum((lambda_vec - lambda_1) ** 2, axis=1)
    OmegaX = omega_0 * (
        omega_vec[:, 0]
        - 2
        * (lambda_vec[:, 1] * lambda_vec[:, 2] - lambda_vec[:, 0] * lambda_vec[:, 3])
    )
    OmegaY = omega_0 * (
        omega_vec[:, 1]
        - 2
        * (lambda_vec[:, 0] * lambda_vec[:, 1] + lambda_vec[:, 2] * lambda_vec[:, 3])
    )
    OmegaZ = omega_0 * (
        omega_vec[:, 2]
        - 2
        * (lambda_vec[:, 2] * lambda_vec[:, 3] - lambda_vec[:, 0] * lambda_vec[:, 1])
    )
    omegaShDif = OmegaX**2 + OmegaY**2 + OmegaZ**2

    total_error = LambdaDif + omegaShDif

    Phi = (h / 2) * np.sum(total_error[:-1] + total_error[1:])

    return Phi


# === Характеристики КА и другие начальные данные
Q = 5e-10  # заряд экрана
# -- Компоненты тензора инерции инерции
A = 1000
delta = 1.2  # B/A
epsilon = 0.8  # C/A

# B = delta * A
# C = epsilon * A
# J = np.diag([A, B, C])  # тензор инерции

# -- О
R = 7.0e6  # радиус орбиты в метрах
i_angle = np.radians(30)  # наклонение орбиты

f = 3.98603e14  # гравитационный параметр Земли
omega_0 = np.sqrt(f / R**3)  # угловая скорость орбитальной системы

omega_E = 7.292132e-5  # угловая скорость вращения земли
R_E = 6.3712e6  # радиус Земли

# -- Магнитное поле Земли (модель - "Прямой диполь")
g10 = -2.93763e-5  # гауссов коэффициент
B_xi = 0.0
B_eat = 0.0
B_zeta = -((R_E**3) * g10) / (R**3)
B = np.array([B_xi, B_eat, B_zeta])  # Вектор магнитной индукции

# -- Прочее
k0 = 1  # коэффициент для автоматической коррекции нормы кватерниона

# === Начальные условия
phi0 = -0.2
psi0 = 0.0
theta0 = 0.1
Omega_x0 = 0.3
Omega_y0 = 1.1
Omega_z0 = 0.5

# === Желаемая ориетация
phi1 = 0.2
psi1 = 0.4
theta1 = 0.6


# === Предварительные вычисления
# начальные значения параметров Родрига-Гамильтона
lambd_0 = to_Rodrigo_Hamilton(phi0, psi0, theta0)
# значения параметров Родрига-Гамильтона
lambd_1 = to_Rodrigo_Hamilton(phi1, psi1, theta1)

# матрицы перехода (начальной и желаемой ориентаций)
Matrix = to_transition_matrix(lambd_0)
Matrix_0 = to_transition_matrix(lambd_1)


S = 100  # Число частиц
h = 0.05
N = 1  # Число интервалов
max_iter = 400
w, c1, c2 = 0.7, 1.4, 1.4
param_min = np.array([0.1, 1.0e6, 500.0, 0.5e9])
param_max = np.array([50.0, 7.0e6, 3500.0, 3.5e9])
t_max = 25.0


particles = np.zeros((S, 4, N))
velocities = np.random.uniform(-1, 1, (S, 4, N))

for i in range(4):
    particles[:, i, :] = np.random.uniform(param_min[i], param_max[i], (S, N))


pbest = particles.copy()
value_at_pbest = np.full(S, np.inf)
# глобальное лучшее пока просто нули, на первой итерации сразу перезапишется
gbest = particles[0].copy()
# значение целевой функции в gbest
value_at_gbest = np.inf

# PSO
# Запускаем главный цикл, используем tqdm для вывода шкалы процесса
for i in tqdm(range(max_iter), desc="Итерации", unit="итерация"):
    # Идем по каждой частице
    for j in range(S):
        # Моделируем
        current_function = function(
            particles[j],
            lambd_1,
            lambd_0,
            Omega_x0,
            Omega_y0,
            Omega_z0,
            Matrix_0,
            R,
            epsilon,
            delta,
            omega_0,
            omega_E,
            i_angle,
            t_max,
            N,
            h,
        )
        # Ищем индивидуально лучшую минимизацию Phi
        if current_function < value_at_pbest[j]:
            # Eсли текущее значение меньше индивидуально лучшего, то запоминаем значение и управление
            value_at_pbest[j] = current_function
            pbest[j] = particles[j].copy()
        # Ищем глобально лучшую минимизацию Phi
        if current_function < value_at_gbest:
            # Eсли текущее значение меньше глобально лучшего, то запоминаем значение и управление
            value_at_gbest = current_function
            gbest = particles[j].copy()

    # Обновление скоростей и позиций согласно методу
    r1 = np.random.rand(S, 4, N)
    r2 = np.random.rand(S, 4, N)
    velocities = (
        w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
    )
    particles += velocities
    # Ограничение управления
    for k in range(4):
        particles[:, k, :] = np.clip(particles[:, k, :], param_min[k], param_max[k])


print("Лучшее управление:", gbest)
