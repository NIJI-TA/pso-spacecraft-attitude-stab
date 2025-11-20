import numpy as np
from tqdm import tqdm

from Phi_function import function_phi


# param_min, param_max = np.array([a, b, c, d]), np.array([e, f, g, h])
def run_pso(
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
):
    # Коэффициенты метода PSO
    # w - инерционный вес, c1 и c2 - социальный и когнитивный коэффициенты
    # w_low, c1_low, c2_low - минимальные значения; w_up, c1_up, c2_up - максимальные значения
    w_low, c1_low, c2_low = 0.4, 0.0, 0.0
    w_up, c1_up, c2_up = 0.9, 2.05, 2.05

    particles = np.zeros((s, 4, n))
    velocities = np.zeros((s, 4, n))

    for i in range(4):
        particles[:, i, :] = np.random.uniform(param_min[i], param_max[i], (s, n))
        velocities[:, i, :] = np.random.uniform(
            -0.15 * (param_max[i] - param_min[i]),
            0.15 * (param_max[i] - param_min[i]),
            (s, n),
        )

    pbest = particles.copy()
    value_at_pbest = np.full(s, np.inf)

    gbest = particles[
        0
    ].copy()  # глобальное лучшее на первой итерации сразу перезапишется
    value_at_gbest = np.inf  # значение целевой функции в gbest
    # Список для хранения значений целевой функции на каждой итерации для анализа сходимости

    conv = []  # Для анализа сходимости

    # PSO
    for i in tqdm(range(max_iter), desc="Итерации PSO", unit="итерация"):
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
        conv.append(value_at_gbest)
        # Обновление скоростей и позиций согласно методу
        w = (w_low - w_up) * (i / max_iter) + w_up
        c1 = (c1_low - c1_up) * (i / max_iter) + c1_up
        c2 = (c2_up - c2_low) * (i / max_iter) + c2_low
        r1 = np.random.uniform(0, c1_up, (s, 4, n))
        r2 = np.random.uniform(0, c2_up, (s, 4, n))
        velocities = (
            w * velocities
            + c1 * r1 * (gbest - particles)
            + c2 * r2 * (pbest - particles)
        )
        particles += velocities
        # Ограничение управления
        for k in range(4):
            particles[:, k, :] = np.clip(particles[:, k, :], param_min[k], param_max[k])

            mask_min = velocities[:, k, :] < 0.15 * param_min[k]
            mask_max = velocities[:, k, :] > 0.15 * param_max[k]

            v_max = 0.15 * (param_max[k] - param_min[k])

            velocities[:, k, :][mask_min] = np.random.uniform(
                -v_max, v_max, mask_min.sum()
            )
            velocities[:, k, :][mask_max] = np.random.uniform(
                -v_max, v_max, mask_max.sum()
            )

    return gbest, value_at_gbest, conv
