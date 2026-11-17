import numpy as np
from dynamics import simulate_dinamics

from config import OMEGA_0


# Считаем целевую функцию Phi
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
    OmegaX = OMEGA_0 * (
        omega_vec[:, 0]
        - 2 * (lambd_vec[:, 1] * lambd_vec[:, 2] + lambd_vec[:, 0] * lambd_vec[:, 3])
    )
    OmegaY = OMEGA_0 * (
        omega_vec[:, 1]
        - (
            lambd_vec[:, 0] * lambd_vec[:, 0]
            - lambd_vec[:, 1] * lambd_vec[:, 1]
            + lambd_vec[:, 2] * lambd_vec[:, 2]
            - lambd_vec[:, 3] * lambd_vec[:, 3]
        )
    )
    OmegaZ = OMEGA_0 * (
        omega_vec[:, 2]
        - 2 * (lambd_vec[:, 2] * lambd_vec[:, 3] - lambd_vec[:, 0] * lambd_vec[:, 1])
    )

    total_error = (
        np.sum((lambd_vec - lambd_vec_1) ** 2, axis=1)
        + OmegaX * OmegaX
        + OmegaY * OmegaY
        + OmegaZ * OmegaZ
    )

    return (h / 2) * np.sum(total_error[:-1] + total_error[1:])
