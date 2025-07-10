import numpy as np


# Вычисление значений параметров Родрига-Гамильтона
def to_Rodrigo_Hamilton(phi, psi, theta):
    angles = np.array([phi, psi, theta])
    cos_phi, cos_psi, cos_theta = np.cos(angles / 2)
    sin_phi, sin_psi, sin_theta = np.sin(angles / 2)

    return np.array(
        [
            cos_phi * cos_psi * cos_theta + sin_phi * sin_psi * sin_theta,
            sin_phi * cos_psi * cos_theta - cos_phi * sin_psi * sin_theta,
            cos_phi * cos_psi * sin_theta + sin_phi * sin_psi * cos_theta,
            cos_phi * sin_psi * cos_theta - sin_phi * cos_psi * sin_theta,
        ]
    )


# Преобразование параметров Родрига-Гамильтона в углы Эйлера
def from_Rodrigo_Hamilton(lambd_vec):
    matrix = to_transition_matrix(lambd_vec)

    return np.array(
        [
            np.arctan2(matrix[2, 1], matrix[2, 2]),
            np.arctan2(matrix[1, 0], matrix[0, 0]),
            -np.arcsin(matrix[2, 0]),
        ]
    )


# Вычисл. матрицы перехода от сист. главных центральных осей инерции к орбитальной сист. координат
def to_transition_matrix(lambd_vec):
    lambd_0, lambd_1, lambd_2, lambd_3 = lambd_vec

    return np.array(
        [
            [
                lambd_0 * lambd_0
                + lambd_1 * lambd_1
                - lambd_2 * lambd_2
                - lambd_3 * lambd_3,
                2 * (lambd_1 * lambd_2 - lambd_0 * lambd_3),
                2 * (lambd_1 * lambd_3 + lambd_0 * lambd_2),
            ],
            [
                2 * (lambd_1 * lambd_2 + lambd_0 * lambd_3),
                lambd_0 * lambd_0
                - lambd_1 * lambd_1
                + lambd_2 * lambd_2
                - lambd_3 * lambd_3,
                2 * (lambd_2 * lambd_3 - lambd_0 * lambd_1),
            ],
            [
                2 * (lambd_1 * lambd_3 - lambd_0 * lambd_2),
                2 * (lambd_2 * lambd_3 + lambd_0 * lambd_1),
                lambd_0 * lambd_0
                - lambd_1 * lambd_1
                - lambd_2 * lambd_2
                + lambd_3 * lambd_3,
            ],
        ]
    )
