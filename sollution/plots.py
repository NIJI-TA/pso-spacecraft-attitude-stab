import matplotlib.pyplot as plt
import numpy as np


def get_angles(phi_values, psi_values, theta_values, u_vec):
    # phi, psi, theta
    plt.figure(figsize=(12, 5))
    plt.plot(u_vec, phi_values, label="φ (phi)", color="blue")
    plt.plot(u_vec, psi_values, label="ψ (psi)", color="green")
    plt.plot(u_vec, theta_values, label="θ (theta)", color="red")
    plt.xlabel("u")
    plt.ylabel("рад.")
    plt.title("Изменение углов φ, ψ, θ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_domega(domega_1, domega_2, domega_3, u_vec):
    plt.figure(figsize=(12, 5))
    plt.plot(u_vec, domega_1, label="dω1", color="blue")
    plt.plot(u_vec, domega_2, label="dω2", color="green")
    plt.plot(u_vec, domega_3, label="dω3", color="red")
    plt.xlabel("u")
    plt.ylabel("Проекции domega")
    plt.title("Изменение проекций domega")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_lambd(lambd_vec, u_vec):
    lambd0_vec = lambd_vec[:, 0]
    lambd1_vec = lambd_vec[:, 1]
    lambd2_vec = lambd_vec[:, 2]
    lambd3_vec = lambd_vec[:, 3]

    plt.figure(figsize=(12, 5))
    plt.plot(u_vec, lambd0_vec, label="lambd0", color="blue")
    plt.plot(u_vec, lambd1_vec, label="lambd1", color="green")
    plt.plot(u_vec, lambd2_vec, label="lambd2", color="red")
    plt.plot(u_vec, lambd3_vec, label="lambd3", color="black")
    plt.xlabel("u")
    plt.ylabel("lambd_vec")
    plt.title("Изменение параметров Родрига-Гамильтона во времени")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_convergence(convergence, max_iter):
    plt.figure(figsize=(12, 5))
    plt.plot(range(max_iter), convergence, label="Phi", color="blue")
    plt.xlabel("Итерация")
    plt.ylabel("Значение целевой функции")
    plt.title("Изменение значения целевой функции")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_kl(kl, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(kl, kl[-1]), where="post", label="kl", color="blue")
    plt.xlabel("u")
    plt.ylabel("kl")
    plt.title("Изменение параметра kl")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_km(km, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(km, km[-1]), where="post", label="km", color="black")
    plt.xlabel("u")
    plt.ylabel("km")
    plt.title("Изменение параметра km")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_hl(hl, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(hl, hl[-1]), where="post", label="hl", color="green")
    plt.xlabel("u")
    plt.ylabel("hl")
    plt.title("Изменение параметра hl")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_hm(hm, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(hm, hm[-1]), where="post", label="hm", color="red")
    plt.xlabel("u")
    plt.ylabel("hm")
    plt.title("Изменение параметра hm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
