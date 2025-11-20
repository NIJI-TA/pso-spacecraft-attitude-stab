import matplotlib.pyplot as plt
import numpy as np


def plot_angles(phi_values, psi_values, theta_values, u_vec):
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


def plot_domega(domega_1, domega_2, domega_3, u_vec):
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


def plot_lambd(lambd_vec, u_vec):
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


def plot_convergence(convergence, max_iter):
    plt.figure(figsize=(12, 5))
    plt.plot(range(max_iter), convergence, label="Phi", color="blue")
    plt.xlabel("Итерация")
    plt.ylabel("Значение целевой функции")
    plt.title("Изменение значения целевой функции")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kl(kl, n, u_max):
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


def plot_km(km, n, u_max):
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


def plot_hl(hl, n, u_max):
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


def plot_hm(hm, n, u_max):
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


# Для двух методов сразу


def plot_angles_two_methods(
    phi_values,
    psi_values,
    theta_values,
    phi_values2,
    psi_values2,
    theta_values2,
    u_vec1,
    u_vec2,
):
    # phi, psi, theta
    plt.figure(figsize=(12, 5))
    plt.plot(u_vec1, phi_values, label="φ", color="blue", linewidth=1.5)
    plt.plot(u_vec1, psi_values, label="ψ", color="green", linewidth=1.5)
    plt.plot(u_vec1, theta_values, label="θ", color="red", linewidth=1.5)
    plt.plot(u_vec2, phi_values2, color="blue", linestyle="--", linewidth=1.5)
    plt.plot(u_vec2, psi_values2, color="green", linestyle="--", linewidth=1.5)
    plt.plot(u_vec2, theta_values2, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("u", fontsize=16)
    plt.ylabel("рад", fontsize=16)
    plt.grid(
        True,
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16, edgecolor="grey", loc="lower right", fancybox=False)
    plt.tight_layout()
    plt.show()


def plot_kl_two_methods(kl1, kl2, n, u_max, param_min, param_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(
        x,
        np.append(kl1, kl1[-1]),
        where="post",
        label="k_L (PSO)",
        color="red",
        linestyle="-",
    )
    plt.step(
        x,
        np.append(kl2, kl2[-1]),
        where="post",
        label="k_L (SA)",
        color="blue",
        linestyle="--",
    )
    plt.ylim(1.05 * param_min[0], 1.05 * param_max[0])
    plt.xlabel("u", fontsize=16)
    plt.ylabel("k_L", fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16, edgecolor="grey", loc="lower right", fancybox=False)
    plt.tight_layout()
    plt.show()


def plot_km_two_methods(km1, km2, n, u_max, param_min, param_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(
        x,
        np.append(km1, km1[-1]),
        where="post",
        label="k_M (PSO)",
        color="red",
        linestyle="-",
    )
    plt.step(
        x,
        np.append(km2, km2[-1]),
        where="post",
        label="k_M (SA)",
        color="blue",
        linestyle="--",
    )
    plt.ylim(1.05 * param_min[1], 1.05 * param_max[1])
    plt.xlabel("u", fontsize=16)
    plt.ylabel("k_M", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16, edgecolor="grey", loc="lower right", fancybox=False)
    plt.tight_layout()
    plt.show()


def plot_hl_two_methods(hl1, hl2, n, u_max, param_min, param_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(
        x,
        np.append(hl1, hl1[-1]),
        where="post",
        label="h_L (PSO)",
        color="red",
        linestyle="-",
    )
    plt.step(
        x,
        np.append(hl2, hl2[-1]),
        where="post",
        label="h_L (SA)",
        color="blue",
        linestyle="--",
    )
    plt.ylim(1.05 * param_min[2], 1.05 * param_max[2])
    plt.xlabel("u", fontsize=16)
    plt.ylabel("h_L", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16, edgecolor="grey", loc="lower right", fancybox=False)
    plt.tight_layout()
    plt.show()


def plot_hm_two_methods(hm1, hm2, n, u_max, param_min, param_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(
        x,
        np.append(hm1, hm1[-1]),
        where="post",
        label="h_M (PSO)",
        color="red",
        linestyle="-",
    )
    plt.step(
        x,
        np.append(hm2, hm2[-1]),
        where="post",
        label="h_M (SA)",
        color="blue",
        linestyle="--",
    )
    plt.ylim(1.05 * param_min[3], 1.05 * param_max[3])
    plt.xlabel("u", fontsize=16)
    plt.ylabel("h_M", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16, edgecolor="grey", loc="lower right", fancybox=False)
    plt.tight_layout()
    plt.show()
