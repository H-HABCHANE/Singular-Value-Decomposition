"""Décomposition en valeurs singulières (SVD) : calcul, vérifications et approximation de rang faible.

Toute matrice A (m x n) s'écrit A = U S V^T avec
  - U (m x m) et V (n x n) orthogonales,
  - S (m x n) diagonale, valeurs singulières s1 >= s2 >= ... >= 0.

Exécuter : python svd.py
"""

import numpy as np


def svd(A):
    """SVD réduite : renvoie U (m x r), s (r,), Vt (r x n) avec r = min(m, n)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt


def reconstruct(U, s, Vt, k=None):
    """Somme des k premiers termes de rang 1 : s1 u1 v1^T + ... + sk uk vk^T."""
    if k is None:
        k = len(s)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def energy(s):
    """Part d'énergie cumulée captée par les k premiers modes : sum(s_i^2, i<=k) / sum(s_i^2)."""
    e = s**2
    return np.cumsum(e) / e.sum()


def rank_for_energy(s, target):
    """Plus petit k tel que les k premiers modes captent au moins `target` de l'énergie."""
    return int(np.searchsorted(energy(s), target) + 1)


def svd_from_eigen(A):
    """SVD obtenue à partir des valeurs propres de A^T A (pour comprendre le lien SVD / valeurs propres).

    A^T A = V S^2 V^T : les vecteurs propres de A^T A sont les vecteurs singuliers à droite,
    et s_i = sqrt(lambda_i). Ensuite u_i = A v_i / s_i.
    Moins précise que np.linalg.svd (on élève les valeurs au carré) : à but pédagogique.
    """
    lam, V = np.linalg.eigh(A.T @ A)
    order = np.argsort(lam)[::-1]  # valeurs décroissantes
    lam, V = lam[order], V[:, order]
    r = min(A.shape)
    s = np.sqrt(np.clip(lam[:r], 0.0, None))
    U = A @ V[:, :r] / s
    return U, s, V[:, :r].T


def main():
    np.set_printoptions(precision=4, suppress=True)

    # 1. Petit exemple calculable à la main
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    U, s, Vt = svd(A)
    print("A =\n", A)
    print("U =\n", U)
    print("valeurs singulières =", s, " (attendu : sqrt(3) = 1.7321 et 1)")
    print("V^T =\n", Vt)
    print("erreur ||A - U S V^T|| =", np.linalg.norm(A - reconstruct(U, s, Vt)))
    print("erreur ||U^T U - I||   =", np.linalg.norm(U.T @ U - np.eye(2)))

    # 2. Approximation de rang faible (Eckart–Young) :
    #    la meilleure approximation de rang k a une erreur (norme 2) égale à s_{k+1}
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 200)
    t = np.linspace(0, 1, 120)
    X, T = np.meshgrid(x, t, indexing="ij")
    field = (np.sin(2 * np.pi * (X - T)) + 0.5 * np.sin(6 * np.pi * X) * np.cos(4 * np.pi * T)
             + 0.01 * rng.standard_normal(X.shape))
    U, s, Vt = svd(field)
    print("\nChamp 200 x 120 (onde + mode stationnaire + bruit)")
    print(" k   erreur norme 2   s_(k+1)    énergie captée")
    for k in [1, 2, 4, 6, 10]:
        err = np.linalg.norm(field - reconstruct(U, s, Vt, k), 2)
        print(f"{k:2d}   {err:12.5f}   {s[k]:9.5f}   {energy(s)[k - 1]:10.6f}")
    print("modes pour 99,9 % de l'énergie :", rank_for_energy(s, 0.999))


if __name__ == "__main__":
    main()
