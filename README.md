# Décomposition en valeurs singulières (SVD)

Calcul et vérification de la SVD en Python (NumPy), approximation de rang faible et lien avec
la décomposition orthogonale aux valeurs propres (POD) utilisée en réduction de modèles.

## La SVD en bref

Toute matrice réelle $A$ de taille $m \times n$ se décompose en

$$A = U \Sigma V^T = \sum_{i=1}^{r} \sigma_i \, u_i v_i^T$$

- $U$ et $V$ sont orthogonales : leurs colonnes $u_i$, $v_i$ sont des bases orthonormées ;
- $\Sigma$ est diagonale, avec les valeurs singulières $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ ;
- $A^T A = V \Sigma^2 V^T$ : les $\sigma_i^2$ sont les valeurs propres de $A^T A$.

**Théorème d'Eckart–Young.** La meilleure approximation de rang $k$ de $A$ est la somme des
$k$ premiers termes, et son erreur en norme 2 vaut exactement $\sigma_{k+1}$.
La part d'énergie captée par les $k$ premiers modes est $\sum_{i \le k} \sigma_i^2 / \sum_i \sigma_i^2$.

**Lien avec la POD.** Si chaque colonne de $A$ est un instantané d'un champ physique
(vitesse, champ magnétique...), les $u_i$ sont les modes POD et $\sigma_i^2$ leur énergie :
quelques modes suffisent souvent à représenter le champ, c'est la base des modèles réduits.

## Contenu

| Fichier | Rôle |
|---|---|
| `svd.py` | SVD réduite, reconstruction de rang $k$, énergie, SVD via les valeurs propres de $A^T A$ |
| `test_svd.py` | 6 tests : exemple calculé à la main, orthogonalité, Eckart–Young, rang exact, énergie |

## Exécution

```bash
pip install numpy
python svd.py              # démonstration
python -m unittest -v      # tests
```

## Résultats

Exemple à la main : $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$ donne
$\sigma_1 = \sqrt{3}$ et $\sigma_2 = 1$, reconstruction exacte à $10^{-16}$ près.

Champ 200 x 120 composé d'une onde progressive (rang 2), d'un mode stationnaire (rang 1)
et d'un bruit de 1 % :

| $k$ | Erreur en norme 2 | $\sigma_{k+1}$ | Énergie captée |
|---|---|---|---|
| 1 | 77.32 | 77.32 | 44.6 % |
| 2 | 38.79 | 38.79 | 88.8 % |
| 4 | 0.236 | 0.236 | 99.98 % |
| 10 | 0.220 | 0.220 | 99.99 % |

L'erreur est égale à $\sigma_{k+1}$ (Eckart–Young). Trois modes captent 99,9 % de l'énergie :
c'est exactement le rang du champ sans bruit ; au-delà, les modes ne représentent plus que le bruit.

Application à un vrai champ d'écoulement (modes POD et métamodèle paramétrique validé) :
[pod-rom-lbm](https://github.com/H-HABCHANE/pod-rom-lbm).
