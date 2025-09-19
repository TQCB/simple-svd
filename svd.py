from typing import Tuple

import numpy as np
import plotly.express as px

from utils import scatter_2d_with_loadings

class PCA:
    def __init__(
            self,
            n_components,
    ):
        self.n_components = n_components

    def truncate(
            self,
    ):
        U_k = self.U[:, :self.n_components]
        S_k = self.S[:self.n_components,:self.n_components]
        V_k = self.V[:, :self.n_components]

        return U_k, S_k, V_k

    def fit(
        self,
        X
    ):
        self.U, self.S, self.V = svd(X, standardize=True)
        self.U_k, self.S_k, self.V_k = self.truncate()

        self.approx_X = self.U_k @ self.S_k @ self.V_k.T

        self.pc_variance = self.S_k.sum(axis=0)
        total_variance = self.S.sum()
        self.pc_variance_ratio = self.pc_variance / total_variance

    def transform(
        self,
        X,
        standardize=True,
    ):
        if standardize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)

        return X @ self.V_k

def svd(
        mat: np.ndarray,
        standardize: bool=False,
        strict: bool=True
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates SVD of a matrix.

    Args:
        mat (np.ndarray): matrix to decompose
        standardize (bool): whether to standardize the input matrix
        strict (bool): whether to fail if output is not exact

    Returns:
        U (np.ndarray): left singular vectors
        S (np.ndarray): singular values
        V (np.ndarray): right singular vectors
    """
    if standardize:
        mat = (mat - mat.mean(axis=0)) / mat.std(axis=0)

    # V or right singular vectors and singular values
    ATA = mat.T @ mat
    eig_ATA, V_unsorted = np.linalg.eigh(ATA)

    # sort V and S
    sorted_indices = np.argsort(eig_ATA)[::-1]
    V = V_unsorted[:, sorted_indices]
    S_vals = np.sqrt(eig_ATA[sorted_indices])

    # get sigma matrix
    m, n = mat.shape
    S = np.zeros((m, n))
    min_dim = min(m, n)
    S[:min_dim, :min_dim] = np.diag(S_vals[:min_dim])

    # To ensure sign consistency, calculate U from A, V and S
    s_inv_diag = np.zeros_like(S_vals)
    non_zero_s_vals = S_vals > 1e-12
    s_inv_diag[non_zero_s_vals] = 1 / S_vals[non_zero_s_vals]

    S_inv = np.zeros((n, m))
    S_inv[:min_dim, :min_dim] = np.diag(s_inv_diag[:min_dim])

    U = mat @ V @ S_inv

    if strict and not np.allclose(U @ S @ V.T, mat):
        raise RuntimeError("Equality assertion against initial matrix failed. Check for collinearity in input data.")

    return U, S, V

if __name__ == '__main__':
    # Get some data
    import plotly.express as px
    data = px.data.iris().drop(columns=['species', 'species_id']).to_numpy()

    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(data)

    # Get variance ratios
    print("Explained variance ratios:")
    [print(f"PC{i}: {pc_var*100:.0f}%", end=' ') for i, pc_var in enumerate(pca.pc_variance_ratio)]
    print()

    # Get approximated X and transformed X
    approximated_data = pca.approx_X
    transformed_data = pca.transform(data)

    # Visualize result
    scatter_2d_with_loadings(transformed_data, pca.V_k, pca.S)