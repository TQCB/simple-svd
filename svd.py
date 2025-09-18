from typing import Tuple

import numpy as np
import plotly.express as px

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
        total_variance = self.pc_variance.sum()
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
        strict: bool=False
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
        raise RuntimeError("Equality assertion against initial matrix failed.")

    return U, S, V

def scatter_2d_with_loadings(
        data_matrix,
        right_singular_vectors,
        singular_values,
        scale_factor=0.1,
        width=600,
        height=400,
        margin=35
        ) -> None:
    
    # Get PC loadings from singular values
    loadings = singular_values.sum(axis=0)[:2] * scale_factor
    
    fig = px.scatter(
        x=data_matrix[:, 0],
        y=data_matrix[:, 1],
        title='2D PCA Visualization',
        labels={'x': 'PC1', 'y': 'PC2'},
        width=width, height=height
    )

    # Add the first principal component vector as an arrow
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=right_singular_vectors[0, 0] * loadings[0], y1=right_singular_vectors[1, 0] * loadings[0],
        line=dict(color='red', width=3),
        name='PC1'
    )

    # Add the second principal component vector as an arrow
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=right_singular_vectors[0, 1] * loadings[1], y1=right_singular_vectors[1, 1] * loadings[1],
        line=dict(color='green', width=3),
        name='PC2'
    )

    fig.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin))
    fig.show()

def generate_data(
        n_samples,
        minimalist=False,
        seed=42,
    ) -> np.ndarray:
    np.random.seed(42)

    age = np.random.normal(loc=35, scale=10, size=(n_samples, 1)).astype(int)
    experience = np.random.normal(loc=10, scale=5, size=(n_samples, 1)).astype(int)
    satisfaction = np.random.normal(loc=3, scale=1, size=(n_samples, 1)).astype(int)

    if not minimalist:
        salary = 2000 * age + 5000 * experience + np.random.normal(loc=50000, scale=20000, size=(n_samples, 1))
    else:
        salary = 2000 * age + np.random.normal(loc=50000, scale=20000, size=(n_samples, 1))

    age = np.clip(age, 20, 65)
    experience = np.clip(experience, 0, age - 18)
    satisfaction = np.clip(satisfaction, 0, 5)
    salary = np.clip(salary, 30000, 200000)

    if not minimalist:
        A = np.hstack((age, salary.astype(int)))
    else:
        A = np.hstack((age, salary.astype(int), satisfaction, experience))

    return A

if __name__ == '__main__':
    import plotly.express as px

    df = px.data.iris()

    df['species'] = df['species'].map({
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2,
        })
    
    data = df.to_numpy()

    pca = PCA(n_components=3)
    pca.fit(data)

    print(pca.pc_variance_ratio)
    print(pca.transform(data))