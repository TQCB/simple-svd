import numpy as np
import plotly.express as px

def scatter_2d_with_loadings(
        data_matrix,
        right_singular_vectors,
        singular_values,
        standardize=True,
        scale_factor=0.1,
        width=600,
        height=400,
        margin=35
        ) -> None:
    
    if standardize:
        data_matrix = (data_matrix - data_matrix.mean(axis=0)) / data_matrix.std(axis=0)
    
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
    np.random.seed(seed)

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