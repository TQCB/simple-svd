# SVD Numeric Guide
We aim to find the singular value decomposition of a rectangular matrix $A$, which is a factorization of the form $A = U \Sigma V^T$. For application to PCA specifically, see section entitled **PCA**.

## $\Sigma$ and V
First we calculate our right singular vectors $V$, along with the eigen-values, of $A^TA$. The eigenvalues of $A^TA$ are equal to the square of the singular values of $A$ (by definition, $\sigma_i = \sqrt{\lambda_i}$).

## U
If we calculate $U$ directly from $AA^T$, in the same manner as we did with $V$ from $A^TA$, we may end up with eigenvectors that have flipped signs. If you want to see exactly how this can happen, and why it is problematic for SVD, then read the next section, if not then skip directly to "**Calculation of U from $\Sigma, V$**".

### Sign Inconsistency
We cannot calculate U in the same manner as V, due to sign inconsistency with eigen-vectors. An eigen-vector is defined as any vector $x$ which satisfies the following equation:
$$Ax = \lambda x$$
It then follows that:
$$A(-x) = -(Ax) = -(\lambda x) = \lambda (-x)$$

Our initial form $A = U \Sigma V^T$ can be adjusted into:
$$AV = U \Sigma V^T V$$
Since V is orthonormal *(see Annex 1)*:
$$AV = U \Sigma$$
If we only look at a single column instead of the entire transformation:
$$A v_i = s_i u_i$$

From this form, let us try to understand why our decomposition does not work if the signs of U and V are not dependent.

Let us imagine a case where, during our left singular vector calculation for $U$, we got a $-u_i$ vector instead of $u_i$ (see earlier for why these are both valid). We are going to test if the following relation holds:
$$A v_i \stackrel{?}{=} s_i (-u_i)$$

Our ground truth allows us to test:
$$s_i u_i \stackrel{?}{=} -s_i u_i$$

This equality only holds true in two cases:
- If $s_i = 0$
- If $u_i$ is the zero vector

For us to correctly calculate $U$ so that our equation holds true, we calculate it dependently on our $\Sigma$ and $V$ matrices.

### Calculation of U from $\Sigma, V$
Using simple algebraic reformulation, and the fact $V$ is orthonormal *(see Annex 1 for definition, proof and relation to the following equations)*, we can rearrange to find:
$$A = U \Sigma V^T$$
$$A V = U \Sigma V^T V$$
$$A V = U \Sigma I$$
$$A V = U \Sigma$$

To now remove the $\Sigma$ term, we must use the inverse of $\Sigma$. Since $\Sigma \in \R^{n, m}$ and is thus rectangular, we must use the pseudo-inverse of $\Sigma$ *(see Annex 2)*, noted $\Sigma^{+}$, which will allow us to remove the term:

$$A V \Sigma^{+}= U \Sigma \Sigma^{+}$$
$$A V \Sigma^{+}= U$$

## PCA


# Annex
## 1: Orthonormality of left and right singular vectors
#### **Orthonormality definition**
An orthonormal matrix is a real square matrix whose columns are rows are orthonormal vectors. This can also be expressed as:
$$M^TM = MM^T = I$$

We can show a matrix is orthonormal by showing that their vectors form an orthonormal set:
$$v^T_i v_j = \delta_{ij}$$
Where $\delta_{ij}$ is the Kronecker delta: $1$ if $i = j$, $0$ if $i \neq j$

In the next two sections, I'll show why this is the case.

#### **1.** Normality
By definition, eigenvectors are scaled to be unit vectors, so their Euclidean norm is $1$.
$$||v_i|| = \sqrt{v^T_i v_i} = 1$$
$$v^T_i v_i = 1$$

#### **2.** Orthogonality
Consider two distinct eigenvalues, $\lambda_i \ and \ \lambda_j$, with respective eigenvectors $v_i$ and $v_j$ *(see Annex 3 for cases of repeated eigenvalues)*:
$$A^T A v_i = \lambda_i v_i \ and \ A^T A v_j = \lambda_j v_j$$
We can take the transpose of the first statement and post-multiply by $v_j$:
$$(v^T_i(A^T A))^T v_j = (\lambda_i v_i)^T v_j$$
$${v}_i^T A^T A {v}_j = \lambda_i {v}_i^T {v}_j$$
Substitute $A^T A v_j = \lambda_j v_j$ into the left hand side:
$$v^T_i(\lambda_j v_j) = \lambda_i v^T_i v_j$$
$$\lambda_j(v^T_i v_j) = \lambda_i(v^T_i v_j)$$
Rearranging terms we get:
$$(\lambda_j - \lambda_i)(v^T_i v_j) = 0$$
Since we assumed $\lambda_i \neq \lambda_j$, the difference which non-zero, which means the inner-product has to be zero:
$$v^T_i v_j = 0$$
Yippee, our eigenvectors are orthonormal.

## 2: Pseudo-inverse of rectangular matrices
Since our singular values are a rectangular matrix, they are not invertible. However, by utilizing the Moore-Penrose pseudoinverse, we can remove our $\Sigma$ term to find $U$!

Specifically, in the special case of diagonal matrices (such as $\Sigma$), the pseudo-inverse is ["... obtained by taking the reciprocal of nonzero diagonal elements."](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse), and then transposing the resulting matrix.

For example, a matrix $A$:  
$\begin{bmatrix}
\sigma_1 & 0 \\
0 & \sigma_2 \\
0 & 0
\end{bmatrix}$

will have the pseudo-inverse $A^+$:  
$\begin{bmatrix}
1/ \sigma_1 & 0 & 0 \\
0 & 1/ \sigma_2 & 0
\end{bmatrix}$

## 3: Repeated eigenvectors
There are cases where eigenvalues are repeated for symmetric matrices. In this case it is still possible to find sets of orthogonal eigenvectors for our corresponding eigenspace. See the [Gram-Schmidt algorithm](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) for how to ensure orthogonality in these cases.

While I won't dive into how this works, we can note that the algorithm takes a finite, linearly dependent set of vectors (repeated) $S$ and generates an orthogonal set that spans the same $k$-dimensional subspace of $\R^n$ as $S$.