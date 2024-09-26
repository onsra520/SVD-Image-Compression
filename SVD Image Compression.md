# SVD Image Compression - Explained

**Singular Value Decomposition** (**SVD**) là một khái niệm cơ bản trong đại số tuyến tính và đặc biệt quan trọng trong lĩnh vực học máy cho các nhiệm vụ như dimensionality reduction, data compression, and noise reduction.
 

 ## Singular Value Decomposition

Singular Value Decomposition - SVD là một kỹ thuật phân tích ma trận, phân tách bất kỳ ma trận nào thành ba ma trận riêng biệt.

$$
    \mathbf{A} = \mathbf{U\Sigma V^\mathsf{T}}
$$

Có thể áp dụng decomposition cho bất kỳ ma trận $m \times n$ ma trận $\mathbf A$, kết quả là ba ma trận:
- $\mathbf U$: Đây là ma trận trực giao(orthogonal matrix) kích thước $m \times m$. Các cột của ma trận này là left-singular vectors of $\mathbf A$.
- $\mathbf \Sigma$: This is an $m \times n$ diagonal matrix. The diagonal values are denoted $\sigma_i$ and are called the *singular values* of $\mathbf A$.
- $\mathbf V^\mathsf{T}$: This is an $n \times n$ transposed orthogonal matrix. The columns of the non-transposed matrix, $\mathbf V$, are the right-singular vectors of $\mathbf A$. 