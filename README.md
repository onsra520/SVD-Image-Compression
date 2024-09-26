# SVD Image Compression - Explained

**Singular Value Decomposition** (**SVD**) là một khái niệm cơ bản trong đại số tuyến tính và đặc biệt quan trọng trong lĩnh vực học máy cho các nhiệm vụ như dimensionality reduction, data compression, and noise reduction.
 

 ## Singular Value Decomposition

Singular Value Decomposition - SVD là một kỹ thuật phân tích ma trận, phân tách bất kỳ ma trận nào thành ba ma trận riêng biệt.

$$
    **\mathbf{A} = \mathbf{U\Sigma V^\mathsf{T}}**
$$

Có thể áp dụng decomposition cho bất kỳ ma trận $m \times n$ ma trận $\mathbf A$, kết quả là ba ma trận:
- $\mathbf U$: Đây là ma trận trực giao(orthogonal matrix) kích thước $m \times m$. Các cột của ma trận này là left-singular vectors của $\mathbf A$.
- $\mathbf \Sigma$: Đây là ma trận đường chéo (diagonal matrix) kích thước $m \times n$. Các giá trị trên đường chéo được ký hiệu là $\sigma_i$ và được gọi là *singular values* của $\mathbf A$.
- $\mathbf V^\mathsf{T}$: Đây là ma trận trực giao chuyển vị (transposed orthogonal matrix) kích thước $n \times n$. Các cột của ma trận chưa chuyển vị (non-transposed matrix) $\mathbf V$, là right-singular vectors của $\mathbf A$. 

Có thể tính decomposition bằng cách phân tích giá trị riêng (Eigenvalues) và vector riêng (Eigenvectors) của $\mathbf{A^\mathsf{T}A}$ và $\mathbf{AA^\mathsf{T}}$, trong đó các giá trị riêng (Eigenvalues) của cả hai ma trận này đều bằng bình phương của singular values. Sau đó, chúng ta sắp xếp các giá trị kỳ dị này theo thứ tự giảm dần và đưa chúng vào đường chéo của ma trận $\mathbf \Sigma$.

Dựa trên thứ tự của corresponding singular values, ta xây dựng các cột của ma trận $\mathbf U$ từ các vector riêng (Eigenvectors) của ma trận $\mathbf{AA^\mathsf{T}}$, và các hàng của ma trận $\mathbf V^\mathsf{T}$ (các cột của $\mathbf V$) từ các vector riêng (Eigenvectors) của $\mathbf{A^\mathsf{T}A}$.

Với SVD, chúng ta có thể diễn giải lại phép biến đổi tuyến tính này như ba phép biến đổi riêng biệt (được áp dụng từ phải sang trái):

1. **Phép quay hệ trục tọa độ với ma trận** $\mathbf{V}^\mathsf{T}$:  
   Vì $\mathbf{V}^\mathsf{T}$ là ma trận kích thước $n \times n$, phép này tương ứng với một phép quay trong không gian của chiều đầu vào.
   
3. **Phép co giãn bởi singular values** $\sigma_i$ cho mọi $i$:  
   Số lượng các giá trị này không vượt quá $\text{min}(m, n)$. Việc nhân với ma trận này cũng sẽ mở rộng các vector của ma trận mới bằng các giá trị 0.
   
5. **Phép quay hệ trục tọa độ với ma trận** $\mathbf{U}$:  
   Vì $\mathbf{U}$ là ma trận kích thước $m \times m$, phép này tương ứng với một phép quay trong không gian mới $\mathbb{R}^m$.

---

Ba phép biến đổi này giúp ta hiểu rõ hơn về cách mà SVD làm thay đổi dữ liệu thông qua các bước:  
- Quay,
- Co giãn,  
- Và quay trong không gian.

# Singular Value Decomposition (SVD) and Its Applications

## Image Compression

SVD có thể rất hữu ích trong việc tìm kiếm các mối quan hệ quan trọng trong dữ liệu. Điều này có nhiều ứng dụng trong học máy, tài chính và khoa học dữ liệu. Một trong những ứng dụng của SVD là trong **image compression**. Mặc dù không có định dạng hình ảnh lớn nào sử dụng SVD do độ phức tạp tính toán của nó, SVD vẫn có thể được áp dụng trong các trường hợp khác như một cách để nén dữ liệu.

```python
import cv2
import matplotlib.pyplot as plt

Image = cv2.imread('Meme.png')
Gray_Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

plt.imshow(image, cmap='gray')
plt.title('Cat Image')
plt.show()
```

![png](Markdown/Output_1.png)

## Giải Thích Về Chuyển Đổi Hình Ảnh Sang Ảnh Xám

Khi bạn đọc một hình ảnh và chuyển nó thành ảnh xám, quá trình chuyển đổi không đơn giản là thay thế một ma trận 3x3 (mà bạn thấy khi đọc hình ảnh màu) bằng một ma trận 3x3 khác. Thay vào đó, đó là một quá trình xử lý hình ảnh bao gồm nhiều bước.

### 1. Đọc Hình Ảnh
Khi bạn sử dụng `cv2.imread()` để đọc hình ảnh, OpenCV trả về một ma trận ba chiều (3D) với kích thước \( H \times W \times 3 \), trong đó:
- **(\ H \)**: Chiều cao của hình ảnh.
- **\( W \)**: Chiều rộng của hình ảnh.
- **3**: Ba kênh màu (Blue, Green, Red).

### 2. Chuyển Đổi Sang Ảnh Xám
Khi bạn chuyển đổi hình ảnh màu sang ảnh xám bằng `cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)`, OpenCV sử dụng công thức sau để tính giá trị độ xám cho mỗi pixel:

\[
Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B
\]

Trong đó:
- \( R, G, B \): Giá trị màu của từng pixel trong ba kênh màu.
- \( Y \): Giá trị độ xám tương ứng.

### 3. Ma Trận Ảnh Xám
Kết quả của quá trình chuyển đổi này là một ma trận hai chiều (2D) với kích thước \( H \times W \). Mỗi giá trị trong ma trận này đại diện cho độ xám của pixel tại vị trí tương ứng.

---

### Tóm lại
- **Ma trận 3x3** mà bạn thấy ở hình ảnh màu chứa thông tin cho ba kênh màu.
- **Ma trận 2D** (ảnh xám) chỉ chứa giá trị độ xám cho mỗi pixel, không còn thông tin màu sắc riêng biệt.

Do đó, ảnh xám và ma trận ban đầu không giống nhau về kích thước và nội dung. Ảnh xám đơn giản hóa hình ảnh bằng cách giảm số lượng kênh màu từ ba xuống một, trong khi vẫn giữ lại thông tin ánh sáng tổng thể.






