# SVD Image Compression - Explained

**Singular Value Decomposition** (**SVD**) là một khái niệm cơ bản trong đại số tuyến tính và đặc biệt quan trọng trong lĩnh vực học máy cho các nhiệm vụ như dimensionality reduction, data compression, and noise reduction.
 

 ## Singular Value Decomposition

Singular Value Decomposition - SVD là một kỹ thuật phân tích ma trận, phân tách bất kỳ ma trận nào thành ba ma trận riêng biệt.

$$
    \mathbf{A} = \mathbf{U\Sigma V^\mathsf{T}}
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
- Quay.
- Co giãn 
- Quay trong không gian.

# Singular Value Decomposition (SVD) and Its Applications

## Import các thư viện và tạo các Folder cần thiết cho Image Compression.

```python
import os
import numpy as np
import cv2
from PIL import Image
```
    os.makedirs('Images_Folder', exist_ok=True)
    os.makedirs('Result_Folder', exist_ok=True)

## Class và Define liên quan đến Image Compression

```python
class SVD_Image_Compression_Proccessing:
    def __init__(self, Image_Name, Matrix_Approximation):
        self.Original_Matrix = self.Convert_Image_To_Matrix(Image_Name)
        self.K = Matrix_Approximation
        self.Shape = np.shape(self.Original_Matrix)
        Result = self.Singular_Value_Decomposition()
        self.Show_Image(Result)

    def Convert_to_PNG(self):
        for Image_File in os.listdir("Images_Folder"):
            if not Image_File.endswith(".png"):
                New_Image_File = os.path.splitext(Image_File)[0] + ".png"
                os.rename(
                    os.path.join("Images_Folder", Image_File),
                    os.path.join("Images_Folder", New_Image_File),
                )

    def Convert_Image_To_Matrix(self, Image_Name):
        self.Convert_to_PNG() 
        if Image_Name in os.listdir('Images_Folder'):
            print("Found the picture:", Image_Name)
            Image = cv2.imread(os.path.join('Images_Folder', Image_Name))
            Gray_Image = np.zeros((Image.shape[0], Image.shape[1]))
            for Row in range(Image.shape[0]):
                for Col in range(Image.shape[1]):
                    Pixel = Image[Row, Col]
                    Gray_Pixel = Pixel[0] * 0.114 + Pixel[1] * 0.587 + Pixel[2] * 0.299
                    Normalization = Gray_Pixel / 255
                    Gray_Image[Row, Col] = Normalization
            return Gray_Image
        else:
            print("Picture not found!!!!")  
    
    def Find_Eigenvalues_and_Eigenvectors(self, Matrix):
        Eigenvalues, Eigenvectors = np.linalg.eig(Matrix)
        return Eigenvalues, Eigenvectors

    def Sigma_Matrix(self, Matrix):
        Singular_Values = np.sqrt(np.abs(self.Find_Eigenvalues_and_Eigenvectors(Matrix)[0]))
        D = np.zeros(self.Shape, dtype='float_')
        for i in range(min(len(Singular_Values), self.Shape[0], self.Shape[1])):
            D[i, i] = Singular_Values[i]
        return D
    
    def Singular_Value_Decomposition(self):
        A = self.Original_Matrix
        AtA = np.matmul(A.T, A)
        V = self.Find_Eigenvalues_and_Eigenvectors(AtA)[1]
        D = self.Sigma_Matrix(AtA)
        U = np.zeros((self.Shape[0], self.K), dtype='float_')
        
        for i in range(self.K):
            U[:, i] = np.matmul(A, V[:, i]) / D[i, i]
        Result = np.matmul(U[:, :self.K], D[:self.K, :self.K]) @ V[:, :self.K].T
        
        return Result

    def Show_Image(self, Result):
        Img_Array = np.clip(Result * 255, 0, 255).astype(np.uint8)
        if not os.path.exists('Result_Folder'):
            os.makedirs('Result_Folder')
        Filename = f"{self.K}_SVD_Image_Compression.jpg"
        cv2.imwrite(os.path.join('Result_Folder', Filename), Img_Array)

        Img = Image.fromarray(Img_Array)
        Img.show()

if __name__ == '__main__':
    SVD_Image_Compression_Proccessing('Cute Cat.png', 40)

```

## Hàm khởi tạo của class

```python
def __init__(self, Image_Name, Matrix_Approximation):
    self.Original_Matrix = self.Convert_Image_To_Matrix(Image_Name)
    self.K = Matrix_Approximation
    self.Shape = np.shape(self.Original_Matrix)
    Result = self.Singular_Value_Decomposition()
    self.Show_Image(Result)
```
1. Chuyển ảnh thành ma trận từ tệp ảnh được cung cấp.
2. Lưu trữ kích thước của ma trận gốc.
3. Thực hiện SVD để nén ảnh dựa trên số lượng thành phần kỳ dị 𝐾
4. Hiển thị và lưu ảnh đã nén ra tệp.
## Chuyển đổi định dạng ảnh sang PNG

### Tại sao sử dụng ảnh định dạng PNG trong Image Compression?

1. **Không mất dữ liệu (Lossless)**: PNG sử dụng phương pháp nén không làm mất dữ liệu, đảm bảo chất lượng hình ảnh không thay đổi sau khi nén, rất phù hợp cho các ứng dụng cần độ chính xác cao (ví dụ: đồ họa hoặc hình ảnh y tế).

2. **Hỗ trợ kênh alpha (độ trong suốt)**: PNG hỗ trợ kênh alpha, cho phép hiển thị các vùng trong suốt, điều này rất quan trọng trong thiết kế đồ họa hoặc các ứng dụng cần lớp nền trong suốt.

3. **Khả năng nén tốt cho hình ảnh ít màu**: PNG hiệu quả hơn đối với các hình ảnh có ít màu sắc hoặc sự chuyển màu rõ rệt, như biểu đồ, biểu tượng hoặc ảnh chứa văn bản.

Tuy nhiên, với hình ảnh phức tạp, nhiều màu sắc, định dạng JPEG có thể hiệu quả hơn trong việc giảm kích thước file nhờ nén có mất dữ liệu.

```python
    def Convert_to_PNG(self):
        for Image_File in os.listdir("Images_Folder"):
            if not Image_File.endswith(".png"):
                New_Image_File = os.path.splitext(Image_File)[0] + ".png"
                os.rename(
                    os.path.join("Images_Folder", Image_File),
                    os.path.join("Images_Folder", New_Image_File),
                )
```

## Image Compression

SVD có thể rất hữu ích trong việc tìm kiếm các mối quan hệ quan trọng trong dữ liệu. Điều này có nhiều ứng dụng trong học máy, tài chính và khoa học dữ liệu. Một trong những ứng dụng của SVD là trong **image compression**. Mặc dù không có định dạng hình ảnh lớn nào sử dụng SVD do độ phức tạp tính toán của nó, SVD vẫn có thể được áp dụng trong các trường hợp khác như một cách để nén dữ liệu.


## Giải Thích Về Chuyển Đổi Hình Ảnh Sang Ảnh Xám Và Chuẩn Hóa Dữ Liệu.

Khi bạn đọc một hình ảnh và chuyển nó thành ảnh xám, quá trình chuyển đổi không đơn giản là thay thế một ma trận 3x3 (mà bạn thấy khi đọc hình ảnh màu) bằng một ma trận 3x3 khác. Thay vào đó, đó là một quá trình xử lý hình ảnh bao gồm nhiều bước.

```python
def Convert_Image_To_Matrix(self, Image_Name):
    self.Convert_to_PNG() 
    if Image_Name in os.listdir('Images_Folder'):
        print("Found the picture:", Image_Name)
        Image = cv2.imread(os.path.join('Images_Folder', Image_Name))
        Gray_Image = np.zeros((Image.shape[0], Image.shape[1]))
        for Row in range(Image.shape[0]):
            for Col in range(Image.shape[1]):
                Pixel = Image[Row, Col]
                Gray_Pixel = Pixel[0] * 0.114 + Pixel[1] * 0.587 + Pixel[2] * 0.299
                Normalization = Gray_Pixel / 255
                Gray_Image[Row, Col] = Normalization
        return Gray_Image
    else:
        print("Picture not found!!!!") 
```
![png](Markdown_Folder/Output_1.png)

### 1. Đọc Hình Ảnh
Khi bạn sử dụng `cv2.imread()` để đọc hình ảnh, OpenCV trả về một ma trận ba chiều (3D) với kích thước $\ H \times W \times 3$, trong đó:
- ***H***: Chiều cao của hình ảnh.
- ***W***: Chiều rộng của hình ảnh.
- ***3***: Ba kênh màu (Blue, Green, Red).

Khi lấy ra ma trận $\ 3 \times 3$ nó sẽ có dạng:

```python
import cv2
Image = cv2.imread(os.path.join('Images_Folder','Cute Cat.png'))
Image_3x3 = Image[0:3, 0:3]
print(Image_3x3)
```
Sau khi chạy đoạn code trên thì sẽ ra ma trận $3 \times 3$ với mỗi vị trí sẽ là **vector** với 3 hàng :
|    | x1                | x2                | x3                |
|----|-------------------|-------------------|-------------------|
| y1 | [1, 8, 41]       | [1, 11, 45]      | [1, 13, 49]      |
| y2 | [7, 20, 55]      | [11, 26, 62]     | [21, 35, 71]     |
| y3 | [21, 40, 77]     | [26, 45, 82]     | [32, 50, 89]     |

### 2. Chuyển Đổi Sang Ảnh Xám
Khi bạn chuyển đổi hình ảnh màu sang ảnh xám bằng `cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)`, OpenCV sử dụng công thức sau để tính giá trị độ xám cho mỗi pixel:

$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$

Trong đó:
- \( R, G, B \): Giá trị màu của từng pixel trong ba kênh màu.
- \( Y \): Giá trị độ xám tương ứng.

Vì mặc định của thư viện OpenCV khi đọc ảnh là ở định đạng màu **BGR** nên ta phải nhân tương ứng với vector:

Ở điểm ảnh đầu tiên:

|B|G|R|
|-|-|-| 
|1|8|41|

$Y = 0.114 \cdot B + 0.587 \cdot G + 0.299 \cdot R$

$Y = 0.114 \cdot 1 + 0.587 \cdot 8 + 0.299 \cdot 41$

$Y = 17.069$

Sau khi tính toán thì sẽ thu được ma trận $3 \times 3$ như sau:

|    | x1    | x2    | x3     |
|----|-------|-------|--------|
| y1 |17.069|19.557|22.053|
| y2 |85.468|95.171|106.779|
| y3 |162.292|177.031|192.719|

### 3. Ma Trận Ảnh Xám
Kết quả của quá trình chuyển đổi này là một ma trận hai chiều (2D) với kích thước $\ H  \times W$. Mỗi giá trị trong ma trận này đại diện cho độ xám của pixel tại vị trí tương ứng.

---

### Tóm lại
- **Ma trận 3x3** mà bạn thấy ở hình ảnh màu chứa thông tin cho ba kênh màu.
- **Ma trận 2D** (ảnh xám) chỉ chứa giá trị độ xám cho mỗi pixel, không còn thông tin màu sắc riêng biệt.

Do đó, ảnh xám và ma trận ban đầu không giống nhau về kích thước và nội dung. Ảnh xám đơn giản hóa hình ảnh bằng cách giảm số lượng kênh màu từ ba xuống một, trong khi vẫn giữ lại thông tin ánh sáng tổng thể.

## Tại sao trong SVD Image Compression phải chuyển qua ảnh xám?

Chuyển đổi hình ảnh sang màu xám trong **Image Compression bằng SVD** có một số lý do chính, bao gồm:

### 1. Giảm Kích Thước Dữ Liệu
- Hình ảnh màu thường có ba kênh màu (Red, Green, Blue), trong khi hình ảnh xám chỉ có một kênh. Điều này có nghĩa là khi chuyển đổi sang ảnh xám, số lượng dữ liệu cần xử lý giảm đi một phần ba. Việc giảm kích thước này giúp tăng tốc độ xử lý và giảm lượng bộ nhớ cần thiết.

### 2. Giảm Độ Phức Tạp Tính Toán
- Việc làm việc với ma trận 2D (hình ảnh xám) thay vì ma trận 3D (hình ảnh màu) đơn giản hóa các phép toán. SVD cần phải thực hiện trên các ma trận lớn, và việc giảm kích thước ma trận sẽ giúp giảm độ phức tạp tính toán.


## Chuẩn hóa (Normalization)

**Chuẩn hóa** (_Normalization_) là quá trình biến đổi các giá trị của một tập dữ liệu sao cho chúng nằm trong một khoảng xác định, thường là từ **0 đến 1** hoặc từ **-1 đến 1**. Điều này giúp đảm bảo rằng tất cả các giá trị đều có cùng thang đo, giúp các thuật toán xử lý hiệu quả hơn.

Trong **Image Compression**, khi xử lý hình ảnh dưới dạng ma trận số, giá trị của mỗi pixel thường nằm trong khoảng từ 0 đến 255 (đối với ảnh 8-bit). Việc **chia ma trận cho 255** giúp chuẩn hóa (Normalization) các giá trị pixel về khoảng [0, 1].

$$
\text{giá trị chuẩn hóa} = \frac{\text{giá trị pixel}}{255}
$$

### Lấy ra ma trận $3 \times 3$ đã chuyển sang ảnh xám để tính:

**Ma trận ban đầu**
|    | x1                | x2                | x3                |
|----|-------------------|-------------------|-------------------|
| y1 | 17.069            | 19.557            | 22.053            |
| y2 | 85.468            | 95.171            | 106.779           |
| y3 | 162.292           | 177.031           | 192.719           |

**Ma trận sau khi chia cho 255**

|    | x1                  | x2                  | x3                  |
|----|---------------------|---------------------|---------------------|
| y1 | $\frac{17.069}{255}$| $\frac{19.557}{255}$| $\frac{22.053}{255}$|
| y2 | $\frac{85.468}{255}$| $\frac{95.171}{255}$| $\frac{106.779}{255}$|
| y3 | $\frac{162.292}{255}$| $\frac{177.031}{255}$| $\frac{192.719}{255}$|

**Kết quả cuối cùng**:
|    | x1       | x2       | x3       |
|----|----------|----------|----------|
| y1 | 0.06689  | 0.0767   | 0.0864   |
| y2 | 0.335    | 0.373    | 0.418    |
| y3 | 0.636    | 0.694    | 0.754    |

## Tính Singular Value Decomposition
Để tính được Singular Value Decomposition ta cần 3 phần tử và cũng là 3 ma trận riêng biệt:

$$
    \mathbf{A} = \mathbf{U\Sigma V^\mathsf{T}}
$$

- $\mathbf U$: Đây là ma trận trực giao(orthogonal matrix) kích thước $m \times m$. Chứa *Eigenvector* của $\mathbf{AA^\mathsf{T}}$.
- $\mathbf \Sigma$: Đây là ma trận đường chéo (diagonal matrix) kích thước $m \times n$. Các giá trị trên đường chéo được ký hiệu là $\sigma_i$ và được gọi là *singular values* của $\mathbf A$.
- $\mathbf V^\mathsf{T}$: Đây là ma trận trực giao chuyển vị (transposed orthogonal matrix) kích thước $n \times n$. Chứa *Eigenvector* của $\mathbf{A^\mathsf{T}A}$.

## Eigenvalues và Eigenvectors

Dùng thư viện numpy để tính *Eigenvalues* và *Eigenvectors* của ma trận vuông

> **linalg.eig()** -> Trả về một tuple gồm một mảng chứa *Eigenvalues* và một ma trận chứa các *Eigenvectors* tương ứng.

```python
def Find_Eigenvalues_and_Eigenvectors(self, Matrix):
    Eigenvalues, Eigenvectors = np.linalg.eig(Matrix)
    return Eigenvalues, Eigenvectors
```

### Tính Eigenvalues và Eigenvectors:

Để tìm các **Eigenvalues** **$\lambda$**, ta cần giải phương trình đặc trưng:

$$
    \det
    \begin{pmatrix}
    \mathbf{P - \lambda I} 
    \end{pmatrix}= 0
$$

Với ma trận **$\mathbf{P}$** đã chuẩn hóa:

$$
    \mathbf{P - \lambda I} =
    \left(\begin{array}{cc}
    0.06689 & 0.0767  & 0.0864  \\
    0.335   & 0.373   & 0.418   \\
    0.636   & 0.694   & 0.754  
    \end{array}\right)
    -\lambda \left(\begin{array}{cc}
    1   & 0 & 0  \\
    0   & 1 & 0   \\
    0   & 0 & 1  
    \end{array}\right)        
$$

$$
    \mathbf{P - \lambda I} =
    \begin{pmatrix}
    0.06689 - \lambda   & 0.0767            & 0.0864 \\
    0.335               & 0.373 - \lambda   & 0.418 \\
    0.636               & 0.694             & 0.754 - \lambda
    \end{pmatrix}       
$$


**Giải phương trình:**


$$
    \det
    \begin{pmatrix}
    0.06689 - \lambda & 0.0767         & 0.0864         \\
    0.335             & 0.373 - \lambda & 0.418          \\
    0.636             & 0.694           & 0.754 - \lambda
    \end{pmatrix}
    = 0
$$

Sau khi giải phương trình bậc 3 này, ta sẽ tìm được các **Eigenvalues** $\lambda_1, \lambda_2, \lambda_3$.  

$$
    \lambda_1 = 1.20560426,\quad \lambda_2 = -0.00123017,\quad \lambda_3 = -0.0104841
$$


Tiếp theo, ta sẽ thay từng **Eigenvalue** vào $(\mathbf{P - \lambda I})  \mathbf{v} = 0$ để tìm **Eigenvectors** tương ứng.

Trong đó: **Eigenvector** cần tìm là 

$$
    \mathbf{v} =
    \begin{pmatrix}
    \mathbf{x_1} \\
    \mathbf{x_2} \\
    \mathbf{x_3} \\
    \end{pmatrix}
$$

- #### Với  $\lambda_1 = 1.20560426$ thay vào $(\mathbf{P - \lambda I})  \mathbf{v} = 0$ ta được:

$$
    \begin{pmatrix}
    −1.13871426 & 0.0767         & 0.0864         \\
    0.335             & −0.83260426 & 0.418          \\
    0.636             & 0.694           & −0.45160426 \\
    \end{pmatrix}
    \begin{pmatrix}
    \mathbf{x_1} \\
    \mathbf{x_2} \\
    \mathbf{x_3} \\
    \end{pmatrix}
    = 
    \begin{pmatrix}
    0 \\
    0 \\ 
    0 \\
    \end{pmatrix}  
$$
  
#### Lập hệ phương trình tuyến tính từ phép nhân ma trận:  

$$
    \begin{aligned}
    −1.13871426\cdot&\mathbf{x_1}\quad ​+ &0.0767\cdot&\mathbf{x_2}\quad ​+ &0.0864\cdot&\mathbf{x_3}\quad= 0 ​\\
    0.335\cdot&\mathbf{x_1}\quad ​+ &0.83260426\cdot&\mathbf{x_2}\quad ​+ &0.418\cdot&\mathbf{x_3}\quad = 0 \\
    0.636\cdot&\mathbf{x_1}\quad ​+ &0.694\cdot&\mathbf{x_2}\quad − ​&0.45160426\cdot&\mathbf{x_3}\quad = 0 \\
    \end{aligned}
$$
Giải hệ phương trình ta được: $\quad\begin{aligned}\mathbf{x_1} = -0.09841821,\quad\mathbf{x_2} = -0.47783762,\quad\mathbf{x_3} = -0.87291756\quad\end{aligned}$
    
- #### Với  $\lambda_2 = −0.00123017$:

$$
    \begin{aligned}
    \mathbf{x_1} = -0.47083932,\quad
    \mathbf{x_2} = 0.81054406,\quad
    \mathbf{x_3} = -0.34832264\quad
    \end{aligned}
$$

- #### Với  $\lambda_3 = -0.0104841$:

$$
    \begin{aligned}
    \mathbf{x_1} = -0.20082236,\quad
    \mathbf{x_2} = -0.63637886,\quad
    \mathbf{x_3} = 0.7447767\quad
    \end{aligned}
$$

#### Ta tìm được **Eigenvectors** - một ma trận $n \times n$ mà mỗi cột tương ứng với một **Eigenvector** của một **Eigenvalue**.

$$
    \begin{pmatrix}
    -0.09841821 & -0.47083932 & -0.20082236 \\
    -0.47783762 &  0.81054406 & -0.63637886 \\
    -0.87291756 & -0.34832264 &  0.7447767
    \end{pmatrix}
$$
