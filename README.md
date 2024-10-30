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
        if self.Original_Matrix is not False:
            self.Shape = np.shape(self.Original_Matrix)
            Result = self.Singular_Value_Decomposition()
            self.Show_Image(Result)
        else:
            print(f"Picture not found!!!!")


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
            return False
    
    def Find_Eigenvalues_and_Eigenvectors(self, Matrix):
        Eigenvalues, Eigenvectors = np.linalg.eig(Matrix)
        return Eigenvalues, Eigenvectors

    def Sigma_Matrix(self, Matrix):
        Singular_Values = np.sqrt(np.abs(self.Find_Eigenvalues_and_Eigenvectors(Matrix)[0]))
        Sigma_Matrix = np.zeros(self.Shape, dtype='float_')
        for i in range(min(len(Singular_Values), self.Shape[0], self.Shape[1])):
            Sigma_Matrix[i, i] = Singular_Values[i]
        return Sigma_Matrix
    
    def Singular_Value_Decomposition(self):
        A = self.Original_Matrix
        AtA = np.matmul(A.T, A)
        U = np.zeros((self.Shape[0], self.K), dtype='float_')
        D = self.Sigma_Matrix(AtA)
        V = self.Find_Eigenvalues_and_Eigenvectors(AtA)[1]
        
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
- $R, G, B$: Giá trị màu của từng pixel trong ba kênh màu.
- $Y$: Giá trị độ xám tương ứng.

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

### Ví dụ với ma trận **$\mathbf{P}$** đã chuẩn hóa:

#### Cách tính Eigenvectors và Eigenvectors:

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

Trong đó **Eigenvector** cần tìm là:

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
    −1.13871426 & 0.0767        & 0.0864    \\
    0.335       & −0.83260426   & 0.418     \\
    0.636       & 0.694         & −0.45160426
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
    −1.13871426\cdot\mathbf{x_1} ​+ 0.0767\cdot\mathbf{x_2} ​+ 0.0864\cdot\mathbf{x_3}= 0 ​\\
    0.335\cdot\mathbf{x_1} ​+ 0.83260426\cdot\mathbf{x_2} ​+ 0.418\cdot\mathbf{x_3} = 0 \\
    0.636\cdot\mathbf{x_1} ​+ 0.694\cdot\mathbf{x_2} − ​0.45160426\cdot\mathbf{x_3} = 0 \\
    \end{aligned}
$$

Giải hệ phương trình ta được: 

$$
    \begin{aligned}
    \mathbf{x_1} = -0.09841821,\quad
    \mathbf{x_2} = -0.47783762,\quad
    \mathbf{x_3} = -0.87291756\quad
    \end{aligned}
$$
    
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

### Ma trận **Σ** từ các Singular Values

Trong **Singular Value Decomposition** ma trận **Σ** chứa các **Singular Values** của ma trận gốc


### Cấu trúc của ma trận Σ

- **Ma trận Σ** là một ma trận chéo, nghĩa là chỉ có các phần tử nằm trên đường chéo chính là khác không, còn lại là **0**.
  
- Các phần tử trên đường chéo của **Σ** là các **Singular Values** của ma trận $A $. Các giá trị này được sắp xếp theo thứ tự giảm dần:

$$
\Sigma = 
\begin{pmatrix}
\sigma_1 & 0 & \ldots & 0 \\
0 & \sigma_2 & \ldots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \ldots & \sigma_n \\
\end{pmatrix}
$$

Trong đó:

$$
\quad \sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_n \geq 0\
$$

- **n** là số hạng không bằng không, hoặc số hạng không tầm thường trong ma trận.

### Ý nghĩa của các **Singular Values**

- **Độ lớn**: Các giá trị đặc biệt thể hiện độ lớn của các thành phần trong không gian. Giá trị lớn hơn nghĩa là thành phần đó đóng góp nhiều hơn vào ma trận gốc.

- **Biểu diễn kích thước**: Các giá trị này cho phép chúng ta hiểu được cấu trúc của ma trận, từ đó có thể phân tích và biểu diễn dữ liệu.

- **Giảm chiều (Dimensionality Reduction)**: SVD được sử dụng trong các kỹ thuật như PCA (Principal Component Analysis) để giảm chiều dữ liệu. Bằng cách giữ lại các giá trị đặc biệt lớn nhất và loại bỏ những cái nhỏ, chúng ta có thể giảm số chiều mà vẫn giữ được thông tin quan trọng.

```python
def Sigma_Matrix(self, Matrix):
    Singular_Values = np.sqrt(np.abs(self.Find_Eigenvalues_and_Eigenvectors(Matrix)[0]))
    Sigma_Matrix = np.zeros(self.Shape, dtype='float_')
    for i in range(min(len(Singular_Values), self.Shape[0], self.Shape[1])):
        Sigma_Matrix[i, i] = Singular_Values[i]
    return Sigma_Matrix
```

Cách hoạt động của hàm `Sigma_Matrix()` như sau:

1. Tính căn bậc hai của **Eigenvalues** của ma trận **$\mathbf U$** hoặc **$\mathbf V$** để tìm **Singular Values**.
2. Tạo ma trận **Σ** (ma trận đường chéo) chứa các giá trị kỳ dị.

Ví dụ ta có 1 ma trận **A** với kích thước là $5 \times 4$

$$
    \mathbf A =
    \begin{pmatrix}
    1 & 2 & 3 & 4 \\
    5 & 6 & 7 & 8 \\
    9 & 10 & 11 & 12 \\
    13 & 14 & 15 & 16 \\
    17 & 18 & 19 & 20 \\
    \end{pmatrix}
$$

---

### Tính ma trận **U** = $\mathbf A \mathbf A^\mathsf{T}$

$$
    \mathbf A \mathbf A^\mathsf{T} = 
    \begin{pmatrix}
    1 & 2 & 3 & 4 \\
    5 & 6 & 7 & 8 \\
    9 & 10 & 11 & 12 \\
    13 & 14 & 15 & 16 \\
    17 & 18 & 19 & 20 \\
    \end{pmatrix}
    \begin{pmatrix}
    1 & 5 & 9 & 13 & 17 \\
    2 & 6 & 10 & 14 & 18 \\
    3 & 7 & 11 & 15 & 19 \\
    4 & 8 & 12 & 16 & 20 \\
    \end{pmatrix}
$$

Kết quả phép nhân ma trận sẽ là ma trận **U** có kích thước $5 \times 5$:

$$
    \mathbf A \mathbf A^\mathsf{T} = 
    \begin{pmatrix}
    30 & 70 & 110 & 150 & 190 \\
    70 & 174 & 278 & 382 & 486 \\
    110 & 278 & 446 & 614 & 782 \\
    150 & 382 & 614 & 846 & 1078 \\
    190 & 486 & 782 & 1078 & 1374 \\
    \end{pmatrix}
$$

**Eigenvalue** của ma trận **U** = $\mathbf A \mathbf A^\mathsf{T}$:

$$
    \lambda_1 = 2.86441422e+03,\quad \lambda_2 = 5.58578432e+00,\quad \lambda_3 = -4.67578995e-14,\quad \lambda_4 = 1.22637131e-15,\quad \lambda_5 = -9.35600838e-16
$$

**Eigenvectors** của ma trận **U** = $\mathbf A \mathbf A^\mathsf{T}$:

$$
    \begin{pmatrix}
    -0.09654784 &  0.76855612 & -0.47716104 &  0.18582136 & -0.27376797 \\
    -0.24551564 &  0.48961420 &  0.32759876 &  0.03539164 & -0.03585407 \\
    -0.39448345 &  0.21067228 &  0.68166006 & -0.74154641 &  0.26304427 \\
    -0.54345125 & -0.06826963 & -0.43747224 &  0.63363247 &  0.67654556 \\
    -0.69241905 & -0.34721155 & -0.09462554 & -0.11329906 & -0.62996778
    \end{pmatrix}
$$

---

### Tính ma trận **V** = $\mathbf A^\mathsf{T} \mathbf A$

$$
    \mathbf A^\mathsf{T} \mathbf A = 
    \begin{pmatrix}
    1 & 5 & 9 & 13 & 17 \\
    2 & 6 & 10 & 14 & 18 \\
    3 & 7 & 11 & 15 & 19 \\
    4 & 8 & 12 & 16 & 20 \\
    \end{pmatrix}    
    \begin{pmatrix}
    1 & 2 & 3 & 4 \\
    5 & 6 & 7 & 8 \\
    9 & 10 & 11 & 12 \\
    13 & 14 & 15 & 16 \\
    17 & 18 & 19 & 20 \\
    \end{pmatrix}
$$

Kết quả phép nhân ma trận sẽ là ma trận **U** có kích thước $5 \times 5$:   

$$
    \mathbf A^\mathsf{T} \mathbf A = 
    \begin{pmatrix}
    335 & 400 & 465 & 530 \\
    400 & 480 & 560 & 640 \\
    465 & 560 & 655 & 750 \\
    530 & 640 & 750 & 860 \\
    \end{pmatrix}
$$

**Eigenvalue** của ma trận **V** = $\mathbf A^\mathsf{T} \mathbf A$:

$$
    \lambda_1 = 2.86441422e+03,\quad \lambda_2 = 5.58578432e+00,\quad \lambda_3 = -4.67578995e-14,\quad \lambda_4 = 1.22637131e-15
$$

**Eigenvectors** của ma trận **V** = $\mathbf A^\mathsf{T} \mathbf A$:

$$
    \begin{pmatrix}
    -0.44301884 &  0.70974242 & -0.36645027 & -0.07257593 \\
    -0.47987252 &  0.26404992 &  0.79201995 &  0.50141641 \\
    -0.51672621 & -0.18164258 & -0.48468907 & -0.78510502 \\
    -0.55357989 & -0.62733508 &  0.05911940 &  0.35626454
    \end{pmatrix}
$$

---

### Xây dựng **Ma trận Σ** từ **Singular Values**:

- **Ma trận Σ** chứa các Singular Values $\sigma_i$ trên đường chéo chính.

- Các **Singular Values** là căn bậc hai của **Eigenvalues** của **$\mathbf A \mathbf A^\mathsf{T}$** hoặc **$\mathbf A^\mathsf{T} \mathbf A$**.

```python
def Sigma_Matrix(self, Matrix):
    Singular_Values = np.sqrt(np.abs(self.Find_Eigenvalues_and_Eigenvectors(Matrix)[0]))
    Sigma_Matrix = np.zeros(self.Shape, dtype='float_')
    for i in range(min(len(Singular_Values), self.Shape[0], self.Shape[1])):
        Sigma_Matrix[i, i] = Singular_Values[i]
    return Sigma_Matrix
```

### Tính Singular Values $\sigma_i$

**Eigenvalue** của ma trận **V** = $\mathbf A^\mathsf{T} \mathbf A$:

$$
    \lambda_1 = 2.86441422e+03,\quad \lambda_2 = 5.58578432e+00,\quad \lambda_3 = 1.90476307e-13,\quad \lambda_4 = 4.92313283e-14
$$

Ma trận **Σ** có kích thước $m \times n$:

Với $\mathbf m$ là số hàng của **A** và $\mathbf n$ là số cột của **A**.

$$
    \Sigma = 
    \begin{pmatrix}
    \sqrt 2.86441422e+03 & 0 & 0 & 0 \\
    0 & \sqrt 5.58578432e+00 & 0 & 0 \\
    0 & 0 & \sqrt 1.90476307e-13 & 0 \\
    0 & 0 & 0 & \sqrt 4.92313283e-14 \\
    0 & 0 & 0 & 0
    \end{pmatrix}
$$

### Chọn **Eigenvalues** từ **U** = $\mathbf A \mathbf A^\mathsf{T}$ khác gì khi chọn **Eigenvalues** từ **V** = $\mathbf A^\mathsf{T} \mathbf A$

| Giống nhau | Khác nhau |
|------------|-----------|
| Các **Eigenvalue** của $\mathbf{A} \mathbf{A}^\mathsf{T}$ và $\mathbf{A}^\mathsf{T} \mathbf{A}$ có cùng *tập hợp các giá trị*. <br> Đều cung cấp bình phương của cùng các **Singular Values** $\sigma_1^2$. | Số lượng phần tử vì kích thước của $\mathbf{A} \mathbf{A}^\mathsf{T}$ và $\mathbf{A}^\mathsf{T} \mathbf{A}$ khác nhau. <br> $\mathbf{U} = \mathbf{A} \mathbf{A}^\mathsf{T}$ là ma trận $m \times m$ <br> $\mathbf{V} = \mathbf{A}^\mathsf{T} \mathbf{A}$ là ma trận $n \times n$ |

$\Rightarrow$ Có thể chọn **Eigenvalue** từ **U** = $\mathbf A \mathbf A^\mathsf{T}$ hoặc **V** = $\mathbf A^\mathsf{T} \mathbf A$


### Kết quả cuối cùng của *Singular Value Decomposition*

$$
    \mathbf A = \mathbf{U \Sigma V^\mathsf{T}}
$$

#### Sau khi tìm được các ma trận $\mathbf U$, $\Sigma$ và ma trận chuyển vị $V^\mathsf{T}$ 

$$
    \mathbf A =
    \begin{pmatrix}
    -0.09654784 &  0.76855612 & -0.47716104 &  0.18582136 & -0.27376797 \\
    -0.24551564 &  0.48961420 &  0.32759876 &  0.03539164 & -0.03585407 \\
    -0.39448345 &  0.21067228 &  0.68166006 & -0.74154641 &  0.26304427 \\
    -0.54345125 & -0.06826963 & -0.43747224 &  0.63363247 &  0.67654556 \\
    -0.69241905 & -0.34721155 & -0.09462554 & -0.11329906 & -0.62996778
    \end{pmatrix}
    \begin{pmatrix}
    \sqrt 2.86441422e+03 & 0 & 0 & 0 \\
    0 & \sqrt 5.58578432e+00 & 0 & 0 \\
    0 & 0 & \sqrt 1.90476307e-13 & 0 \\
    0 & 0 & 0 & \sqrt 4.92313283e-14 \\
    0 & 0 & 0 & 0
    \end{pmatrix}
    \begin{pmatrix}
    -0.44301884 & -0.47987252 & -0.51672621 & -0.55357989 \\
    0.70974242 &  0.26404992 & -0.18164258 & -0.62733508 \\
    -0.36645027 &  0.79201995 & -0.48468907 &  0.05911940 \\
    -0.07257593 &  0.50141641 & -0.78510502 &  0.35626454
    \end{pmatrix}  
    =
    \begin{pmatrix}
    3.57838896 & 2.95925411 & 2.34011949 & 1.7209847 \\
    6.64258118 & 6.61109983 & 6.57961831 & 6.54813692 \\
    9.70677343 & 10.26294547 & 10.8191172 & 11.37528914 \\
    12.77096579 & 13.91479089 & 15.05861618 & 16.20244136 \\
    15.83515804 & 17.56663653 & 19.29811507 & 21.02959357
    \end{pmatrix}      
$$

> Kết quả có sai số cao vì khi tìm **Eigenvectors** từ hàm `linalg.eig()` của thư viện `Numpy`:

Sự khác nhau của hàm *`linalg.eig()`* và *`linalg.eigh()`*

| **Thuộc tính**                     | **`linalg.eigh()`**                                                | **`linalg.eig()`**                                              |
|-------------------------------------|-----------------------------------------------------------|-------------------------------------------------------|
| **Mục đích**                       | Dùng cho ma trận đối xứng hoặc Hermitian $\mathbf A =\mathbf A^\mathsf{T}$ hoặc $\mathbf A =\mathbf A^\mathsf{H}$. | Dùng cho mọi loại ma trận vuông.                      |
| **Đầu vào**                        | Ma trận đối xứng thực hoặc Hermitian phức.                | Bất kỳ ma trận vuông nào.                             |
| **Kết quả**                        | Giá trị riêng luôn là số thực (đối với ma trận đối xứng). | Giá trị riêng có thể là thực hoặc phức.               |
| **Hiệu suất**                      | Tối ưu hơn về tốc độ và độ chính xác cho ma trận đối xứng hoặc Hermitian. | Chậm hơn vì hỗ trợ cho mọi loại ma trận. |
| **Sắp xếp kết quả**                | Giá trị riêng được sắp xếp theo thứ tự tăng dần.         | Giá trị riêng không được sắp xếp.                     |
| **Hàm tương ứng trong Linear Algebra** | **Phân rã Eigen** cho ma trận Hermitian.                    | **Phân rã Eigen** tổng quát.                              |

$\Rightarrow$ Để cho ra kết quả đúng nhất khi tính Singular Value Decomposition ta sử dụng hàm *`linalg.eigh()`*



## **Matrix Approximation** - Ma trận xấp xỉ

Matrix Approximation là một kỹ thuật trong đại số tuyến tính và khoa học dữ liệu nhằm tìm ra một ma trận gần đúng với ma trận gốc, nhưng có thứ hạng (rank) thấp hơn hoặc đơn giản hơn. Mục tiêu của việc xấp xỉ là giảm kích thước và độ phức tạp của dữ liệu, đồng thời vẫn giữ lại càng nhiều thông tin quan trọng càng tốt.

### Mô hình của *Matrix Approximation*.

Giả sử bạn có một ma trận $A \in \mathbb{R}^{m \times n}$, ma trận này có thể được xấp xỉ bằng một ma trận $\hat{A}$ có thứ hạng thấp hơn:

$$
A \approx \hat{A}
$$

Trong đó:

- **A**: Ma trận gốc.
- **$\hat{A}$**: Ma trận xấp xỉ có thứ hạng thấp hơn.

###  **Matrix Approximation** trong **Singular Value Decomposition**

Kỹ thuật này xấp xỉ ma trận bằng cách sử dụng các *Singular Values* lớn nhất, giữ lại thông tin quan trọng và bỏ đi các giá trị nhỏ không đáng kể.

$$
    \mathbf{\hat{A}}_k = \mathbf{U_k\Sigma_k V^\mathsf{T}_k}
$$

- Chỉ giữ lại **$\mathbf{k}$** giá trị *Singular Values* lớn nhất.
- **$\mathbf{k}$-rank approximation** giúp giảm bớt kích thước dữ liệu.

$$
    \\
    \\
$$

Ví dụ ta lấy **$\mathbf{k} = 3$** với ma trận $\mathbf A$ đã tìm được $\mathbf U, \Sigma, \mathbf V^\mathsf{T}$ .

$$
    \mathbf A =
    \begin{pmatrix}
    1 & 2 & 3 & 4 \\
    5 & 6 & 7 & 8 \\
    9 & 10 & 11 & 12 \\
    13 & 14 & 15 & 16 \\
    17 & 18 & 19 & 20 \\
    \end{pmatrix}
$$

- Với **$\Sigma_k$** :Giữ lại 3 *Singular Values* lớn nhất trong ma trận **$\Sigma$**. Ma trận **$\Sigma_3$** có kích thước là $3 \times 3$

$$
    \Sigma = 
    \begin{pmatrix}
    \sqrt 2.86441422e+03 & 0 & 0  \\
    0 & \sqrt 5.58578432e+00 & 0  \\
    0 & 0 & \sqrt 1.90476307e-13  \\
    \end{pmatrix}
$$

- Với $\mathbf{U_k}$: Giữ lại 3 cột đầu tiên của ma trận **$\mathbf U$**. Ma trận **$\mathbf{U_3}$** có kích thước là $5 \times 3$
  
$$
    \mathbf{U_3} = 
    \begin{pmatrix}
    -0.09654784 &  0.76855612 & -0.47716104 \\
    -0.24551564 &  0.48961420 &  0.32759876 \\
    -0.39448345 &  0.21067228 &  0.68166006 \\
    -0.54345125 & -0.06826963 & -0.43747224 \\
    -0.69241905 & -0.34721155 & -0.09462554 
    \end{pmatrix}
$$

- Với $\mathbf{V^\mathsf{T}_k}$: Giữ lại 3 hàng đầu tiên của ma trận **$\mathbf V^\mathsf{T}$**. $\mathbf{V^\mathsf{T}_3}$: có kích thước là $3 \times 4$

$$
    \mathbf V^\mathsf{T}_3=
    \begin{pmatrix}
    -0.44301884 & -0.47987252 & -0.51672621 & -0.55357989 \\
    0.70974242 &  0.26404992 & -0.18164258 & -0.62733508 \\
    -0.36645027 &  0.79201995 & -0.48468907 &  0.05911940 \\
    \end{pmatrix} 
$$

Kết quả cuối cùng của *Singular Value Decomposition*

$$
    \mathbf{\hat{A}}_k =
    \begin{pmatrix}
    -0.09654784 &  0.76855612 & -0.47716104 \\
    -0.24551564 &  0.48961420 &  0.32759876 \\
    -0.39448345 &  0.21067228 &  0.68166006 \\
    -0.54345125 & -0.06826963 & -0.43747224 \\
    -0.69241905 & -0.34721155 & -0.09462554 
    \end{pmatrix}
    \begin{pmatrix}
    \sqrt 2.86441422e+03 & 0 & 0  \\
    0 & \sqrt 5.58578432e+00 & 0  \\
    0 & 0 & \sqrt 1.90476307e-13  \\
    \end{pmatrix}
    \begin{pmatrix}
    -0.44301884 & -0.47987252 & -0.51672621 & -0.55357989 \\
    0.70974242 &  0.26404992 & -0.18164258 & -0.62733508 \\
    -0.36645027 &  0.79201995 & -0.48468907 &  0.05911940 \\
    \end{pmatrix} 
    =
    \begin{pmatrix}
    3.5783889   & 2.95925391 & 2.34011948 & 1.72098462 \\
    6.642581    & 6.61109972 & 6.57961822 & 6.54813685 \\
    9.70677341  & 10.26294563 & 10.81911733 & 11.37528936 \\
    12.77096583 & 13.91479077 & 15.05861647 & 16.20244153 \\
    15.83515801 & 17.56663642 & 19.29811531 & 21.02959375 \\
    \end{pmatrix}
$$

Ở trên là cách hoạt động cơ bản của hàm `Singular_Value_Decomposition()`:

```python
def Singular_Value_Decomposition(self):
    A = self.Original_Matrix
    AtA = np.matmul(A.T, A)
    U = np.zeros((self.Shape[0], self.K), dtype='float_')
    D = self.Sigma_Matrix(AtA)
    V = self.Find_Eigenvalues_and_Eigenvectors(AtA)[1]
    
    for Column in range(self.K):
        U[:, Column] = np.matmul(A, V[:, Column]) / D[Column, Column]
    Result = np.matmul(U[:, :self.K], D[:self.K, :self.K]) @ V[:, :self.K].T
    
    return Result
```

Hàm `Singular_Value_Decomposition()` là bước quan trọng trong quá trình nén ảnh bằng phương pháp *Singular Value Decomposition*, giúp phân tích và tạo ra các ma trận cần thiết để nén ảnh hiệu quả. Công thức liên quan giúp hình dung rõ ràng cách thức hoạt động và ý nghĩa của từng thành phần trong *Singular Value Decomposition*

### 1. Khởi tạo **Variable** để lưu Lưu trữ ma trận ảnh gốc vào biến **A**
```python
A = self.Original_Matrix
```
### 2. Tìm *Eigenvectors* của $\mathbf V= \mathbf A^\mathsf{T}\mathbf A$ .

```python
AtA = np.matmul(A.T, A)
V = self.Find_Eigenvalues_and_Eigenvectors(AtA)[1]
```

- Sử dụng hàm `Find_Eigenvalues_and_Eigenvectors()` để tìm *Eigenvectors* của ma trận $\mathbf A^\mathsf{T} \mathbf A$ có kích thước $n \times n$ , với $n$ là số cột của ma trận $\mathbf A$.
- Các *Eigenvector* sẽ tạo thành ma trận $V$ trong phân tích *Singular Value Decomposition*.

### 3. Xây dựng **Ma trận Σ**:

```python
D = self.Sigma_Matrix(AtA)
```

- Gọi hàm `Sigma_Matrix()` để tạo Ma trận đường chéo **Σ** chứa các **Eigenvalue Value** được tính từ ma trận $\mathbf A^\mathsf{T}\mathbf A$.

### 4. Tính ma trận $U$

```python
U = np.zeros((self.Shape[0], self.K), dtype='float_')
for Column in range(self.K):
    U[:, Column] = np.matmul(A, V[:, Column]) / D[Column, Column]
```
- Tạo 1 ma trận *Zero* với kích thước là $m \times k$, với $m$ là số hàng của ma trận **$A$**

- Vòng lặp `for Column in range(self.K)` duyệt qua các thành phần **Singular Values** mà ta muốn giữ lại trong quá trình *Image Compression*, số lượng thành phần được giữ lại là **$k$**.

Sử dụng công thức tính *Singular Value Decomposition* cơ bản để tính $U$.

$$
    \mathbf{A} = \mathbf{U\Sigma V^\mathsf{T}}
    \quad\rightarrow\quad
    \mathbf{U[:,\mathbf{i}]} = \frac{\mathbf{A} \cdot \mathbf{V[:,\mathbf{i}]}}{\sigma_i}
$$


- Trong đó:

  - $\mathbf{U[:,\mathbf{i}]}$ là cột thứ $i$ của ma trận $U$.
  - $A$ là ma trận gốc.
  - $\mathbf{V[:,\mathbf{i}]}$ là cột thứ $i$ của ma trận $V$
  - $\sigma_i$ là Singular Value thứ $i$.

Ví dụ ta có 1 ma trận $A$ kích thước $3 \times 2$ như sau:

$$
    \mathbf A =
    \begin{pmatrix}
    3 & 2 \\
    2 & 3 \\
    1 & 0 \\
    \end{pmatrix}
$$

**Eigenvalues** của ma trận $\mathbf A^\mathsf{T} \mathbf A$:

$$
    \lambda_1 =25, \quad \lambda_2 = 2
$$

**Eigenvector** tương ứng của ma trận $\mathbf A^\mathsf{T} \mathbf A$:

$$
    \mathbb{v_1} =
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} \\
    \end{pmatrix}
    ,\quad
    \mathbb{v_2} =
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} \\
    -\frac{1}{\sqrt{2}} \\
    \end{pmatrix}
    \quad \rightarrow \quad
    \mathbf{V} =
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\
    \end{pmatrix}
$$

**Singular Values** $\sigma_i$: 

$$
    \sigma_1 =\sqrt{25} = 5, \quad \sigma_2 = \sqrt{2}
$$

Tính các cột của ma trận $U$:

- Tính cột thứ nhất của $U$ với $i = 1$:

$$
    \mathbf{U[:,\mathbf{1}]} = \frac{\mathbf{A} \cdot \mathbf{V[:,\mathbf{1}]}}{\sigma_1} \\
$$

$$
    \frac{\mathbf{A} \cdot \mathbf{V[:,\mathbf{1}]}}{\sigma_1} = 
    \frac{
    \begin{pmatrix}
    3 & 2 \\
    2 & 3 \\
    1 & 0 \\
    \end{pmatrix}
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} \\
    \end{pmatrix}}
    {5}
    =
    \frac{
    \begin{pmatrix}
    \frac{5}{\sqrt{2}} \\
    \frac{5}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} \\       
    \end{pmatrix}}
    {5}
    = 
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} \\
    \frac{\sqrt{2}}{10} \\
    \end{pmatrix}
$$

- Tính cột thứ hai của $U$ với $i = 2$:

$$
    \mathbf{U[:,\mathbf{2}]} = \frac{\mathbf{A} \cdot \mathbf{V[:,\mathbf{2}]}}{\sigma_2} \\
$$

$$
    \frac{\mathbf{A} \cdot \mathbf{V[:,\mathbf{2}]}}{\sigma_2} = 
    \frac{
    \begin{pmatrix}
    3 & 2 \\
    2 & 3 \\
    1 & 0 \\
    \end{pmatrix}
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} \\
    -\frac{1}{\sqrt{2}} \\
    \end{pmatrix}}
    {\sqrt{2}}
    =
    \frac{
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} \\
    -\frac{1}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} \\       
    \end{pmatrix}}
    {\sqrt{2}}
    = 
    \begin{pmatrix}
    \frac{1}{2} \\
    -\frac{1}{2} \\
    \frac{1}{2} \\
    \end{pmatrix}
$$

Kết quả ma trận $U$ là ma trận có kích thước $3 \times 2$:

$$
    \mathbf{U} =
    \begin{pmatrix}
    \frac{1}{\sqrt{2}} & \frac{1}{2} \\
    \frac{1}{\sqrt{2}} & -\frac{1}{2} \\
    \frac{\sqrt{2}}{10} & \frac{1}{2} \\
    \end{pmatrix}
$$

### Dùng hàm `Find_Eigenvalues_and_Eigenvectors()` và công thức trên để tính $U$ có gì khác nhau. 


| **Tiêu chí**                        | **Phương pháp công thức**: $U[:, i] = \frac{A \cdot V[:, i]}{\sigma_i}$                         | **Phương pháp dùng eigenvectors**: $U$ là các vector riêng của $A A^T$               |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Ý tưởng**                        | Nhân ma trận $A$ với các vector riêng $V$ của $A^T A$ rồi chuẩn hóa bằng giá trị kỳ dị $\sigma_i$. | Tính Eigenvectors của ma trận $A A^T$ để tìm các cột của ma trận $U$.               |
| **Đầu vào cần thiết**               | - Ma trận $A$ <br> - Ma trận $V$ (vector riêng của $A^T A$) <br> - Giá trị kỳ dị $\sigma_i$.   | - Ma trận $A$ <br> - Eigenvalues và Eigenvectors của $A A^T$.                       |
| **Độ phức tạp tính toán**           | Nhanh hơn, vì chỉ cần nhân ma trận với vector và chia cho $\sigma_i$.                                        | Chậm hơn, vì cần tính eigenvalues và eigenvectors của ma trận $A A^T$ kích thước $n \times n$. |
| **Độ ổn định số học**               | Nhạy cảm hơn với các giá trị kỳ dị nhỏ (do phép chia).                                                          | Ổn định hơn, nhưng có thể gặp vấn đề với các ma trận gần suy biến (singular).                 |
| **Chi phí tính toán**               | **Thấp hơn**, vì chỉ cần nhân ma trận và chuẩn hóa.                                                             | **Cao hơn**, vì phải tính eigenvalues và eigenvectors của $A A^T$ với độ phức tạp $O(n^3)$. |
| **Khi nào sử dụng?**                | Khi đã có sẵn các vector riêng $V$ và giá trị kỳ dị $\sigma_i$.                                          | Khi chỉ cần tính riêng ma trận $U$ mà không cần biết các vector riêng $V$.            |
| **Ưu điểm**                        | - Nhanh và hiệu quả khi $V$ đã có. <br> - Tránh phải tính eigenvectors của $A A^T$.                     | - Phù hợp khi chỉ cần $U$. <br> - Không phụ thuộc vào $V$ hoặc giá trị kỳ dị.         |
| **Nhược điểm**                     | - Không thể dùng nếu $\sigma_i = 0$. <br> - Yêu cầu có cả $V$ và $\sigma_i$.                       | - Tốn kém tài nguyên cho ma trận lớn. <br> - Khó triển khai khi $A A^T$ không khả nghịch. |

### 5. Tạo lại **Matrix Approximation** và trả về kết quả

```python
Result = np.matmul(U[:, :self.K], D[:self.K, :self.K]) @ V[:, :self.K].T
return Result
```
  
- Tính toán **Matrix Approximation** dựa trên công thức:

$$
    \mathbf{\hat{A}}_k = \mathbf{U_k\Sigma_k V^\mathsf{T}_k}
$$


### 6. Hiển thị hình ảnh từ **Matrix Approximation**

```python
def Show_Image(self, Result):
    Img_Array = np.clip(Result * 255, 0, 255).astype(np.uint8)
    
    if not os.path.exists('Result_Folder'):
        os.makedirs('Result_Folder')

    Filename = f"{self.K}_SVD_Image_Compression.jpg"
    cv2.imwrite(os.path.join('Result_Folder', Filename), Img_Array)

    Img = Image.fromarray(Img_Array)
    Img.show()
```
- Mục đích: Hiển thị và lưu ảnh đã được nén.
- Cách hoạt động:
    - Chuẩn hóa lại ma trận ảnh về phạm vi $[0,255]$
    - Lưu ảnh nén trong thư mục `Result_Folder`. 
    - Hiển thị ảnh bằng thư viện `PIL`.

## Đánh giá mô hình

```python
def Performance_Metrics(self):
    Compression_Ratio = (self.Shape[0]*self.Shape[1]) / (self.K*(self.Shape[0] + self.Shape[1] + 1))
    print(f'Compression Ratio: {Compression_Ratio}')
    MSE = np.sum(np.square(self.Original_Matrix - self.Result))/(self.Shape[0]*self.Shape[1])
    print(f'MSE: {MSE} \n')
```

## **Compression Ratio (CR)**

- **Định nghĩa**:  
  Compression Ratio (CR) là **tỷ lệ giữa kích thước bộ nhớ của ảnh gốc và ảnh đã nén**. Nó cho biết mức độ tiết kiệm dung lượng sau khi ảnh được nén bằng phương pháp **SVD**.

- **Công thức**:  

$$
  CR = \frac{m \times n}{k \times (m + n + 1)}
$$

  Trong đó:
  - $m$: Số dòng của ma trận ảnh (chiều cao ảnh).  
  - $n$: Số cột của ma trận ảnh (chiều rộng ảnh).  
  - $k$: Số thành phần kỳ dị (**Singular Values**) được giữ lại trong quá trình nén.  

---

### **Ý nghĩa của các thành phần trong công thức**
1. **Ảnh gốc**:  
   Được biểu diễn bằng một ma trận có kích thước $m \times n$.  
   → Bộ nhớ cần để lưu ảnh gốc là $m \times n$ phần tử.

2. **Ảnh đã nén** sau SVD gồm 3 ma trận:
   - **Ma trận U**: Kích thước $m \times k$.
   - **Ma trận $\Sigma$**: Đường chéo, kích thước $k \times k$.
   - **Ma trận V**: Kích thước $n \times k$.

3. **Kích thước bộ nhớ của ảnh đã nén**:  
   Để lưu trữ ảnh sau khi nén, cần:
   - $m \times k$ phần tử cho $U$,
   - $k$ phần tử cho đường chéo của $\Sigma$,
   - $n \times k$ phần tử cho $V$.

4. **Tổng bộ nhớ của ảnh nén**: 

   $$
   k \times (m + n + 1)
   $$

---

### **Phân tích Compression Ratio qua ví dụ**

Giả sử ảnh có:
- $m = 1000$, $n = 800$ (kích thước 1000x800 pixel).
- Giữ lại $k = 50$ thành phần kỳ dị.

### **Tính CR**:

$$
CR = \frac{1000 \times 800}{50 \times (1000 + 800 + 1)} = \frac{800000}{50 \times 1801} = \frac{800000}{90050} \approx 8.89
$$

- **Kết quả**: CR bằng **8.89**, tức là ảnh nén cần ít bộ nhớ hơn ảnh gốc khoảng **8.89 lần**.

---

### **Ý nghĩa của Compression Ratio trong thực tế**
- **CR càng cao**: Mức độ nén càng lớn → Tiết kiệm nhiều bộ nhớ hơn.
- **Nhược điểm**: Khi $k$ nhỏ, CR tăng nhưng chất lượng ảnh giảm.
- **Chọn $k$ hợp lý**: Cần cân bằng giữa **CR** và **chất lượng ảnh** (thông qua các chỉ số như PSNR – Peak Signal-to-Noise Ratio).

---

## Mean Square Error (MSE)

**Mean Square Error (MSE)** là một thước đo phổ biến được sử dụng để đánh giá chất lượng của một ảnh nén so với ảnh gốc. MSE tính toán mức độ khác biệt giữa các pixel của ảnh gốc và ảnh đã nén, giúp xác định mức độ suy giảm chất lượng do quá trình nén.



### Công Thức Tính MSE
MSE được tính theo công thức sau:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (I_{original}[i] - I_{compressed}[i])^2
$$

Trong đó:
- $I_{original}[i]$ là giá trị pixel tại vị trí $i$ của ảnh gốc.
- $I_{compressed}[i]$ là giá trị pixel tại vị trí $i$ của ảnh nén.
- $N$ là tổng số pixel trong ảnh.

### Giải Thích Chi Tiết

| **Thuộc Tính**            | **Mô Tả**                                                                                          |
|---------------------------|----------------------------------------------------------------------------------------------------|
| **Ý Nghĩa của MSE**       | - MSE đo lường **độ chênh lệch** giữa hai ảnh. <br> - MSE nhỏ: Ảnh nén gần giống ảnh gốc. <br> - MSE lớn: Có sự khác biệt đáng kể. |
| **Cách Thức Tính Toán**   | - **Sự khác biệt giữa các pixel**: So sánh giá trị pixel tương ứng của ảnh gốc và ảnh nén. <br> - **Bình phương sự khác biệt**: Loại bỏ giá trị âm bằng cách bình phương. <br> - **Tính trung bình**: Lấy trung bình tất cả các giá trị bình phương để có MSE. |
| **Đơn Vị Của MSE**        | - **Squared intensity**: Nếu pixel có giá trị từ 0 đến 255, thì MSE sẽ có giá trị từ 0 đến 65,025. |
| **Ưu điểm**               | - Dễ tính toán và thực hiện. <br> - Cung cấp một giá trị duy nhất để đánh giá chất lượng ảnh nén. |
| **Nhược điểm**            | - Không phản ánh tốt cảm nhận của con người về chất lượng ảnh. <br> - Nhạy cảm với sự thay đổi lớn: Một vài pixel khác biệt lớn có thể làm tăng MSE đáng kể. |
| **Ứng Dụng của MSE**      | - **Nén ảnh**: Đánh giá chất lượng ảnh nén so với ảnh gốc. <br> - **Khôi phục ảnh**: Đo lường độ chính xác của ảnh được khôi phục. <br> - **Học máy**: Đánh giá độ chính xác của các mô hình dự đoán. |





