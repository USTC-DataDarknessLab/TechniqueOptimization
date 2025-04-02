# Sparse-dense Matrix Multiplication (SpMM)
## Concept
Multiplies a sparse matrix (稀疏矩阵) by a dense matrix (稠密矩阵), equivalent to $A@X$.
- Parameters: 
  - A (SparseMatrix) - Sparse matrix of shape (L, M) with scalar values
  - X (torch.Tensor) - Dense matrix of shape (M, N) or (M)
- Returns: The dense matrix of shape (L, N) or (L)
- Return type: torch.Tensor

## 使用库
**dgl.sparse.spmm**
- Examples:
```python
indices = torch.tensor([[0, 1, 1], [1, 0, 1]])
val = torch.randn(indices.shape[1])
A = dglsp.spmatrix(indices, val)
X = torch.randn(2, 3)
result = dglsp.spmm(A, X)
type(result)
result.shape
```

## 底层实现

### 核心优化思想
1. 避免计算零值：只计算A的非零位置与X的对应行列的乘积
2. 并行化：每个输出行可以独立计算（行级并行）
3. 内存访问优化：合并全局内存访问（Colesced Memory Access），利用共享内存缓存

### 数学定义
对于稀疏举证A的第i行和稠密矩阵X的第j列，输出$Y_{i, j}$的计算方式为：
$$
Y_{i, j} = \sum_k A_{i, k} \times X_{k, j}
$$
关键点：只有$A_{i, k}$为非零值时才需要计算，避免零值乘法

- Example

稀疏矩阵A（CSR）
```python
row_ptr = [0, 2, 4, 5]    # 行指针
col_ind = [1, 2, 0, 2, 1]  # 列索引
values = [3, 5, 2, 4, 1]   # 非零值

0, 3, 5
2, 0, 4
0, 1, 0

```
稠密矩阵X：
```python
X = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
]
```
(1) 并行化策略：
- 行级并行：每行计算独立，适合GOU的SIMT（单指令多线程）架构
- 每个线程块处理一行，线程处理该行的不同列
```cpp
__global__ void spmm_csr_kernel(int *row_ptr, int *col_ind, float *values,
                               float *X, float *Y, int M, int K) {
    int row = blockIdx.x;  // 每行一个线程块
    if (row < M) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {  // 列并行
            float sum = 0;
            for (int p = row_ptr[row]; p < row_ptr[row+1]; p++) {
                int col = col_ind[p];
                sum += values[p] * X[col * K + k];
            }
            Y[row * K + k] = sum;
        }
    }
}
```
(2) 内存访问优化
- 合并访问（Coalesced Access）：确保线程对x的访问连续（如X[col * k + k]中K对齐）。
- 共享内存缓存：韩村频繁访问x数据快，减少全局内存访问延迟。
```cpp
__shared__ float X_shared[BLOCK_SIZE][BLOCK_SIZE];
```

(3) 循环展开

(4) 向量化加载
```c++
float4 x_val = *reinterpret_cast<float4*>(&X[col * K + k]);
sum += values[p] * x_val.x;
```

