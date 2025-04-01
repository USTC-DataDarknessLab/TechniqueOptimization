# Sampled Dense-Dense Matrix Multiplication (SDDMM)

## Concept
    
**sddmm** matrix-multiplies two dense matrices $x_1$ and $x_2$, then elementwise-multiplies the result with sparse matrix $A$ at the non zero locations.

Mathematically **sddmm** is formulated as:
$$
out = (X_1@X_2)*A
$$
In particular, $x_1$ and $x_2$ can be 1-D, then $x_1@x_2$ becomes the out-product of the two vectors (which result in a matrix)

## 使用库：
1. dgl.sparse.sddmm
- Parameters: 
    - A (SparseMatrix) - Sparse matrix of shape (L, N)
    - X1 (torch.Tensor) - Dense matrix of shape (L, M) or (L, )
    - X2 (torch.Tensor) - Dense matrix of shape (M, N) or (N, )
- Returns: Sparse matrix of shape (L, N)
- Return type: Sparse Matrix

- Examples:
```python
>>> indices = torch.tensor([[1, 1, 2], [2, 3, 3]])
>>> val = torch.arange(1, 4).float()
>>> A = dglsp.spmatrix(indices, val, (3, 4))
[
 [0.0, 0.0, 0.0, 0.0],  # 第0行
 [0.0, 0.0, 1.0, 2.0],  # 第1行
 [0.0, 0.0, 0.0, 3.0]   # 第2行
]
>>> X1 = torch.randn(3, 5)
>>> X2 = torch.randn(5, 4)
>>> dglsp.sddmm(A, X1, X2)

SparseMatrix(indices=tensor([[1, 1, 2],
                             [2, 3, 3]]),
             values=tensor([-1.6585, -3.9714, -0.5406]),
             shape=(3, 4), nnz=3)
# nzz is number of non-zero (elements)
```

## 典型应用
- 在GNN中，常用于边特征更新，例如：
    - 用节点特征计算边的权重。
    - 集合邻接矩阵$A$和节点特征$X$计算消息传递的边特征
如果$A$是图的邻接举证，$X_1$和$X_2$是节点特征，那么$sddmm(A, X_1, X_2)$ 可以计算每条边上的特征（如注意力权重）


## 底层实现 (Underlying Mechanism)

### 关键优化思路
由于A是稀疏的，SDDMM的核心优化是：
- 避免计算完整的$X_1@X_2$ (稠密矩阵乘法)，而是只计算A的非零位置对应的值。
- 这样计算量从$O(L* N * M)$降低到$O(nnz * M)$

### Python实现
- (1) 提取A的非零坐标（行、列）
```python
row_indices = A.indices[0] # [1, 1, 2]
col_indices = A.indices[1] # [2, 3, 3]
```
- (2) 只计算$X_1@X_2$在A非零位置的值
    - 普通举证乘法需要计算3*4=12个值
    - SDDMM只计算在(1, 2), (1, 3), 和(2, 3)的值
```python
# 只取 X1 和 X2 的对应行/列
X1_rows = X1[row_indices]  # shape=(3,5), 取 X1 的第1、1、2行
X2_cols = X2[:, col_indices]  # shape=(5,3), 取 X2 的第2、3、3列
D_sampled = (X1_rows * X2_cols.T).sum(dim=1)  # 逐元素乘后求和
```

- (3) 逐元素乘法
```python
output_values = A.values * D_sampled
```

### CUDA优化
在DGL或PyTorch中，SDDMM通常用CUDA核函数优化，主要优化点

1. 并行计算：每个线程处理一个非零元素（nzz并行）
2. 共享内存：缓存$x_1$和$x_2$的常用部分，减少全局内存访问
3. 避免冗余计算：只计算必要的行列组合
   
```cpp
__global__ void sddmm_kernel(
    int *row_indices, int *col_indices, 
    float *A_values, float *X1, float *X2, 
    float *output, int nnz, int feat_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        int row = row_indices[tid];
        int col = col_indices[tid];
        float sum = 0.0f;
        for (int k = 0; k < feat_dim; k++) {
            sum += X1[row * feat_dim + k] * X2[k * X2_cols + col];
        }
        output[tid] = A_values[tid] * sum;
    }
}
```
每个线程计算一个(row, col)对应的点积，并乘以A值


For循环循环展开优化：

循环展开(Loop Unrolling)是一种通过减少循环迭代次数和分支预测开销来提升程序性能的优化技术。在CUDA核函数中，循环展开可以减少指令开销、提高指令级并行(ILP)，并更好地利用GPU的寄存器资源

- (1) 手动展开
```c++
__device__ float dot_product_unrolled4(float* X1_row, float* X2_col, int feat_dim) {
    float sum = 0.0f;
    for (int k = 0; k < feat_dim; k += 4) {
        sum += X1_row[k]     * X2_col[k];
        sum += X1_row[k + 1] * X2_col[k + 1];
        sum += X1_row[k + 2] * X2_col[k + 2];
        sum += X1_row[k + 3] * X2_col[k + 3];
    }
    return sum;
}
```

- (2) 使用 #pragma unroll指令

让编译器自动展开循环（需指定展开因子）
```c++
__device__ float dot_product_unroll_pragma(float* X1_row, float* X2_col, int feat_dim) {
    float sum = 0.0f;
    #pragma unroll 4  // 提示编译器展开4次
    for (int k = 0; k < feat_dim; k++) {
        sum += X1_row[k] * X2_col[k];
    }
    return sum;
}
```

- (3) 模板元编程
```c++
template <int UNROLL_FACTOR>
__device__ float dot_product_template(float* X1_row, float* X2_col, int feat_dim) {
    float sum = 0.0f;
    for (int k = 0; k < feat_dim; k += UNROLL_FACTOR) {
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
            sum += X1_row[k + i] * X2_col[k + i];
        }
    }
    return sum;
}
```