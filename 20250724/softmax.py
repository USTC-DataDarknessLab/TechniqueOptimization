import torch
import triton
import triton.language as tl

def naive_softmax(x : torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1, keepdim=True)[0]           #(1) 每一行取最大值=
    safe_x = x - x_max                              #(2) 所有元素减去最大值，避免指数爆炸
    numerator = torch.exp(safe_x)                   #(3) 求指数
    denominator = numerator.sum(dim=1, keepdim=True)#(4) 每行求和
    softmax_out = numerator / denominator           #(5) 广播机制
    return softmax_out

def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert x.dim() == 2, f"Expected 2D input, but got {x.dim()}D input"

    block_size = triton.next_power_of_2(cols)     # next_power_of_2(5)=8

    #根据block_size动态调整 num_wraps
    num_warps = 4                # 每个wrap中有32个线程
    
    """
        #查看每个wrap中多少个线程, 这个是由GPU型号决定
        import torch
        print(torch.cuda.get_device_properties(0).warp_size)
    """

    #分段调整
    if block_size > 2047:
        num_warps = 8
    if block_size > 4095:
        num_warps = 16

    #定义网格大小，每个线程块（Block）处理一行数据
    grid = (rows,)  #这个写法会创建一个只包含rows（一个元素）的tuple

    sm_out = torch.empty_like(x)  # dim: 8 * 10

    # 调用triton内核（使用方括号传入grid，然后将参数传递给内核）
    _softmax_fwd_kernel[grid](
        sm_out,                     # output_ptr 输出张量在内存中的起始地址
        sm_out.stride(0),           # stide_output_row 输出张量在行上的步长 （即每行在内存中的间隔）10
        x,                          # input_ptr 输入张量在内存中的起始位置
        x.stride(0),                # stride_out 输入张量在行方向上的步幅
        cols,                       # 输入张量的列数
        block_size=block_size,      # block_size: 块大小，编译时常量，决定每个线程块处理的元素数量
        num_warps=num_warps         # num_warps: 线程束数目
    )
    return sm_out


    """
    #python装饰器
    def my_decorator(func):
        def wrapper():
            print("before function")
            func()
            print("after function")
        return wrapper

    @my_decorator
    def say_hello():
        print("hello")

    say_hello()

    --------------------------
    before function
    hello
    after function

    """

@triton.jit                         #python装饰器
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,       #常量表达式
):
    # 获取当前程序的ID（行索引）
    row_index = tl.program_id(0)

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0, block_size) #生成[0, 1, 2, ..., block_size - 1]
    input_pointers = row_start_ptr + col_offsets

    # 创建掩码，防止越界访问
    row_mask = col_offsets < num_cols

    # 从全局内存加载数据到片上SRAM
    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    # softmax计算
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    # 将结果写回全局内存

    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

from pathlib import Path
benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())