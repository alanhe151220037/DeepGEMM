# 混合长度 batch 的 Q mask 分流设计

日期：2026-10-16。基线：DeepGEMM v0.1.5 定制分支与本地 SGLang main。

## 目标与边界

同一 ragged prefill batch 内，按请求当前完整 KV 长度分流。KV 长度大于
`min_kv_length`（默认 256 × 1024）的请求走定制四算子路径，其余请求走原版
logits + TopK。判断不采用 CP 切分后的 Q 行数，也不按每个 Q 的 causal 长度
重新分类。已有 enabled、CUDA、sgl-kernel backend、init/local token 等适用条件保留。

保持 Q、weights、连续 KV、页表、请求顺序和最终索引坐标不变。每层仍执行一次
现有 KV gather/CP 通信。两条路径使用同一 CUDA stream 串行执行，共用
`int32[Q, 2048]` 最终输出。中间缓冲允许按全部 Q 行分配，不引入数据重排、
紧凑输出行映射、dev 的 KV split 调度或新的 32 位索引容量约束。

## Metadata 与 API

在 attention metadata 中按配置缓存路由计划。利用 CPU 请求 KV 长度和本卡请求
Q 行数构造 bool[Q] 的 long/short mask、是否存在两类请求、短路径输出宽度和
全零 logits row_starts。这些内容每次 forward 构造一次、跨层复用；不增加每层
GPU scalar 读取。已有 backward metadata 的缓存和 Q block 共用输出布局保留。

DeepGEMM 的原版 FP8 logits、backward logits、backward TopK、filter candidates、
final TopK 增加可选 `q_mask`，True 表示本次调用负责该行，None 保持原行为。
final TopK 增加可选 `out_indices`，直接写公共输出或其连续 chunk 视图。
SGLang 原版 TopK 的普通、PAGED 和 RAGGED 接口增加可选 q_mask/out；未选择行
不读取 logits、不改写 out。所有新增校验只检查形状、dtype、device、stride。

## logits 调度与流水线

mask 在 scheduler 的 next_q_block 中统一生效；所有生产者和消费者得到相同的
有效 block 序列，跳过全无效 block 时不推进 Q/KV/TMEM 流水线。
混合 block 的 KV 起止只由有效真实行决定；尾部 padding 不视为有效行。
矩阵 tile 仍整体计算，写出只允许有效行。TMEM release/barrier 不能因行屏蔽
被跳过。未传 mask 的 logits 使用编译期 specialization 保留原路径。

backward logits 继续按 request/chunk 边界划分 Q block，保持共同列起点；filter
与原版 logits 的固定四行 block 可以跨请求，必须正确处理 mixed mask。
backward TopK 对无效行写安全的 prefix 哨兵（最低分、-1），供后续阈值 reduction
读取；filter 的 counts/overflow 初始为零，无效行保持零。final TopK 跳过无效行。

## 原路径列布局与统一输出

原路径启用 v0.1.5 已有 `max_seqlen_k` 布局，宽度取短请求最大 KV 长度，最多为
配置阈值（分配时按 256 对齐）。每行写 `kv_absolute - ks[row]`，TopK 从第 0 列
读取，长度仍为当前 Q 的 causal KV 长度。PAGED 使用原页表，RAGGED 使用原 KV
偏移；未融合模式仍返回请求内索引。长请求 mask 必须在写出前生效，避免向短
路径有限宽度的缓冲写入长请求数据。

公共 topk_result 只初始化一次。短路径写短请求行，定制路径写长请求行；禁用
mask 的调用保持既有接口行为。输出不用 torch.where 合并，也不做整段复制覆盖。
全短/全长 batch 跳过空路径，避免无效 kernel launch 和大缓冲分配。

## 显存生命周期

外层只持有公共输入、metadata 和最终输出。短路径在独立函数作用域内申请
logits 并执行 TopK，返回前释放全部临时引用；随后进入长路径。定制 chunk 内
使用完的临时变量在下一 chunk 分配前释放，避免 Python 赋值右侧求值造成重叠。
不得为回收空间添加 synchronize/empty_cache。

每卡总 Q=8192、短路径宽度=262144、FP32 时 logits 为 8 GiB；公共输出为
64 MiB。存活张量峰值按公共空间加两条路径临时空间的较大者估算。
之前的 8.2734 GiB 是特定 Q/KV/候选容量下的定制缓冲估算，并非任意 batch 上限。
缓存分配器保留空间和碎片另计。

## 验证与交付

CPU：阈值等号/超一、交错长短请求、CP 本卡请求行数、缓存跨层复用、无 GPU
scalar 同步、空分组、共享输出不被另一分支覆盖、临时张量生命周期。
GPU：原版 masked compressed logits 与未 masked 原版有效区域一致；Q block
混合请求、尾部 1/2/3 行、全 false mask；两条路径共享输出与各自单独结果一致；
PAGED/RAGGED/未融合输出和显式 chunk 边界；保留现有精度和逐 kernel 性能测试。
定制路径的候选截断与阈值筛选精度性质不因分流而改变。

本机没有 CUDA 编译/执行环境，CPU 与源码检查可本地运行；GPU JIT、AOT 编译、
精度与性能必须在目标 SM100 服务器验证。部署需同步并重编译 DeepGEMM 和
SGLang 的 sgl-kernel AOT 扩展，不能只更新 Python 文件。

## 实施与服务器验证入口

DeepGEMM：两个 SM100 scheduler 在 Q block 入口读取并缓存有效行 bitmask；
共享 core 保留 TMEM release 后再屏蔽行归约/写出；FFI 增加 DLPack bool 转换；
五个算子的 Python/FFI/JIT 参数贯通，final TopK 接受公共输出。
SGLang：原版 TopK AOT 的 CUDA/ROCm 注册、C++ 声明、Python 包装共同更新；
DSA metadata 缓存路由计划，graph metadata 刷新时同时清空缓存。

在目标服务器安装两个更新后的扩展后，从 DeepGEMM 根目录运行：

```bash
python -m unittest discover -s tests -p test_long_context_mqa_migration.py -v
python tests/bench_sglang_long_context_topk.py --sglang-root /path/to/sglang --preset mixed --modes all
python tests/bench_sglang_long_context_topk.py --sglang-root /path/to/sglang --preset mixed --max-rows 8 --modes all
python tests/bench_sglang_long_context_topk.py --sglang-root /path/to/sglang --preset long --modes all
python -m pytest /path/to/sglang/python/sglang/kernels/aot/tests/test_topk.py -k q_mask_shared_output -q
```

mixed preset 使用 KV 长度 262144、262145、8192、1048576，对应 Q 行数 5、7、3、9。
阈值默认 262144，覆盖等号、超一、交错路由和非四行对齐边界；其他 preset 仍默认
阈值 0，以延续原有定制四算子的测试覆盖。`--min-kv-length` 可显式覆盖阈值。
benchmark 的 original 是禁用定制路径的基线，custom 在 mixed 场景代表新混合路径。
性能输出继续只报告每个 GPU kernel 的执行时间。

## 本地验证记录

- 34 项 CPU 回归通过：DeepGEMM metadata/wrapper 6 项、benchmark 18 项、
  SGLang metadata/分流/生命周期 10 项。
- mixed preset 的源代码契约 dry-run 通过；CUDA/ROCm 六个 TopK 注册 schema
  可解析，包含可选 mask 和可写输出声明。
- 修改范围的 Python 格式、Ruff、SGLang C++ clang-format 和 git diff 检查通过。
- DeepGEMM 的 6 项 SM100 GPU 测试因无 CUDA 设备跳过；新增 SGLang AOT GPU
  用例尚未运行。没有在本机验证 CUDA 编译、GPU 精度、耗时或实际显存峰值。
- SGLang 全仓 CI registry 检查仍报告原有四个 Mooncache 测试缺少注册，涉及
  test_mooncache_piece_writeback.py、test_mooncache_cpp_tree.py、test_mooncache_r5.py、
  test_mooncache_store_block.py。本次修改的 CPU 测试自身已注册。
