# rm -rf tests/test_mqa_backward_logits_topk.py
# rm -rf csrc/jit_kernels/impls/smxx_mqa_backward_topk.hpp
# rm -rf csrc/jit_kernels/impls/smxx_mqa_final_topk.hpp
# rm -rf deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits_checksum.cuh
# rm -rf deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits_no_epilogue.cuh
# git reset --hard
# git apply $1

export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=10.3
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="/usr/local/cuda/include/cccl${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

MAX_JOBS=32 bash build_sgl_deep_gemm.sh
pip uninstall sgl-deep-gemm
pip install dist/sgl_deep_gemm-0.1.2-py3-none-any.whl
rm -rf /root/.deep_gemm/cache/kernel.sm*