export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_CLIENT_MEM_FRACTION=0.95

export JAX_ENABLE_X64=1
export JAX_TRACEBACK_FILTERING=off

export HIP_VISIBLE_DEVICES=0