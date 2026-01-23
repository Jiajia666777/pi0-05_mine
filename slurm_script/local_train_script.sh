#!/bin/bash
################################################################################
# 脚本名称：local_train_single_process.sh
# 脚本用途：A6000 单进程多GPU训练（无端口冲突、无torchrun、适配新版JAX）
# 使用步骤：
# Step 1: chmod +x local_train_single_process.sh
# Step 2: bash local_train_single_process.sh
################################################################################

# ========================== A6000 硬件配置（仅需改这里）=========================
export LOCAL_GPU_IDS="5,6"        # 你的A6000 GPU ID（nvidia-smi查看）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # 48GB显存预留15%（避免OOM）
export CUDA_VISIBLE_DEVICES=$LOCAL_GPU_IDS

# ========================== A6000 专属优化（新版JAX兼容）=========================
# Ampere SM 8.0 架构适配
export NCCL_ARCH=80
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export CUDA_DEVICE_MAX_CONNECTIONS=16

# 新版JAX配置（核心兼容）
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=False
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16
export JAX_ENABLE_BFLOAT16=True

# CUDA 12.4 路径配置
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/compat:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH

# ========================== 训练脚本路径（改这里为你的实际路径）=========================
TRAIN_SCRIPT_PATH="/home/ubuntu/data1/hjt/pi0-05_mine/scripts/train_multi_node.py"

# ========================== 环境检查 ==========================
set -e
echo "=== 启动A6000单进程多GPU训练 ==="
echo "GPU ID: $LOCAL_GPU_IDS"
echo "CUDA版本: $(nvidia-smi | grep -oP 'CUDA Version: \K\d+\.\d+')"
echo "训练脚本路径: $TRAIN_SCRIPT_PATH"
echo "JAX版本: $(python3 -c 'import jax; print(jax.__version__)')"

# 检查脚本是否存在
if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    echo "ERROR: 训练脚本不存在！路径：$TRAIN_SCRIPT_PATH"
    exit 1
fi

# 清理残留进程（避免显存占用）
echo "清理残留Python进程..."
pkill -f train_multi_node.py > /dev/null 2>&1 || true

# ========================== 启动单进程多GPU训练 ==========================
python3 $TRAIN_SCRIPT_PATH debug --exp-name=a6000_single_process --overwrite