#!/bin/bash
################################################################################
# 脚本名称：srun_script.sh
# 脚本用途：在SLURM集群环境中启动JAX分布式训练任务，包含CUDA环境校验、分布式参数配置
# 使用步骤：
#!/bin/bash
# Step 1: salloc --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --cpus-per-task=12 --mem=175G --exclude master 
# Step 2: srun bash srun_script.sh
################################################################################

# ========================== NCCL/ CUDA 环境变量配置 ==========================
# NCCL（NVIDIA Collective Communications Library）：GPU间通信库配置
export NCCL_BUFFSIZE=4194304  # 设置NCCL通信缓冲区大小为4MiB（4*1024*1024字节），优化小数据传输性能
export NCCL_IB_QPS_PER_CONNECTION=4  # 设置每个IB连接的QPS（队列对）数量，提升InfiniBand通信性能
export NCCL_NVLS_ENABLE=0  # 禁用NVLS（NVIDIA Local Storage），避免存储相关的通信干扰
# CUDA 12.4兼容库路径：解决不同CUDA版本的库依赖问题
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/compat
# TensorFlow日志级别：仅显示ERROR和FATAL级别的日志，减少冗余输出
export TF_CPP_MIN_LOG_LEVEL=2  
# 以下两行为调试用配置，已注释以降低日志冗余
# export TF_CPP_VMODULE=asm_compiler=5  # 开启TF汇编编译器的详细日志（级别5）
# export TF_XLA_FLAGS="--tf_xla_dump_hlo_graphs"  # 导出XLA的HLO计算图，用于调试XLA编译逻辑

# ========================== 脚本执行安全配置 ==========================
set -e  # 脚本中任意命令执行失败时，立即退出脚本（避免错误累积）

# ========================== 调试信息输出 ==========================
# 打印关键环境变量，便于排查分布式训练的节点/进程问题
echo "==================================== 调试信息 ===================================="
echo "运行主机名: $(hostname)"                  # 当前执行脚本的节点名称
echo "SLURM_PROCID=$SLURM_PROCID"              # 全局进程ID（跨节点唯一）
echo "SLURM_LOCALID=$SLURM_LOCALID"            # 节点内本地进程ID（单节点内唯一）
echo "SLURM_NODEID=$SLURM_NODEID"              # 节点ID（多节点任务中标识不同节点）
echo "SLURM_NTASKS=$SLURM_NTASKS"              # 总任务数（对应申请的GPU/进程数）
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"  # 当前进程可见的GPU设备列表
echo "================================================================================="

# 无需手动映射CUDA_VISIBLE_DEVICES：SLURM会自动为每个任务分配对应的GPU
# 开启HYDRA完整错误输出：HYDRA是配置管理工具，开启后能打印完整的错误堆栈，便于调试
export HYDRA_FULL_ERROR=1

# ========================== JAX 分布式训练环境配置 ==========================
export JAX_USE_PJRT_CUDA_DEVICE=True  # JAX使用PJRT接口（新一代设备接口）管理CUDA设备，性能更优
# export NCCL_DEBUG=INFO  # 注释掉以关闭NCCL调试日志，如需排查通信问题可开启
export JAX_PROCESS_COUNT=$SLURM_NTASKS        # JAX分布式进程总数（等于SLURM总任务数）
export JAX_PROCESS_INDEX=$SLURM_PROCID        # 当前进程的全局索引（对应SLURM_PROCID）
export JAX_LOCAL_PROCESS_INDEX=$SLURM_LOCALID  # 当前进程的节点内索引（对应SLURM_LOCALID）
export JAX_NODE_RANK=$SLURM_NODEID            # 当前节点的分布式排名（对应SLURM_NODEID）
# 分布式主节点地址：从SLURM任务节点列表中取第一个节点作为主节点
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=12355  # 主节点通信端口（需确保端口未被占用）

# ========================== 函数定义：检查CUDA版本 ==========================
# 功能：校验当前环境的CUDA版本是否为12.8（训练任务依赖的版本）
# 返回值：0=版本正确，1=版本错误/检测失败
check_cuda_version() {
    # 检查nvidia-smi命令是否存在（验证NVIDIA驱动是否安装）
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: 未找到nvidia-smi命令，NVIDIA驱动可能未安装！"
        return 1
    fi
    
    # 从nvidia-smi输出中提取CUDA版本号
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "检测到的CUDA版本: $CUDA_VERSION"

    # 校验CUDA版本是否为12.4
    if [[ "$CUDA_VERSION" != "12.4" ]]; then
        echo "ERROR: CUDA版本不是12.4（检测到：$CUDA_VERSION）"
        echo "主机 $(hostname): CUDA版本不匹配"

        # 检查CUDA 12.4是否已安装但未激活
        if [ -d "/usr/local/cuda-12.4" ]; then
            echo "CUDA 12.4似乎已安装在 /usr/local/cuda-12.4"
            echo "请更新环境变量以激活CUDA 12.4："
            echo "  export PATH=/usr/local/cuda-12.4/bin:\$PATH"
            echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH"
        else
            echo "未检测到CUDA 12.4安装目录"
            echo "请安装CUDA 12.4或更新CUDA版本"
        fi
        
        return 1
    else
        echo "CUDA 12.4版本校验通过"
        return 0
    fi
}

# ========================== 主执行流程 ==========================
echo "开始校验CUDA版本..."
if check_cuda_version; then
    echo "CUDA版本校验通过，继续执行训练任务。"
else
    echo "CUDA版本校验失败，退出脚本。"
    exit 1
fi

# ========================== 启动分布式训练任务 ==========================
# XLA_PYTHON_CLIENT_MEM_FRACTION：设置XLA客户端可用的GPU内存比例（0.9=90%），避免内存溢出
# uv run：使用uv包管理器运行训练脚本，确保依赖环境正确
# scripts/train_multi_node.py：分布式训练主脚本
# debug：运行模式（调试模式）
# --exp-name=my_experiment：实验名称，用于区分不同训练任务
# --overwrite：覆盖已存在的实验结果（避免文件冲突）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python3 scripts/train_multi_node.py debug --exp-name=my_experiment --overwrite