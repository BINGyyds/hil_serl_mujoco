#!/usr/bin/env bash
# =============================================================================
#  HIL-SERL MuJoCo -- Mac Actor 端环境配置脚本
#
#  适用于方案 B：Mac M5 运行 Actor（MuJoCo 仿真 + SpaceMouse 干预），
#  远程 Linux GPU 服务器运行 Learner（JAX CUDA 训练）。
#
#  用法:
#    bash setup_env_mac.sh
#
#  前置要求:
#    - macOS (Apple Silicon M1/M2/M3/M4/M5)
#    - Conda (miniconda3 / anaconda3)
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 ------------------------------------------
ENV_NAME="hil_serl"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- 辅助函数 --------------------------------------
info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

# ----------------------------- 检测平台 --------------------------------------
if [[ "$(uname)" != "Darwin" ]]; then
    error "此脚本仅适用于 macOS。Linux 请使用 setup_env.sh"
fi
ARCH="$(uname -m)"
info "检测到 macOS ($ARCH)"

# ----------------------------- 1. 检测 Conda ---------------------------------
info "检测 Conda ..."
if command -v conda &>/dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]; then
    CONDA_BIN="/opt/homebrew/Caskroom/miniconda/base/bin/conda"
else
    error "未找到 Conda，请先安装:\n  brew install miniconda\n  或: https://docs.conda.io/en/latest/miniconda.html"
fi
info "Conda 路径: $CONDA_BIN"

eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" || eval "$($CONDA_BIN shell.zsh hook 2>/dev/null)" || true

# ----------------------------- 2. 创建 Conda 环境 ----------------------------
if conda env list | grep -qw "$ENV_NAME"; then
    warn "Conda 环境 '$ENV_NAME' 已存在，跳过创建"
else
    info "创建 Conda 环境: $ENV_NAME (Python $PYTHON_VERSION) ..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

info "激活环境: $ENV_NAME"
conda activate "$ENV_NAME"

PYVER=$(python --version 2>&1)
info "Python: $PYVER"

# ----------------------------- 3. 安装 JAX (CPU) -----------------------------
# Mac Actor 端只需要 JAX CPU 模式（用于 agent.sample_actions）
# 不需要 CUDA，不需要 jax-metal
info "安装 JAX (CPU 模式) ..."
pip install jax jaxlib

# ----------------------------- 4. 安装 MuJoCo --------------------------------
info "安装 MuJoCo ..."
pip install mujoco

# ----------------------------- 5. 安装核心依赖 --------------------------------
# Actor 端只需要部分依赖，跳过 tensorflow 等重量级训练包
info "安装 Actor 端核心依赖 ..."
pip install \
    gymnasium==0.29.1 \
    "numpy>=1.24.3" \
    "scipy>=1.11.0" \
    "flax>=0.8.0" \
    "chex>=0.1.85" \
    "optax>=0.1.5" \
    "distrax>=0.1.2" \
    "absl-py>=0.12.0" \
    "tqdm>=4.60.0" \
    natsort \
    easyhid \
    pynput \
    matplotlib \
    imageio \
    opencv-python

# ----------------------------- 6. 安装项目本体 --------------------------------
info "安装项目 ..."
pip install -e "$PROJECT_DIR" --config-settings editable_mode=compat 2>/dev/null \
    || pip install -e "$PROJECT_DIR"

# ----------------------------- 6.1 安装 agentlace (HIL 通信, 可选) ------------
# agentlace 只在 actor-learner 分布式训练时需要
# 如果你只做本地 MuJoCo 仿真训练 (SB3/单进程)，可以跳过
info "安装 agentlace (HIL 分布式通信) ..."
if pip install "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d" 2>/dev/null; then
    info "agentlace 安装成功"
else
    warn "agentlace 安装失败 (网络问题)，跳过。"
    warn "这不影响 MuJoCo 本地仿真训练 (SB3/单进程 SAC)。"
    warn "如需 actor-learner 分布式训练，请稍后手动安装:"
    warn "  pip install agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d"
fi

# ----------------------------- 7. 验证安装 ------------------------------------
info "=========================================="
info "验证安装结果 ..."
info "=========================================="

echo ""
info "[1/4] JAX (CPU 模式):"
python -c "import jax; devs = jax.devices(); print('  ', devs)" 2>&1 || warn "JAX 导入失败"

echo ""
info "[2/4] MuJoCo 版本:"
python -c "import mujoco; print('  mujoco', mujoco.__version__)" 2>&1 || warn "MuJoCo 导入失败"

echo ""
info "[3/4] MuJoCo 环境测试:"
python -c "
import sys; sys.path.insert(0, '$PROJECT_DIR')
from envs.franka_reach_env import FrankaReachEnv
from configs.reach_config import make_reach_config
env = FrankaReachEnv(config=make_reach_config())
obs, _ = env.reset()
print('  obs state keys:', list(obs['state'].keys()))
print('  action space:', env.action_space)
env.close()
print('  环境测试通过')
" 2>&1 || warn "MuJoCo 环境测试失败"

echo ""
info "[4/4] SpaceMouse:"
python -c "from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert; print('  SpaceMouse 模块导入成功')" 2>&1 \
    || warn "SpaceMouse 模块导入失败 (如果未连接 SpaceMouse 可忽略)"

# ----------------------------- 完成 -------------------------------------------
echo ""
info "=========================================="
info "Mac Actor 端环境配置完成!"
info "=========================================="
echo ""
info "注意: macOS 上 SpaceMouse 不需要 udev 规则，直接 USB 连接即可使用。"
echo ""
info "使用方法:"
info "  conda activate $ENV_NAME"
info "  cd $PROJECT_DIR"
echo ""
info "录制 demo (Mac 本地):"
info "  python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl"
echo ""
info "启动训练:"
info "  1. 先在 Linux GPU 服务器上启动 Learner:"
info "     python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --learner \\"
info "         --demo_path demo_data/reach_demos.pkl --checkpoint_path output/reach_hil"
info ""
info "  2. 建立 SSH 隧道 (Mac 上执行):"
info "     ssh -L 5588:localhost:5588 -L 5589:localhost:5589 user@server"
info ""
info "  3. Mac 上启动 Actor:"
info "     python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --actor --ip localhost"
echo ""
