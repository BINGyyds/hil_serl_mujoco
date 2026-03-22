#!/usr/bin/env bash
# =============================================================================
#  HIL-SERL MuJoCo -- 一键环境配置脚本
#
#  用法:
#    bash setup_env.sh            # 完整安装（含 SpaceMouse udev 规则）
#    bash setup_env.sh --no-udev  # 跳过 udev 规则（无 SpaceMouse 或远程服务器）
#
#  前置要求:
#    - Conda (miniconda3 / anaconda3)
#    - NVIDIA GPU + CUDA 12.x 驱动
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 ------------------------------------------
ENV_NAME="hil_serl"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_UDEV=false

for arg in "$@"; do
    case "$arg" in
        --no-udev) SKIP_UDEV=true ;;
    esac
done

# ----------------------------- 辅助函数 --------------------------------------
info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

# ----------------------------- 1. 检测 Conda ---------------------------------
info "检测 Conda ..."
if command -v conda &>/dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
else
    error "未找到 Conda，请先安装 miniconda3: https://docs.conda.io/en/latest/miniconda.html"
fi
info "Conda 路径: $CONDA_BIN"

# 初始化 shell 集成（确保 conda activate 可用）
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

# 确认 Python 版本
PYVER=$(python --version 2>&1)
info "Python: $PYVER"

# ----------------------------- 3. 安装 JAX (CUDA 12) -------------------------
info "安装 JAX (CUDA 12) ... 这可能需要几分钟"
pip install --upgrade "jax[cuda12]"

# ----------------------------- 4. 安装 MuJoCo --------------------------------
info "安装 MuJoCo ..."
pip install mujoco

# ----------------------------- 5. 安装项目依赖 --------------------------------
info "安装 requirements_serl.txt ..."
pip install -r "$PROJECT_DIR/requirements_serl.txt"

# ----------------------------- 6. 安装项目 ------------------------------------
info "安装项目 ..."
pip install -e "$PROJECT_DIR" --config-settings editable_mode=compat 2>/dev/null \
    || pip install -e "$PROJECT_DIR"

# ----------------------------- 6.1 安装 agentlace (HIL 通信, 可选) ------------
info "安装 agentlace (HIL 分布式通信) ..."
if pip install "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d" 2>/dev/null; then
    info "agentlace 安装成功"
else
    warn "agentlace 安装失败 (网络问题)，跳过。"
    warn "如需 actor-learner 分布式训练，请稍后手动安装:"
    warn "  pip install agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d"
fi

# ----------------------------- 7. 安装 Stable-Baselines3 (可选) ---------------
info "安装 stable-baselines3 (可选，用于快速验证) ..."
pip install stable-baselines3 || warn "stable-baselines3 安装失败，非必须依赖"

# ----------------------------- 8. SpaceMouse udev 规则 ------------------------
if [ "$SKIP_UDEV" = false ]; then
    RULES_FILE="$PROJECT_DIR/99-spacemouse.rules"
    if [ -f "$RULES_FILE" ]; then
        info "配置 SpaceMouse udev 规则 (需要 sudo 权限) ..."
        sudo cp "$RULES_FILE" /etc/udev/rules.d/
        sudo udevadm control --reload-rules && sudo udevadm trigger
        info "SpaceMouse udev 规则已安装"
    else
        warn "未找到 99-spacemouse.rules，跳过 SpaceMouse udev 配置"
    fi
else
    info "跳过 SpaceMouse udev 规则 (--no-udev)"
fi

# ----------------------------- 9. 验证安装 ------------------------------------
info "=========================================="
info "验证安装结果 ..."
info "=========================================="

echo ""
info "[1/4] JAX 设备:"
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
info "[4/4] serl_launcher 导入:"
python -c "from serl_launcher.agents.continuous.sac import SACAgent; print('  SACAgent 导入成功')" 2>&1 || warn "serl_launcher 导入失败"

# ----------------------------- 完成 -------------------------------------------
echo ""
info "=========================================="
info "环境配置完成!"
info "=========================================="
echo ""
info "使用方法:"
info "  conda activate $ENV_NAME"
info "  cd $PROJECT_DIR"
echo ""
info "录制 demo:"
info "  python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl"
echo ""
info "启动训练 (两个终端):"
info "  # Terminal 1 - Learner:"
info "  python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --learner \\"
info "      --demo_path demo_data/reach_demos.pkl --checkpoint_path output/reach_hil"
info "  # Terminal 2 - Actor:"
info "  python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --actor --ip localhost"
echo ""
