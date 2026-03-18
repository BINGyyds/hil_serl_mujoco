#!/bin/bash

# 使用 Conda 为机器人控制框架自动部署环境
# 功能:
# 1. 创建一个独立的 Conda 环境。
# 2. 从 conda-forge 频道安装所有必需的依赖库。

# 设置 Conda 环境的名称
ENV_NAME="robotics_env"
PYTHON_VERSION="3.10"  # 您可以指定一个Python版本

echo "--- 开始使用 Conda 部署环境 ---"

# --- 步骤 1: 检查 Conda 是否安装 ---
if ! command -v conda &> /dev/null
then
    echo "错误：未找到 Conda。请先安装 Miniconda 或 Anaconda。"
    echo "访问: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# --- 步骤 2: 检查并创建 Conda 环境 ---
echo "[1/4] 检查并创建 Conda 环境: $ENV_NAME" # 步骤总数调整
if conda env list | grep -q "\<$ENV_NAME\>"; then # 使用 \<\> 确保匹配完整环境名称
    echo "Conda 环境 '$ENV_NAME' 已存在，跳过创建步骤。"
else
    echo "正在创建 Conda 环境，Python 版本为 $PYTHON_VERSION ..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    if [ $? -ne 0 ]; then
        echo "错误：创建 Conda 环境失败。"
        exit 1
    fi
fi

# --- 步骤 3: 分批安装核心 Conda 依赖库 ---

# 第一批：通用工具和数据处理库
echo "[2/4] 正在从 conda-forge 频道安装第一批核心依赖库 (pandas, matplotlib)..."
conda install -n $ENV_NAME -c conda-forge pandas matplotlib -y
if [ $? -ne 0 ]; then
    echo "错误：第一批依赖库安装失败。"
    exit 1
fi
echo "第一批 Conda 依赖库已成功安装。"

# 第二批：机器人和视觉相关库
echo "[3/4] 正在从 conda-forge 频道安装第二批核心依赖库 (pinocchio, mujoco, opencv)..."
conda install -n $ENV_NAME -c conda-forge pinocchio mujoco opencv -y
if [ $? -ne 0 ]; then
    echo "错误：第二批依赖库安装失败。"
    exit 1
fi
echo "第二批 Conda 依赖库已成功安装。"


# --- 步骤 5: 完成与提示 ---
echo "[4/4] 检查已安装的库..." # 步骤总数调整
conda run -n $ENV_NAME pip list

echo ""
echo "--- Conda 环境部署成功！---"
echo "您可以通过以下命令激活环境并开始工作："
echo "conda activate $ENV_NAME"
echo ""