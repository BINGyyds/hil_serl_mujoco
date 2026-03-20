# HIL-SERL + Robot Control Project 集成方案

## 一、背景与目标

本仓库包含两个子项目：

| 子项目 | 核心能力 | 关键技术栈 |
|--------|----------|-----------|
| **hil-serl** | 人机回环强化学习（SAC / RLPD / HG-DAGGER / BC） | JAX/Flax, Gymnasium, agentlace |
| **robot_control_project** | Franka Panda MuJoCo 仿真、运动控制 | MuJoCo, Pinocchio, 位置/力矩控制 |

**目标**：将两者有机结合，用 MuJoCo 仿真替代真实硬件，实现：
- **Phase 1**（当前）：RL 控制机械臂末端运动到目标六维空间位姿（位置 + 姿态）
- **Phase 2**（后续）：在末端加入柔顺对接功能（力/力矩控制 + 接触检测）

---

## 二、整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       统一项目目录结构                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  configs/              训练配置                                   │
│  envs/                 Gymnasium 环境 (MuJoCo 仿真)              │
│  models/               MJCF 模型、mesh 资源                      │
│  │  └── franka_panda/  Franka Panda 模型文件                     │
│  serl_launcher/        RL 核心库 (SAC, BC, 网络, 数据)           │
│  franka_env/           真机接口 (Sim2Real 复用)                  │
│  robot/                MuJoCo 仿真控制 (底层接口, 动力学)         │
│  scripts/              训练 / 评估脚本                           │
│  │  └── experiments/   hil-serl 实验配置                         │
│  output/               生成的输出 (视频, 数据)                    │
│  trajectory/           预计算轨迹数据                             │
│  docs/                 文档                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、核心设计：MuJoCo Gymnasium 环境

### 3.1 为什么需要一个新环境？

hil-serl 的 `FrankaEnv` 通过 HTTP 与真实机器人通信，无法直接用于仿真。
我们需要一个 **MuJoCo 驱动的 Gymnasium 环境**，它：
1. 提供与 `FrankaEnv` **兼容的观测/动作接口**
2. 底层使用 MuJoCo 仿真（复用 `robot_control_project` 的 MJCF 模型）
3. 支持 MuJoCo 相机渲染（为后续视觉 RL 做准备）

### 3.2 观测空间 (Observation Space)

```python
observation_space = Dict({
    "state": Dict({
        "tcp_pose":    Box(shape=(7,)),   # xyz(3) + quaternion(4)
        "tcp_vel":     Box(shape=(6,)),   # linear_vel(3) + angular_vel(3)
        "tcp_force":   Box(shape=(3,)),   # 末端力 (Phase 2)
        "tcp_torque":  Box(shape=(3,)),   # 末端力矩 (Phase 2)
        "q_pos":       Box(shape=(7,)),   # 关节角度
        "q_vel":       Box(shape=(7,)),   # 关节速度
    }),
    "images": Dict({...}),  # MuJoCo 渲染图像 (可选)
})
```

- `tcp_pose` 与 hil-serl 的 `FrankaEnv` 保持一致（xyz + quaternion）
- 额外暴露 `q_pos`/`q_vel` 提供更丰富的本体感知
- `images` 为后续视觉 RL 预留

### 3.3 动作空间 (Action Space)

```python
action_space = Box(shape=(6,), low=-1, high=1)
# [dx, dy, dz, drx, dry, drz]
# 前 3 维：末端位置增量 (缩放后)
# 后 3 维：末端姿态增量 (旋转向量, 缩放后)
```

- 与 hil-serl 的 7D 动作一致（去掉夹爪维度，Phase 1 不需要）
- 通过 Jacobian 伪逆将笛卡尔速度转换为关节速度
- 积分得到关节位置目标 → 送入 MuJoCo 位置执行器

### 3.4 控制流程

```
RL Agent → 6D action (Cartesian delta)
    │
    ▼
Env.step():
    1. 缩放 action → Cartesian velocity (dx, dω)
    2. 获取当前 site Jacobian J (6×7)
    3. 计算关节速度: dq = J⁺ · [dx; dω]  (阻尼伪逆)
    4. 积分: q_target = q_current + dq · dt
    5. 裁剪到关节限位
    6. 写入 MuJoCo 位置执行器 data.ctrl
    7. 多步 mj_step (n_substeps)
    8. 读取新状态 → 返回 obs, reward, done, info
```

### 3.5 奖励函数 (Reach Task)

```
r = -α · ||p_current - p_target||₂           # 位置误差
    -β · arccos((tr(R_err) - 1) / 2)         # 姿态误差 (角度距离)
    + γ · 1{||Δp|| < ε_p AND θ_err < ε_r}   # 成功奖励
```

其中：
- `α, β` 控制位置/姿态的奖励权重
- `γ` 是到达目标的稀疏奖励
- `ε_p, ε_r` 是成功阈值

---

## 四、Phase 1 实现计划

### 4.1 环境层 (`mujoco_sim/envs/`)

| 文件 | 说明 |
|------|------|
| `franka_mujoco_env.py` | 基础环境：加载 MJCF、Jacobian 控制、状态读取、渲染 |
| `franka_reach_env.py` | Reach 任务：随机目标生成、奖励计算、终止条件 |

### 4.2 MJCF 增强

在现有 `panda_nohand_position.xml` 基础上创建 RL 专用场景：
- 添加目标位置可视化标记 (mocap body)
- 添加渲染相机 (腕部 / 第三人称)

### 4.3 训练脚本

- 单进程 SAC 训练（先不使用 actor-learner 分离架构）
- 使用 serl_launcher 的 SACAgent（纯状态输入，无图像）
- Wandb 日志记录

### 4.4 与 hil-serl 的兼容性

- 观测字典结构保持兼容，可复用 `SERLObsWrapper`、`ChunkingWrapper`
- 后续切换到 actor-learner 分离训练只需更改脚本
- 添加 `SpacemouseIntervention` 包装器即可实现 HIL 仿真训练

---

## 五、Phase 2 扩展方向

### 5.1 柔顺对接

- 切换到 effort (力矩) 控制模式
- 在 MJCF 中添加对接目标体和接触传感器
- 观测中加入力/力矩反馈
- 设计接触力/插入深度奖励

### 5.2 视觉 RL

- 启用 MuJoCo 渲染相机
- 使用 serl_launcher 的 ResNet encoder
- 切换到 pixel-based SAC agent

### 5.3 人机回环 (HIL)

- 接入 SpaceMouse 进行仿真环境中的人工干预
- 复用 hil-serl 的 intervention wrapper 和 demo buffer
- RLPD 训练：50/50 在线数据 + 人工演示

### 5.4 Sim2Real

- 仿真中训练好的策略，通过 Domain Randomization 迁移到真机
- 切换环境后端为 `FrankaEnv`（真机）即可
- 利用 hil-serl 的 HIL 机制在真机上微调

---

## 六、依赖关系

```
hil_serl_mujoco (集成层)
    ├── mujoco          (仿真引擎)
    ├── gymnasium        (环境接口)
    ├── numpy, scipy     (数值计算)
    ├── models/franka_panda/  (MJCF 模型资源)
    ├── agentlace        (actor-learner 分离通信)
    └── serl_launcher    (RL agent)
        ├── jax, flax    (神经网络)
        └── wandb        (日志)
```

---

## 七、快速开始

### 7.1 一键环境配置

前置要求：Conda（miniconda3 / anaconda3）+ NVIDIA GPU 驱动（CUDA 12.x）。

```bash
# 完整安装（含 SpaceMouse udev 规则，需 sudo）
bash setup_env.sh

# 远程服务器 / 无 SpaceMouse 时，跳过 udev
bash setup_env.sh --no-udev
```

脚本会自动完成：
1. 创建 `hil_serl` Conda 环境（Python 3.10）
2. 安装 JAX（CUDA 12）、MuJoCo、项目全部依赖
3. 安装 serl_launcher + agentlace
4. 配置 SpaceMouse udev 规则（可选）
5. 验证安装（JAX GPU / MuJoCo / 环境 / serl_launcher）

安装完成后激活环境：

```bash
conda activate hil_serl
```

### 7.2 手动环境配置（逐步）

如果一键脚本遇到问题，可按以下顺序手动安装：

```bash
# 1. 创建环境
conda create -n hil_serl python=3.10 -y
conda activate hil_serl

# 2. 安装 JAX GPU（必须最先安装，避免依赖冲突）
pip install --upgrade "jax[cuda12]"

# 3. 安装 MuJoCo
pip install mujoco

# 4. 安装项目依赖
pip install -r requirements_serl.txt

# 5. 安装项目本体 + serl_launcher + agentlace
pip install -e .

# 6. SpaceMouse udev 规则（Linux 必须，否则需 root 才能读取 HID）
sudo cp 99-spacemouse.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# 7. 验证
python -c "import jax; print(jax.devices())"        # 应含 GpuDevice
python -c "import mujoco; print(mujoco.__version__)" # >= 3.0.0
```

### 7.3 硬件能力说明

| 组件 | 资源需求 | i7 + RTX 4060 |
|------|---------|---------------|
| MuJoCo 仿真 (actor) | CPU 单核 ~5% | 轻松胜任 |
| JAX SAC 训练 (learner) | GPU ~100MB VRAM (state-based MLP) | 8GB VRAM 极其富余 |
| SpaceMouse HID | USB | 本地直连 |
| Replay Buffer 200K | ~1GB RAM | 无压力 |

**结论：i7 + RTX 4060 完全可以本地同时运行 actor + learner，无需远程服务器。**

---

## 八、Phase 1 训练使用方法

### 8.1 训练架构

```
┌─────────────────────────────────────────────────────────┐
│               本地机器 (i7 + RTX4060)                    │
│                                                          │
│  Terminal 1 (Actor):             Terminal 2 (Learner):   │
│  ┌───────────────────┐          ┌─────────────────────┐  │
│  │ FrankaReachEnv    │  ZMQ     │ SAC Agent           │  │
│  │ + SpaceMouse      │◄────────►│ ReplayBuffer        │  │
│  │   Intervention    │ :5588/89 │ + DemoBuffer        │  │
│  └───────────────────┘          └─────────────────────┘  │
│         ▲                              │                 │
│         │ USB                          │ GPU             │
│    SpaceMouse                      RTX 4060              │
└─────────────────────────────────────────────────────────┘
```

Actor 进程运行 MuJoCo 仿真 + SpaceMouse 干预，Learner 进程用 GPU 训练 SAC。
两者通过 agentlace (ZMQ) 在本地通信，Learner 定期将更新的网络参数推送给 Actor。

### 8.2 第 1 步：录制 SpaceMouse 演示

用 SpaceMouse 在 MuJoCo 仿真中控制机械臂，录制成功到达目标的演示数据：

```bash
python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl --num_demos 10
```

操作说明：
- MuJoCo 可视化窗口弹出后，用 SpaceMouse 控制末端 6DOF 运动
- 成功到达红色目标球后，按 **右键** 保存该回合
- 想重来按 **左键** 丢弃该回合
- 录制 5-10 个成功 demo 即可（RLPD 不需要大量 demo）

### 8.3 第 2 步：启动 Learner

```bash
python scripts/train_rlpd_mujoco.py \
    --exp_name mujoco_reach \
    --learner \
    --demo_path demo_data/reach_demos.pkl \
    --checkpoint_path output/reach_hil
```

Learner 启动后会等待 Actor 连接并填充 Replay Buffer 到 `training_starts`（默认 1000 步），然后开始 50/50 RLPD 训练。

### 8.4 第 3 步：启动 Actor（另一个终端）

```bash
python scripts/train_rlpd_mujoco.py \
    --exp_name mujoco_reach \
    --actor \
    --ip localhost
```

Actor 会：
1. 打开 MuJoCo 可视化窗口
2. 最初 `random_steps` 步（默认 1000）随机探索
3. 之后用策略网络采样动作执行
4. **随时可通过 SpaceMouse 干预** -- 推动 SpaceMouse 即替换策略动作
5. 干预数据自动存入 Demo Buffer，提升后续训练效果

### 8.5 SpaceMouse 按键说明

| 按键 | 录制模式 (`record_demos_mujoco.py`) | 训练模式 (`train_rlpd_mujoco.py`) |
|------|-------------------------------------|----------------------------------|
| **左键** | 丢弃当前回合 | 强制终止当前回合（人工判定失败） |
| **右键** | 保存当前回合为 demo | (保留，无特殊功能) |
| **推动摇杆** | 控制末端 6DOF | 人工干预，替换策略动作 |

### 8.6 Wandb 日志

训练过程中，Learner 会自动记录到 Wandb 项目 `hil-serl-mujoco`。如需关闭：

```bash
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --learner --debug \
    --demo_path demo_data/reach_demos.pkl --checkpoint_path output/reach_hil
```

`--debug` 标志会禁用 Wandb 日志。

### 8.7 从 Checkpoint 恢复训练

训练中断后，指定相同的 `--checkpoint_path` 即可自动恢复：

```bash
# Learner 会自动加载最新 checkpoint + 之前保存的 buffer 数据
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --learner \
    --demo_path demo_data/reach_demos.pkl --checkpoint_path output/reach_hil
```

### 8.8 评估已训练的策略

```bash
python scripts/train_rlpd_mujoco.py \
    --exp_name mujoco_reach \
    --actor \
    --ip localhost \
    --checkpoint_path output/reach_hil \
    --eval_checkpoint_step 50000 \
    --eval_n_trajs 20
```

---

## 九、远程服务器部署

如需将 Learner 部署到远程 GPU 服务器（本地只跑 Actor + SpaceMouse）：

### 9.1 架构

```
  本地 PC (Actor)                        远程 GPU 服务器 (Learner)
  ┌────────────────┐                     ┌──────────────────────┐
  │ SpaceMouse     │     Network         │ JAX + GPU            │
  │ MuJoCo Env     │────────────────────►│ SAC Training         │
  │ Actor Loop     │◄────────────────────│ ReplayBuffer         │
  └────────────────┘   ZMQ :5588/5589    └──────────────────────┘
```

### 9.2 网络配置

需要 TCP 端口 5588（数据上行）和 5589（参数广播）双向可达。推荐 SSH 隧道：

```bash
# 在本地执行，将远程端口转发到 localhost
ssh -L 5588:localhost:5588 -L 5589:localhost:5589 user@server
```

### 9.3 启动命令

远程服务器：

```bash
conda activate hil_serl
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --learner \
    --demo_path demo_data/reach_demos.pkl --checkpoint_path output/reach_hil
```

本地（使用 SSH 隧道时 ip 仍为 localhost）：

```bash
conda activate hil_serl
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --actor --ip localhost
```

如果不使用 SSH 隧道而是直接开放端口：

```bash
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --actor --ip <服务器IP>
```

### 9.4 远程服务器环境配置

```bash
# 远程服务器不需要 SpaceMouse，跳过 udev
bash setup_env.sh --no-udev
```

---

## 十、方案 B：Mac Actor + Linux Learner 分布式部署

### 10.1 架构概述

```
┌─────────────── Mac M5 (Actor) ───────────────┐
│  SpaceMouse ─→ MuJoCo 仿真 ─→ Actor 推理      │
│  (USB 直连)     (原生 ARM)     (JAX CPU)       │
│                                               │
│  端口: localhost:5588 ◄─── SSH 隧道 ───┐       │
│        localhost:5589 ◄─── SSH 隧道 ───┤       │
└───────────────────────────────────────┘       │
                                                │
┌─────── Linux i7+RTX4060 (Learner) ───────────┐│
│  JAX CUDA 12 ─→ SAC 梯度更新 ─→ Wandb 日志    ││
│  (GPU 加速)     (ReplayBuffer)                ││
│                                               ││
│  监听端口: 5588 (params) ◄────────────────────┘│
│           5589 (data)                          │
└────────────────────────────────────────────────┘
```

**核心思路**：Mac 负责仿真渲染 + SpaceMouse 人机交互（低延迟体验好），Linux GPU 负责策略训练（高吞吐）。两者通过 agentlace ZMQ 通信，SSH 隧道保证安全。

### 10.2 环境配置

#### Mac 端 (Actor)

```bash
# 使用 Mac 专用配置脚本
bash setup_env_mac.sh
```

脚本自动完成：
- 安装 JAX **CPU 模式**（Apple Silicon 原生，无需 CUDA/Metal）
- 安装 MuJoCo（ARM 原生，性能优秀）
- 安装 Actor 所需最小依赖集（不含 TensorFlow 等训练包）
- macOS 上 SpaceMouse **无需 udev 规则**，USB 插入即用

#### Linux 端 (Learner)

```bash
# Linux 服务器上使用 --no-udev（无 SpaceMouse）
bash setup_env.sh --no-udev
```

### 10.3 Demo 录制

Demo 在 Mac 端录制（SpaceMouse 直连 Mac），然后传到 Linux 服务器。

**Mac 上录制：**
```bash
conda activate hil_serl
python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl --num_demos 20
```

**传输到 Linux 服务器：**
```bash
scp -r demo_data/ user@server:/path/to/hil_serl_mujoco/demo_data/
```

### 10.4 训练启动

#### 第 1 步：Linux 服务器启动 Learner

```bash
conda activate hil_serl
python scripts/train_rlpd_mujoco.py \
    --exp_name mujoco_reach \
    --learner \
    --demo_path demo_data/reach_demos.pkl \
    --checkpoint_path output/reach_hil
```

#### 第 2 步：Mac 建立 SSH 隧道

```bash
# 在 Mac 终端执行，将 Learner 端口映射到本地
ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 user@server
# -N 表示不打开远程 shell，仅转发端口
```

#### 第 3 步：Mac 启动 Actor

```bash
# 另开一个 Mac 终端
conda activate hil_serl
python scripts/train_rlpd_mujoco.py \
    --exp_name mujoco_reach \
    --actor \
    --ip localhost
```

启动后即可看到 MuJoCo 仿真窗口。SpaceMouse 移动时会在仿真中看到末端执行器跟随运动，训练数据通过 SSH 隧道实时发送到 Linux Learner。

### 10.5 Mac 端技术细节

| 项目 | Mac Actor 配置 | 说明 |
|------|---------------|------|
| JAX | CPU 模式 (`pip install jax jaxlib`) | Actor 推理量小，CPU 足够 |
| MuJoCo | 原生 ARM 编译 | Apple Silicon 性能优秀 |
| SpaceMouse | USB 直连，无需额外驱动 | macOS 原生 HID 支持 |
| 渲染 | Metal 后端自动启用 | 窗口流畅无需 X11 |
| 网络 | SSH 隧道 (`-L 5588/5589`) | 安全、不需开放防火墙 |

### 10.6 常见问题 (Mac 特有)

**Q: Mac 上 JAX 要不要装 jax-metal？**
A: 不需要。Actor 端的推理计算量非常小（单次前向传播），CPU 模式延迟 < 5ms，完全满足实时性要求。jax-metal 安装可能带来兼容性问题，建议跳过。

**Q: SSH 隧道断开怎么办？**
A: Actor 会在 ZMQ 层自动重试。重新连接 SSH 隧道后 Actor 会恢复正常。建议使用 `autossh` 保持隧道稳定：
```bash
brew install autossh
autossh -M 0 -N -L 5588:localhost:5588 -L 5589:localhost:5589 user@server
```

**Q: Mac 上 SpaceMouse 无响应？**
A: 检查系统偏好设置 → 安全性与隐私 → 输入监控，确保终端 App 有权限访问 USB 设备。也可尝试重新插拔 SpaceMouse。

**Q: 训练延迟是否影响效果？**
A: Actor-Learner 架构本身就是异步的。SSH 隧道局域网延迟 < 1ms，广域网通常 < 50ms，对 RL 训练几乎无影响。参数同步是周期性的，不需要实时。

---

## 十一、常见问题

**Q: `ModuleNotFoundError: No module named 'easyhid'`**
A: SpaceMouse 驱动依赖，`pip install easyhid` 即可。如果在 Learner 端（无 SpaceMouse），此错误不影响训练 -- 只有 Actor 端需要 easyhid。

**Q: `jax.devices()` 只显示 CPU，没有 GPU**
A: 确认安装了 CUDA 12 版本的 JAX：`pip install --upgrade "jax[cuda12]"`。检查 `nvidia-smi` 确认驱动正常。

**Q: SpaceMouse 无反应 / Permission denied**
A: Linux 需要 udev 规则。执行：
```bash
sudo cp 99-spacemouse.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
然后重新插拔 SpaceMouse。

**Q: Actor 连接 Learner 超时**
A: 确认 Learner 已启动，端口 5588/5589 可达。远程服务器需要 SSH 端口转发或防火墙放行。

**Q: MuJoCo 可视化窗口黑屏**
A: 远程 SSH 连接时，需要 X11 转发（`ssh -X`）或使用 `render_mode=None` 跳过可视化。Learner 端不需要可视化。
