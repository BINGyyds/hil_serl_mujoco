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
mujoco_sim (集成层)
    ├── mujoco          (仿真引擎)
    ├── gymnasium        (环境接口)
    ├── numpy, scipy     (数值计算)
    ├── robot_control_project/model/  (MJCF 模型资源)
    └── serl_launcher    (RL agent, 可选)
        ├── jax, flax    (神经网络)
        └── wandb        (日志)
```
