# hil_serl_mujoco

`hil_serl_mujoco` 是一个面向 MuJoCo 仿真插入任务的 Human-in-the-Loop 强化学习工作区。当前主线已经收敛为单一任务和单一部署流程：

- 任务：Franka Panda peg-in-hole insertion
- 仿真控制基线：`panda_nohand_velocity.xml`
- 数据策略：少量人工 demo 预热 + 在线 HIL-RLPD
- 部署拓扑：本地 actor + 远端 learner

## 仓库结构

```text
hil_serl_mujoco/
├── README.md
├── sim/
└── train/
```

- `sim/`
  - MuJoCo 场景与 MJCF 模型
  - `robot_controllers_mujoco` Python 包
  - SpaceMouse 接入、本地控制器、demo 录制
- `train/`
  - `serl_launcher`
  - `scripts/train_rlpd_mujoco.py`
  - actor / learner 分布式训练
  - replay buffer、demo buffer、checkpoint
  - 本地 metrics 曲线、checkpoint eval、MP4 导出

## 当前工作流

当前推荐工作流如下：

1. 在本地 `sim/` 中验证 insertion 环境与 SpaceMouse 控制链。
2. 录制一批成功的 insertion demos，作为 learner 的初始 `demo_buffer`。
3. 在远端服务器启动 learner。
4. 在本地启动 actor，连接远端 learner。
5. 在线训练期间继续使用 SpaceMouse 进行 intervention。
6. 从 checkpoint 做纯策略 eval，并按需导出 MP4 结果视频。

这个流程对应的是“demo-prefill + online HIL-RLPD”，不是“先完整做一轮离线 BC，再切换到 RL”。

## 关键约束

当前主线有一条固定约束：

- 现在以及后续所有涉及 SpaceMouse 控制的 MuJoCo 场景，都以 `panda_nohand_velocity.xml` 为基准。

这条约束同时适用于：

- `sim/scripts/debug/teleop_main.py`
- `sim/scripts/record_demos_mujoco.py`
- `sim/robot_controllers_mujoco/envs/FrankaInsertionEnv`

## 推荐部署拓扑

推荐使用“本地操作机 + 远端训练服务器”的双机部署：

- 本地机器
  - 安装 `sim/`
  - 安装 `train/` 的 actor 环境
  - 连接 SpaceMouse
  - 打开 MuJoCo viewer
  - 运行 actor
- 远端服务器
  - 安装 `sim/`
  - 安装 `train/` 的 learner 环境
  - 使用 GPU
  - 运行 learner

当前 actor / learner 通信配置来自 [`launcher.py`](/home/lrb/Workspace/hil_serl_mujoco/train/serl_launcher/utils/launcher.py)：

- trainer port: `5588`
- broadcast port: `5589`

推荐网络方式：

- 首选：Tailscale、Zerotier 或其他内网 VPN
- 备选：可直连的公网 IP

## 环境准备

### 本地 sim 环境

Ubuntu：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/sim
./setup_env_ubuntu.sh
conda activate sim_env
```

### 本地 actor 环境

Ubuntu：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_actor_ubuntu.sh
conda activate hil_serl_actor
```

### 远端 learner 环境

```bash
cd /root/workspace/hil_serl_mujoco/train
./setup_env_autodl.sh
conda activate hil_serl_train
```

远端 learner 环境当前固定为：

- Python 3.11
- JAX / JAXLIB / CUDA plugin `0.6.2`
- `tensorflow-probability==0.25.0`
- `distrax==0.1.5`
- `flax==0.10.7`
- `optax==0.2.8`

`hil_serl_train` 默认采用 JAX bundled CUDA/cuDNN，并在激活环境时自动过滤外部 `/usr/local/cuda*` 路径，避免系统 toolkit 与 learner 环境发生动态库冲突。

## 端到端启动顺序

### 1. 录制 insertion demos

在本地 `sim_env` 中执行：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/sim
conda activate sim_env

python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 30 \
  --render-human \
  --realtime
```

默认录制配置已经和当前训练主线对齐：

- 控制频率 `100 Hz`
- `sim_timestep=0.001`
- `n_substeps=10`
- 默认关闭 reset 随机扰动，使录制起点与 debug teleop 的 home 位姿一致

如需恢复训练用随机初始位姿，可额外加 `--random-reset`。

建议初始录制规模：

- 20 到 50 条成功轨迹
- 每条都从预对齐初始位姿开始
- 初始 `demo_buffer` 中只保留成功轨迹

### 2. 将 demo 文件传到远端

```bash
ssh user@100.101.102.103 "mkdir -p /root/workspace/hil_serl_mujoco/train/demo_data"

scp /home/lrb/Workspace/hil_serl_mujoco/sim/demo_data/insertion_demos.pkl \
  user@100.101.102.103:/root/workspace/hil_serl_mujoco/train/demo_data/
```

### 3. 启动远端 learner

```bash
cd /root/workspace/hil_serl_mujoco/train
conda activate hil_serl_train
mkdir -p output/mujoco_insertion

python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --learner \
  --demo_path /root/workspace/hil_serl_mujoco/train/demo_data/insertion_demos.pkl \
  --checkpoint_path /root/workspace/hil_serl_mujoco/train/output/mujoco_insertion
```

正常情况下，learner 应打印：

- `demo buffer size: ...`
- `starting learner loop`

### 4. 启动本地 actor

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
conda activate hil_serl_actor

python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --ip 100.101.102.103 \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_actor
```

默认情况下 actor 会打开 MuJoCo viewer。需要无窗口采样时可加 `--norender_actor`；Ubuntu actor 环境还可选 `--actor_viewer meshcat` 启动轻量网页 viewer。
actor 默认会在达到自身 `max_steps` 后继续采样，直到 learner 回报训练完成，避免 learner 后半段只用冻结 replay buffer 继续训练；如需恢复旧的固定 actor 步数行为，可加 `--noactor_run_until_learner_done`。

actor 启动后会：

- 打开 MuJoCo 环境
- 从 learner 接收最新策略参数
- 持续向 replay buffer 写入在线 rollout
- 在有人机干预时额外写入 intervention 数据

### 5. 数据流说明

- `demo_buffer`
  - learner 启动时由 `--demo_path` 预热
  - 后续可继续写入高质量 intervention 数据
- `replay_buffer`
  - 由 actor 的在线交互持续填充

learner 会先等待 `replay_buffer` 达到 `training_starts`，然后才开始正式更新。

### 6. 训练后纯策略评估与视频导出

训练完成后，可以直接从 checkpoint 做纯策略 eval：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
conda activate hil_serl_actor

python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local \
  --eval_checkpoint_step -1 \
  --eval_n_trajs 10
```

- `--eval_checkpoint_step -1` 表示加载 `checkpoint_path` 下最新 checkpoint
- eval 会自动关闭 SpaceMouse intervention，只执行策略本身

如需导出 MP4：

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local \
  --eval_checkpoint_step -1 \
  --eval_n_trajs 3 \
  --actor_viewer none \
  --eval_video_path /home/lrb/Workspace/hil_serl_mujoco/train/output/eval/mujoco_insertion.mp4
```

当前 eval 视频默认使用世界系跟踪相机 `tool_follow_camera`。如需切回固定前视角，可手动加 `--eval_video_camera front`。

### 7. 本地训练曲线

learner 训练过程中会把 episode 指标持续写入：

```text
train/output/.../metrics/episode_metrics.csv
```

训练正常结束时会自动生成本地 PNG 曲线；也可以手动运行：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
conda activate hil_serl_train

python scripts/plot_mujoco_metrics.py \
  --metrics_dir /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local/metrics
```

默认输出：

- `reward_curve.png`
- `success_curve.png`
- `error_curve.png`

## 快速自测

### 本机仅验证 learner

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
conda activate hil_serl_train

python -c "import jax; print(jax.devices())"
python -c "import jax, jax.numpy as jnp; x=jnp.ones((2048,2048), dtype=jnp.float32); y=(x@x).block_until_ready(); print(jax.devices(), y[0,0])"
python -c "import distrax; from tensorflow_probability.substrates import jax as tfp; print('ok')"
```

### 单机联调

终端 1：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
conda activate hil_serl_train

python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --learner \
  --demo_path /home/lrb/Workspace/hil_serl_mujoco/sim/demo_data/insertion_demos.pkl \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local
```

终端 2：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
conda activate hil_serl_actor

python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --ip localhost \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local_actor \
  --actor_viewer meshcat \
  --actor_render_fps 30
```

如果希望单机联调时使用 Pinocchio + MeshCat 作为 actor viewer，推荐在 actor 终端使用上面的 `--actor_viewer meshcat`。如需回退到原始 MuJoCo viewer，只需去掉这两个参数。
actor 默认会跟随 learner 生命周期运行：即使 actor 先达到 `max_steps`，只要 learner 还没完成，它会继续采样并写入 replay buffer。

## 常见问题

### learner 能启动，但一直停在 `Filling up replay buffer`

这通常不是错误，而是 learner 正在等待 actor 将在线数据写入 `replay_buffer`。保持 learner 运行，再启动 actor 即可。

### learner 提示找不到 `robot_controllers_mujoco`

说明 learner 机器上的 `sim/` 尚未正确安装。请重新执行：

```bash
cd /root/workspace/hil_serl_mujoco/train
./setup_env_autodl.sh
```

### 提示 `No module named 'jax'` 或 `scripts.experiments`

先确认当前 shell 不在 `base` 环境，并且命令从 `train/` 根目录启动：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
conda activate hil_serl_train  # learner
# 或 conda activate hil_serl_actor  # actor
```

### 服务器上担心驱动 / CUDA 版本冲突

当前 learner 主线不依赖系统 `/usr/local/cuda*`，而是使用 JAX bundled CUDA/cuDNN。真正需要保证的是：

- `nvidia-smi` 正常
- NVIDIA GPU 可用
- 激活 `hil_serl_train` 后，`LD_LIBRARY_PATH` 中不再残留 `/usr/local/cuda*`

更详细的仿真侧和训练侧说明分别见：

- [`sim/README.md`](/home/lrb/Workspace/hil_serl_mujoco/sim/README.md)
- [`train/README.md`](/home/lrb/Workspace/hil_serl_mujoco/train/README.md)
