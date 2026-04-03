# hil_serl_mujoco

当前主线已经收敛为一条明确流程：

1. 本地 `sim/` 跑 MuJoCo、`panda_nohand_velocity.xml`、SpaceMouse 和 demo 录制
2. 先录一小批 insertion demos，作为 learner 的初始 `demo_buffer`
3. 远端 `train/` 跑 learner
4. 本地 `train/` 跑 actor，连接远端 learner
5. 在线训练时继续用 SpaceMouse 做 intervention

这份 README 现在是父仓唯一保留的顶层集成文档。旧的 `INTEGRATION_PLAN.md` 和 `MUJOCO_INSERTION_DEPLOYMENT.md` 已收敛到这里。

## 仓库结构

```text
hil_serl_mujoco/
├── README.md
├── sim/
└── train/
```

- `sim/`
  - 本地 MuJoCo 仿真、Franka Panda MJCF、SpaceMouse、demo 录制、本地控制调试
- `train/`
  - `serl_launcher`、`train_rlpd_mujoco.py`、actor / learner、buffer、checkpoint

## 当前主线

当前主线任务为：

- `MuJoCo insertion`
- `state-based HIL-RLPD`
- 本地 actor + 远端 learner

需要固定的一条约束是：

- 现在以及后续所有涉及 SpaceMouse 控制的 MuJoCo 场景，都以 `panda_nohand_velocity.xml` 为基准

## 运行拓扑

推荐拓扑：

- 本地机器
  - `sim/`
  - `train/`
  - SpaceMouse
  - MuJoCo viewer
  - 运行 `actor`
- 远端服务器
  - `sim/`
  - `train/`
  - GPU
  - 运行 `learner`

当前通信端口来自 [`launcher.py`](/home/lrb/Workspace/hil_serl_mujoco/train/serl_launcher/utils/launcher.py#L301)：

- trainer port: `5588`
- broadcast port: `5589`

推荐网络方式：

- 首选：Tailscale / Zerotier / 内网 VPN
- 备选：可直连的公网 IP
- 不建议额外改代码做新 RPC

## 本地机器准备

本地机器建议准备两个 Conda 环境：

- `sim_env`
  - 跑 MuJoCo 仿真
  - 录 demos
  - 调试 `robot_controller`
- `hil_serl_actor`
  - 跑 `train/scripts/train_rlpd_mujoco.py --actor`

### 1. 安装 sim 环境

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/sim
./setup_env_mac.sh
```

或 Ubuntu：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/sim
./setup_env_ubuntu.sh
```

安装完成后：

```bash
conda activate sim_env
python -c "from robot_controllers_mujoco.configs.insertion_config import make_insertion_config; from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv; env = FrankaInsertionEnv(config=make_insertion_config()); env.reset(); env.close(); print('sim insertion env ok')"
```

### 2. 安装本地 actor 环境

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_actor_mac.sh
```

如果本地 actor 机器是 Ubuntu：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_actor_ubuntu.sh
```

安装完成后：

```bash
conda activate hil_serl_actor
python -c "import jax; print(jax.devices())"
```

## 远端服务器准备

远端服务器也需要同时放 `sim/` 和 `train/`，因为 `train` 直接依赖 `robot_controllers_mujoco`。

建议目录：

```text
/root/workspace/hil_serl_mujoco/
├── sim/
└── train/
```

安装远端 learner 环境：

```bash
cd /root/workspace/hil_serl_mujoco/train
./setup_env_autodl.sh
```

安装完成后：

```bash
conda activate hil_serl_train
python -c "import jax; print(jax.devices())"
```

如果这里看不到 GPU，先不要启动 learner。

## 网络连通性检查

假设你使用 Tailscale，先拿到远端服务器 VPN IP，例如：

```text
100.101.102.103
```

本地先测试：

```bash
ping 100.101.102.103
```

如果是公网或云服务器安全组，还需要开放：

- `5588/tcp`
- `5589/tcp`

## 先录 insertion demos

这一步只在本地 `sim_env` 里完成，不需要 learner 在线。

```bash
conda activate sim_env
cd /home/lrb/Workspace/hil_serl_mujoco/sim

python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25
```

按钮语义：

- `SpaceMouse 6D`：控制 tool tip
- `右键`：保存当前成功轨迹
- `左键`：丢弃当前轨迹并 reset

建议第一轮先录：

- 20 到 50 条成功轨迹
- 每条都从预对齐初始位姿开始
- 失败轨迹不要放进初始 `demo_buffer`

如果你想看录制过程：

```bash
python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25 \
  --render-human \
  --realtime
```

## 把 demos 传到远端

如果 learner 在远端，把 `pkl` 拷过去：

```bash
scp /home/lrb/Workspace/hil_serl_mujoco/sim/demo_data/insertion_demos.pkl \
  user@100.101.102.103:/root/workspace/hil_serl_mujoco/train/demo_data/
```

如果远端 `demo_data/` 不存在，先创建：

```bash
ssh user@100.101.102.103 "mkdir -p /root/workspace/hil_serl_mujoco/train/demo_data"
```

## 启动远端 learner

```bash
conda activate hil_serl_train
cd /root/workspace/hil_serl_mujoco/train

mkdir -p output/mujoco_insertion

python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --learner \
  --demo_path /root/workspace/hil_serl_mujoco/train/demo_data/insertion_demos.pkl \
  --checkpoint_path /root/workspace/hil_serl_mujoco/train/output/mujoco_insertion
```

启动后应看到类似信息：

- `demo buffer size: ...`
- `starting learner loop`

## 启动本地 actor

```bash
conda activate hil_serl_actor
cd /home/lrb/Workspace/hil_serl_mujoco/train

python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --ip 100.101.102.103 \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_actor
```

说明：

- `--ip` 写远端 learner 地址
- 本地 actor 会打开 MuJoCo viewer
- SpaceMouse intervention 通过 [`wrappers.py`](/home/lrb/Workspace/hil_serl_mujoco/sim/robot_controllers_mujoco/envs/wrappers.py#L15) 接入
- learner 会把最新参数回推给 actor

## 运行时数据流

- `demo_buffer`
  - learner 启动时由 `--demo_path` 预热
  - 在线 intervention 时继续追加高质量片段
- `replay_buffer`
  - actor 的全量在线 rollout 数据

当前主线是：

1. 少量 demo 预热
2. 在线 HIL-RLPD 持续训练

不是先单独做完整 BC，再切到 RL。

## 推荐操作顺序

第一次完整跑通时，建议严格按这个顺序：

1. 本地确认 insertion env 能 reset
2. 本地录 `25` 条 insertion demos
3. 把 demo `pkl` 传到远端
4. 远端启动 learner
5. 确认 learner 打印 `demo buffer size`
6. 本地启动 actor，连接远端 IP
7. 先观察几轮纯策略行为
8. 再用 SpaceMouse 做 intervention

## 常见问题

### 1. learner 启动了，但 actor 连不上

先排查：

- 远端 IP 是否正确
- `5588/5589` 是否开放
- Tailscale 是否在线
- 本地和远端是否用了兼容版本的 `sim/` 和 `train/`

### 2. actor 可以跑，但没有收到新参数

先看 learner 是否已经开始训练：

- replay buffer 是否超过 `training_starts`
- learner 是否报错退出
- actor 是否真的连到了正确的 learner IP

### 3. learner 报 `robot_controllers_mujoco` 找不到

说明远端没有安装或找不到 `sim/`：

```bash
cd /root/workspace/hil_serl_mujoco/train
./setup_env_autodl.sh
```

### 4. 想先本机自测，不走远端

终端 1：

```bash
conda activate hil_serl_train
cd /home/lrb/Workspace/hil_serl_mujoco/train
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --learner \
  --demo_path /home/lrb/Workspace/hil_serl_mujoco/sim/demo_data/insertion_demos.pkl \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local
```

终端 2：

```bash
conda activate hil_serl_actor
cd /home/lrb/Workspace/hil_serl_mujoco/train
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --ip localhost \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local_actor
```
