# hil_serl_mujoco 当前集成方案

## 1. 当前目标

当前方案已经收敛为：

- 父仓 `hil_serl_mujoco` 作为总入口和集成说明仓库
- 本地目录 `sim/`，对应远端子仓 `hil_serl_mujoco_sim`
  - 运行在本地 `MacBook M5`
  - 负责 MuJoCo 仿真、MJCF 模型、SpaceMouse、demo 录制
- 本地目录 `train/`，对应远端子仓 `hil_serl_mujoco_train`
  - 运行在 AutoDL 服务器
  - 负责 `serl_launcher`、训练脚本、actor / learner 分布式训练

当前主线任务是：

- `Franka Reach`
- `state-based RL`
- 本地仿真 + 远程 learner

## 2. 仓库拓扑

目标 GitHub 结构：

```text
hil_serl_mujoco                 # 父仓
├── sim/                        # submodule -> hil_serl_mujoco_sim
└── train/                      # submodule -> hil_serl_mujoco_train
```

当前本地工作区已经按这个目录形态整理好，但还没有真正写入 `.gitmodules`，因为那一步需要你先把两个子仓推到 GitHub。

补充说明：

- `sim/`、`train/` 是父仓里的本地目录名
- `hil_serl_mujoco_sim`、`hil_serl_mujoco_train` 是 GitHub 远端仓库名
- 当前代码里的 Python 包名仍然是 `robot_controllers_mujoco` 和 `hil_serl_train`

## 3. 当前目录职责

### 父仓 `hil_serl_mujoco`

负责：

- 总 README
- 集成方案
- 子仓关系说明
- GitHub 拆仓流程说明

不再负责：

- 直接承载 MuJoCo 环境代码
- 直接承载训练代码

### `sim/` -> `hil_serl_mujoco_sim`

负责：

- `robot_controllers_mujoco/envs`
- `robot_controllers_mujoco/robot`
- `robot_controllers_mujoco/models/franka_panda`
- `robot_controllers_mujoco/spacemouse`
- `scripts/record_demos_mujoco.py`
- `setup_env_mac.sh`

当前主模型：

- `panda_nohand_position.xml`

保留并继续维护的模型：

- `panda_nohand_effort.xml`

`panda_nohand_effort.xml` 是后续柔顺控制 / 接触控制要继续使用的正式模型，不归档、不删除。

### `train/` -> `hil_serl_mujoco_train`

负责：

- `serl_launcher`
- `scripts/train_sac_reach.py`
- `scripts/train_rlpd_mujoco.py`
- `scripts/experiments/mujoco_reach`
- `setup_env_actor_mac.sh`
- `setup_env_autodl.sh`

说明：

- `train` 仓不再维护第二份 MuJoCo 环境和模型
- `train` 侧通过安装 `robot_controllers_mujoco` 来获取环境和 MJCF

## 4. 运行架构

```text
MacBook M5
  ├── robot_controllers_mujoco
  │   ├── MuJoCo 仿真
  │   ├── FrankaReachEnv
  │   ├── SpaceMouse
  │   └── demo 录制
  └── hil_serl_mujoco_train (actor)
      └── 通过 agentlace 与远程 learner 通信

AutoDL (RTX 4090 / RTX 5090)
  └── hil_serl_mujoco_train (learner)
      ├── SAC / RLPD
      ├── replay buffer
      ├── checkpoint
      └── 参数同步
```

## 5. 环境与硬件结论

### 本地

- 设备：MacBook M5
- 作用：仿真、可视化、demo 录制、actor

### 远程

- 平台：AutoDL
- 推荐：RTX 4090 / 24GB
- 备选：RTX 5090 / 32GB

对于当前 `Franka Reach + state-based RL` 主线：

- `4090` 已经足够
- `5090` 更适合后续视觉策略、更大 batch 或更复杂实验

## 6. 训练仓为什么有两个 setup 脚本

`train/` 里保留两个脚本是因为它服务两种不同机器角色：

- `setup_env_actor_mac.sh`
  - 给本地 Mac 的 actor 用
  - 安装 CPU JAX
  - 安装 `hil_serl_mujoco_train`
  - 安装旁边的 `robot_controllers_mujoco`
  - 用于本地启动 actor

- `setup_env_autodl.sh`
  - 给远程 AutoDL learner 用
  - 安装 CUDA JAX
  - 安装 `hil_serl_mujoco_train`
  - 安装旁边的 `robot_controllers_mujoco`
  - 用于远程启动 learner

所以它们不是重复，而是分别对应：

- 本地 actor 环境
- 远程 learner 环境

## 7. GitHub 拆仓与关联流程

推荐顺序：

1. 先把 `sim/` 推成 `hil_serl_mujoco_sim`
2. 再把 `train/` 推成 `hil_serl_mujoco_train`
3. 最后回到父仓，把 `sim/` 和 `train/` 改成真正的 Git submodule

### 7.1 推送 `sim/`

在 GitHub 新建空仓库：

- `hil_serl_mujoco_sim`

本地执行：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco/sim
git init
git add .
git commit -m "Initial commit for hil_serl_mujoco_sim"
git branch -M main
git remote add origin git@github.com:<your-name>/hil_serl_mujoco_sim.git
git push -u origin main
```

### 7.2 推送 `train/`

在 GitHub 新建空仓库：

- `hil_serl_mujoco_train`

本地执行：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco/train
git init
git add .
git commit -m "Initial commit for hil_serl_mujoco_train"
git branch -M main
git remote add origin git@github.com:<your-name>/hil_serl_mujoco_train.git
git push -u origin main
```

### 7.3 更新父仓并关联 submodule

先备份当前父仓工作区，因为这一步会把本地普通目录替换成 submodule 指针：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco
mv sim /tmp/hil_serl_mujoco_sim_backup
mv train /tmp/hil_serl_mujoco_train_backup
```

然后在父仓执行：

```bash
git rm -r --cached sim train 2>/dev/null || true
rm -rf sim train

git submodule add git@github.com:<your-name>/hil_serl_mujoco_sim.git sim
git submodule add git@github.com:<your-name>/hil_serl_mujoco_train.git train

git add .gitmodules sim train
git commit -m "Add sim and train as submodules"
```

### 7.4 更新父仓远程主仓库

如果父仓还没配置远端：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco
git remote add origin git@github.com:<your-name>/hil_serl_mujoco.git
git branch -M main
git push -u origin main
```

如果父仓已经有远端，只需要：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco
git add -A
git commit -m "Restructure repo into parent + sim/train layout"
git push origin main
```

## 8. 当前执行顺序

建议你下一步按这个顺序做：

1. 检查 `sim/` 和 `train/` 的 README 是否满足你期望
2. 先分别推 `hil_serl_mujoco_sim` 和 `hil_serl_mujoco_train`
3. 再把父仓改成真正的 submodule 管理
4. 最后推送父仓 `hil_serl_mujoco`
