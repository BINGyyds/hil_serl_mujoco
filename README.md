# hil_serl_mujoco

`hil_serl_mujoco` 现在作为父仓使用，负责保留总说明和两个子仓的关系定义。当前工作区已经按未来 GitHub submodule 的形态拆成了两个目录：

- `sim/` 对应远端子仓 `hil_serl_mujoco_sim`
- `train/` 对应远端子仓 `hil_serl_mujoco_train`

当前目录已经是 `submodule-ready` 布局，但还没有真正写入 `.gitmodules`。推荐流程是：

1. 先把 `sim/` 单独推成 `hil_serl_mujoco_sim`
2. 再把 `train/` 单独推成 `hil_serl_mujoco_train`
3. 最后回到父仓，把这两个目录替换成真实 submodule

说明：

- `sim/`、`train/` 是父仓里的本地挂载目录名
- `hil_serl_mujoco_sim`、`hil_serl_mujoco_train` 是 GitHub 远端仓库名
- 当前代码里的 Python 包名暂时仍然分别是 `robot_controllers_mujoco` 和 `hil_serl_train`

## 仓库职责

### `sim/`
- 本地 MuJoCo 仿真环境
- Franka Panda MJCF/mesh 资产
- SpaceMouse 接入
- Demo 录制与本地控制调试

### `train/`
- `serl_launcher`
- MuJoCo Reach 主线训练脚本
- actor / learner 分布式训练入口
- 本地 actor 与 AutoDL learner 的安装脚本

## 当前主线

当前默认主线是：

1. 本地 `MacBook M5` 跑 MuJoCo 仿真和 actor
2. 远程 `AutoDL` 租用 `RTX 4090 / 24GB` 或 `RTX 5090 / 32GB` 跑 learner
3. 任务主线先聚焦 `Franka Reach + state-based RL`

对当前这条主线，`4090` 已经足够。`5090` 更适合后续视觉策略、更大 batch 或更重的实验并发。

## 目录概览

```text
hil_serl_mujoco/
├── README.md
├── INTEGRATION_PLAN.md
├── sim/
│   ├── README.md
│   ├── pyproject.toml
│   ├── setup_env_mac.sh
│   ├── scripts/
│   └── robot_controllers_mujoco/
├── train/
│   ├── README.md
│   ├── pyproject.toml
│   ├── setup_env_actor_mac.sh
│   ├── setup_env_autodl.sh
│   ├── scripts/
│   └── serl_launcher/
```

## 快速开始

### 1. 本地仿真

```bash
cd sim
bash setup_env_mac.sh
conda activate robot_controllers_mujoco
python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl
```

### 2. 本地 actor

```bash
cd train
bash setup_env_actor_mac.sh
conda activate hil_serl_actor
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --actor --ip localhost
```

### 3. AutoDL learner

```bash
cd train
bash setup_env_autodl.sh
conda activate hil_serl_train
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --learner \
  --demo_path demo_data/reach_demos.pkl \
  --checkpoint_path output/reach_hil
```

### 4. SSH 端口转发

```bash
ssh -L 5588:localhost:5588 -L 5589:localhost:5589 user@your-autodl-host
```

## GitHub 建仓与关联

### 1. 创建并推送 `sim/`

```bash
cd sim
git init
git add .
git commit -m "Initial commit for hil_serl_mujoco_sim"
git branch -M main
git remote add origin git@github.com:<your-name>/hil_serl_mujoco_sim.git
git push -u origin main
```

### 2. 创建并推送 `train/`

```bash
cd train
git init
git add .
git commit -m "Initial commit for hil_serl_mujoco_train"
git branch -M main
git remote add origin git@github.com:<your-name>/hil_serl_mujoco_train.git
git push -u origin main
```

### 3. 回父仓改成 submodule

先把当前目录备份到别处，然后在父仓执行：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco
mv sim /tmp/hil_serl_mujoco_sim_backup
mv train /tmp/hil_serl_mujoco_train_backup
git submodule add git@github.com:<your-name>/hil_serl_mujoco_sim.git sim
git submodule add git@github.com:<your-name>/hil_serl_mujoco_train.git train
git add .gitmodules sim train
git commit -m "Add sim and train as submodules"
```

### 4. 更新父仓远程主仓库

如果父仓还没有配置远端：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco
git remote add origin git@github.com:<your-name>/hil_serl_mujoco.git
git branch -M main
git push -u origin main
```

如果父仓已经配置好了远端：

```bash
cd /Users/lirongbing/Workspace/hil_serl_mujoco
git add -A
git commit -m "Restructure repo into parent + sim/train layout"
git push origin main
```

更完整版本见 [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md)
