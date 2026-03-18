# robot_control_project（MuJoCo机械臂仿真控制例程）

**测试系统版本：ubuntu24.04**

**运行环境版本：python3.10**

## 项目文件结构

```
robot_control_project/
├── model
│	├── mjcf
│	│   ├── jxb_position.xml        # 位控机械臂mjcf文件
│	│   ├── jxb_effort.xml          # 力控机械臂mjcf文件
│	│   └── ...
│	└── urdf
│	    ├── jxb_normal.urdf     # 机械臂正常urdf文件
│	    └── ...
├── output
│	├── docs
│	│   ├── base_data.csv       # 仿真过程轨迹信息文件输出：基座偏移
│	│   └── robot_data.csv      # 仿真过程轨迹信息文件输出：机械臂运动
│	├── pictures
│	│   └── ...                 # 仿真过程结果图片文件输出
│	└── videos
│	    └── main_simulation.mp4 # 仿真过程结果视频文件输出
├── src
│	├── main.py                 # 项目主入口，负责解析命令行参数和启动控制器
│	├── config.py               # 定义项目配置的数据类结构 (ProjectConfig)
│	├── robot_controller.py     # 高层控制器，协调所有模块，执行主控制循环
│	├── mujoco_interface.py     # 封装与MuJoCo的所有底层交互（如读写数据、约束控制）
│	├── robot_model.py          # 封装Pinocchio动力学模型，负责运动学/动力学计算
│	├── trajectory_planner.py   # 负责从csv文件读取和管理运动轨迹
│	├── data_manager.py         # 负责仿真数据的记录、存储和可视化
│	└── async_writer.py         # 负责视频的多线程异步输出
├── trajectory
│	├── solution_crawl.csv        # 轨迹序列文件
│	└── ...
├── auto_script.sh          	# 自动化环境部署脚本
└── README.md               	# 项目说明文档
```

## 环境部署

**1. 赋予脚本执行权限**

在终端中，首先给 `auto_script.sh` 脚本添加可执行权限：

```bash
chmod +x auto_script.sh
```

**2. 运行部署脚本**

执行脚本来创建虚拟环境并安装所有依赖项：
```bash
./auto_script.sh
```
该脚本将创建一个名为 `robotics_env` 的虚拟环境，并自动安装所有必需的库。

## 依赖库列表

### 通用工具和数据处理库
- `pandas`: 用于读取csv轨迹文件。
- `matplotlib`: 用于数据可视化和绘图。

### 机器人仿真相关库
- `mujoco`: 物理仿真引擎。
- `pinocchio`: 机器人动力学计算库。
- `opencv`: 用于仿真视频录制

## 运行使用

1.激活环境

```bash
conda activate robotics_env
```

2.运行项目: 
```bash
python src/main.py
```