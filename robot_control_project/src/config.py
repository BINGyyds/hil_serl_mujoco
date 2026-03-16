# config.py

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
from pathlib import Path
import numpy as np

@dataclass
class ProjectConfig:
    """
    一个数据类，用于集中存储项目的所有配置参数。
    
    这个类的实例将在程序主入口（main.py）中被创建，可以整合命令行参数
    以实现灵活的配置管理。所有路径、仿真参数、控制增益等都应在此定义。
    """
    
    # --- 核心控制与模式配置 ---
    CONTROL_MODE: str = "torque"  # 控制模式: 'position' 或 'torque'

    # --- 文件路径配置 ---
    # 计算并存储项目根目录，以便后续所有路径都基于此构建，提高可移植性。
    # Path(__file__) -> 当前文件路径 (.../config.py)
    # .resolve()    -> 获取绝对路径
    # .parent        -> 父目录 (.../)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent, init=False)
    
    # 根据控制模式动态确定的模型文件路径
    MJCF_PATH: Path = field(init=False)
    NORMAL_URDF_PATH: Path = field(init=False)
    INVERTED_URDF_PATH: Path = field(init=False)
    TRAJECTORY_PATH: Path = field(init=False)
    
    # --- 仿真参数配置 ---
    # SIMULATION_DURATION: float = 295.212  # 仿真总时长（秒）
    SIMULATION_DURATION: float = 16  # 仿真总时长（秒）
    TIMESTEP: Optional[float] = 0.001     # 仿真物理步长 (秒), None 表示使用模型默认值

    # --- 可视化与录制 ---
    USE_VIEWER: bool = True              # 是否在运行时启动被动可视化查看器
    MAIN_VIDEO_PATH: Optional[Path] = field(init=False)  # 主视频输出路径, Optional表示可以不录制
    VIDEO_FPS: int = 30                  # 录制视频的帧率
    VIDEO_SPEED_MULTIPLIER: float = 8.0  # 视频播放倍速（相对于实时）

    # --- 数据记录配置 ---
    DATA_OUTPUT_PATH: Optional[Path] = field(init=False) # 仿真数据CSV文件输出路径
    DATA_LOGGING_INTERVAL: float = 0.01   # 数据记录的时间间隔（秒）

    # --- 机器人初始状态 ---
    INITIAL_QPOS: Optional[Tuple[float, ...]] = None # 机械臂的初始关节角度 (弧度)

    # --- 模型内元素名称配置 (与MJCF/URDF文件内容强相关) ---
    END_EFFECTOR_SITE_NAME: str = 'end_effector_site'
    ARM_JOINT_NAMES: Tuple[str, ...] = tuple(f"joint{i+1}" for i in range(7))

    # 力控模式 (torque) 下的PD反馈控制器增益
    TORQUE_KP_GAINS: Tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 20.0) # 比例增益 (Kp)
    TORQUE_KD_GAINS: Tuple[float, ...] = (20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 5.0)      # 微分增益 (Kd)

    # 位控模式 (position) 下的MuJoCo执行器增益
    POS_KP_GAINS: Tuple[float, ...] = (3000.0, 3000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0)
    POS_KV_GAINS: Tuple[float, ...] = (800.0, 800.0, 4000.0, 4000.0, 250.0, 100.0, 100.0)

    def __post_init__(self):
        """
        在dataclass对象初始化完成后，根据其他配置项的值来动态构建路径。
        这使得配置更加模块化和清晰。
        """
        # --- 根据控制模式选择MJCF文件 ---
        if self.CONTROL_MODE == 'position':
            self.MJCF_PATH = self.project_root / "model" / "mjcf" / "space_robot" / "jxb_position.xml"
        elif self.CONTROL_MODE == 'torque':
            self.MJCF_PATH = self.project_root / "model" / "mjcf" / "space_robot" / "jxb_effort.xml"
        else:
            raise ValueError(f"不支持的控制模式: '{self.CONTROL_MODE}'")

        # --- 构建其他固定路径 ---
        self.NORMAL_URDF_PATH = self.project_root / "model" / "urdf" / "jxb_normal.urdf"
        self.TRAJECTORY_PATH = self.project_root / "trajectory" / "solution_crawl.csv"
        
        # --- 构建输出文件路径 ---
        output_dir = self.project_root / "output"
        self.MAIN_VIDEO_PATH = output_dir / "videos" / "main_simulation.mp4"
        self.BASE_DATA_PATH = output_dir / "docs" / "base_data.csv"
        self.ROBOT_DATA_PATH = output_dir / "docs" / "robot_data.csv"
        