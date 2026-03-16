from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


PANDA_HOME_QPOS: Tuple[float, ...] = (
    0.0,
    0.0,
    0.0,
    -1.57079,
    0.0,
    1.57079,
    -0.7853,
)


@dataclass
class ProjectConfig:
    """集中管理 Panda 仿真与后续真机迁移所需的配置。"""

    CONTROL_MODE: str = "position"
    USE_GRIPPER: bool = True

    SIMULATION_DURATION: float = 16.0
    TIMESTEP: Optional[float] = 0.001

    USE_VIEWER: bool = True
    VIDEO_FPS: int = 30
    VIDEO_SPEED_MULTIPLIER: float = 8.0

    DATA_LOGGING_INTERVAL: float = 0.01
    INITIAL_QPOS: Tuple[float, ...] = PANDA_HOME_QPOS

    MUJOCO_ARM_JOINT_NAMES: Tuple[str, ...] = tuple(f"joint{i + 1}" for i in range(7))
    MUJOCO_ARM_ACTUATOR_NAMES: Tuple[str, ...] = tuple(
        f"actuator{i + 1}" for i in range(7)
    )
    PINOCCHIO_ARM_JOINT_NAMES: Tuple[str, ...] = tuple(
        f"panda_joint{i}" for i in range(1, 8)
    )

    MUJOCO_BASE_SITE_NAME: str = "panda_base_site"
    MUJOCO_END_EFFECTOR_SITE_NAME: str = field(init=False)
    PINOCCHIO_END_EFFECTOR_FRAME_NAME: str = field(init=False)
    MUJOCO_GRIPPER_ACTUATOR_NAME: Optional[str] = field(init=False)
    MUJOCO_GRIPPER_OPEN_CTRL: float = 255.0
    MUJOCO_GRIPPER_CLOSE_CTRL: float = 0.0

    TORQUE_KP_GAINS: Tuple[float, ...] = (
        100.0,
        100.0,
        100.0,
        100.0,
        50.0,
        50.0,
        20.0,
    )
    TORQUE_KD_GAINS: Tuple[float, ...] = (20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 5.0)

    POS_KP_GAINS: Tuple[float, ...] = (
        4500.0,
        4500.0,
        3500.0,
        3500.0,
        2000.0,
        2000.0,
        2000.0,
    )
    POS_KV_GAINS: Tuple[float, ...] = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        init=False,
    )
    workspace_root: Path = field(init=False)
    model_root: Path = field(init=False)
    output_root: Path = field(init=False)
    franka_description_root: Path = field(init=False)

    MJCF_PATH: Path = field(init=False)
    NORMAL_URDF_PATH: Path = field(init=False)
    URDF_XACRO_PATH: Path = field(init=False)
    TRAJECTORY_PATH: Optional[Path] = field(init=False)

    MAIN_VIDEO_PATH: Path = field(init=False)
    BASE_DATA_PATH: Path = field(init=False)
    ROBOT_DATA_PATH: Path = field(init=False)

    URDF_XACRO_MAPPINGS: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.workspace_root = self.project_root.parents[1]
        self.model_root = self.project_root / "model"
        self.output_root = self.project_root / "output"
        self.franka_description_root = (
            self.workspace_root / "franka_ros" / "franka_description"
        )

        if self.CONTROL_MODE != "position":
            raise ValueError(
                "旧版 Panda MJCF 当前仅配置了位置型执行器，请使用 CONTROL_MODE='position'。"
            )

        franka_mjcf_root = self.model_root / "mjcf" / "franka_panda"
        self.MJCF_PATH = (
            franka_mjcf_root / "scene.xml"
            if self.USE_GRIPPER
            else franka_mjcf_root / "panda_nohand.xml"
        )

        self.URDF_XACRO_PATH = (
            self.franka_description_root / "robots" / "panda" / "panda.urdf.xacro"
        )
        self.NORMAL_URDF_PATH = (
            self.model_root / "urdf" / "panda_hand.urdf"
            if self.USE_GRIPPER
            else self.model_root / "urdf" / "panda_nohand.urdf"
        )
        self.TRAJECTORY_PATH = None

        if self.USE_GRIPPER:
            self.MUJOCO_END_EFFECTOR_SITE_NAME = "panda_hand_tcp"
            self.PINOCCHIO_END_EFFECTOR_FRAME_NAME = "panda_hand_tcp"
            self.MUJOCO_GRIPPER_ACTUATOR_NAME = "actuator8"
        else:
            self.MUJOCO_END_EFFECTOR_SITE_NAME = "panda_link8_tcp"
            self.PINOCCHIO_END_EFFECTOR_FRAME_NAME = "panda_link8"
            self.MUJOCO_GRIPPER_ACTUATOR_NAME = None

        self.URDF_XACRO_MAPPINGS = {
            "arm_id": "panda",
            "hand": str(self.USE_GRIPPER).lower(),
        }

        self.MAIN_VIDEO_PATH = self.output_root / "videos" / "main_simulation.mp4"
        self.BASE_DATA_PATH = self.output_root / "docs" / "base_data.csv"
        self.ROBOT_DATA_PATH = self.output_root / "docs" / "robot_data.csv"
