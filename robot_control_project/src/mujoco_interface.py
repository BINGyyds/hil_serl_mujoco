import logging
from typing import Dict, Optional, Tuple, Union

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjvCamera

from config import ProjectConfig

logger = logging.getLogger(__name__)


class MujocoInterface:
    """封装 Panda MuJoCo 仿真的底层交互接口。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        try:
            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(config.MJCF_PATH))
            if config.TIMESTEP is not None:
                self.model.opt.timestep = config.TIMESTEP

            self.data: mujoco.MjData = mujoco.MjData(self.model)
            logger.info("MuJoCo 模型加载成功: %s", config.MJCF_PATH)
            logger.info("仿真时间步长 (dt) 设置为: %s", self.model.opt.timestep)
        except Exception as exc:
            logger.exception("加载 MuJoCo 模型失败: %s", config.MJCF_PATH)
            raise exc

        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.renderer: Optional[mujoco.Renderer] = None

        self._get_ids_from_config(config)

        self.camera = MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)

    def _get_ids_from_config(self, config: ProjectConfig) -> None:
        """根据配置缓存 joint、actuator 和 site 的 ID。"""
        self.arm_joint_names = config.MUJOCO_ARM_JOINT_NAMES
        self.arm_joint_ids = np.array(
            [self.model.joint(name).id for name in self.arm_joint_names], dtype=np.int32
        )
        self.arm_qpos_addrs = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dof_addrs = self.model.jnt_dofadr[self.arm_joint_ids]

        self.arm_actuator_names = config.MUJOCO_ARM_ACTUATOR_NAMES
        self.arm_actuator_ids = np.array(
            [self.model.actuator(name).id for name in self.arm_actuator_names],
            dtype=np.int32,
        )

        self.base_site_id = (
            self.model.site(config.MUJOCO_BASE_SITE_NAME).id
            if config.MUJOCO_BASE_SITE_NAME
            else None
        )
        self.end_effector_site_id = self.model.site(
            config.MUJOCO_END_EFFECTOR_SITE_NAME
        ).id

        self.gripper_actuator_id = (
            self.model.actuator(config.MUJOCO_GRIPPER_ACTUATOR_NAME).id
            if config.MUJOCO_GRIPPER_ACTUATOR_NAME
            else None
        )
        self.gripper_ctrl = config.MUJOCO_GRIPPER_OPEN_CTRL

    def set_initial_joint_angles(self, initial_qpos: np.ndarray) -> None:
        """为 Panda 7 个关节设置初始姿态，并同步位置控制目标。"""
        if len(initial_qpos) != len(self.arm_qpos_addrs):
            logger.error(
                "提供的初始角度数量(%s)与机械臂关节数量(%s)不匹配。",
                len(initial_qpos),
                len(self.arm_qpos_addrs),
            )
            return

        self.data.qpos[self.arm_qpos_addrs] = initial_qpos
        self.data.ctrl[self.arm_actuator_ids] = initial_qpos
        if self.gripper_actuator_id is not None:
            self.data.ctrl[self.gripper_actuator_id] = self.gripper_ctrl
        mujoco.mj_forward(self.model, self.data)
        logger.info("机械臂初始位置已设置为: %s", initial_qpos)

    def get_arm_joint_angles(self) -> np.ndarray:
        """读取机械臂当前关节角度。"""
        return self.data.qpos[self.arm_qpos_addrs].copy()

    def get_arm_joint_velocities(self) -> np.ndarray:
        """读取机械臂当前关节速度。"""
        return self.data.qvel[self.arm_dof_addrs].copy()

    def get_base_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取 Panda 基座 site 的全局位姿。"""
        if self.base_site_id is None:
            return None

        base_pos = self.data.site_xpos[self.base_site_id]
        base_xmat = self.data.site_xmat[self.base_site_id].reshape(3, 3)
        base_quat = np.zeros(4)
        mujoco.mju_mat2Quat(base_quat, base_xmat.flatten())
        return base_pos.copy(), base_quat.copy()

    def get_joint_torques(self) -> np.ndarray:
        """读取 Panda 7 个关节执行器当前输出。"""
        return self.data.actuator_force[self.arm_actuator_ids].copy()

    def set_actuator_gains_inverted(self, inverted: bool) -> None:
        """
        根据倒置状态切换位置执行器增益。

        Panda MJCF 使用 general actuator，需同时更新 gainprm 与 biasprm。
        """
        original_kp = np.array(self.config.POS_KP_GAINS)
        original_kv = np.array(self.config.POS_KV_GAINS)

        new_kp = original_kp[::-1] if inverted else original_kp
        new_kv = original_kv[::-1] if inverted else original_kv
        mode = "倒置" if inverted else "正常"
        logger.info("MuJoCo 执行器增益已切换到 -> %s 模式。", mode)

        for index, actuator_id in enumerate(self.arm_actuator_ids):
            self.model.actuator_gainprm[actuator_id, 0] = new_kp[index]
            self.model.actuator_biasprm[actuator_id, 1] = -new_kp[index]
            self.model.actuator_biasprm[actuator_id, 2] = -new_kv[index]

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器 site 的当前世界坐标。"""
        return self.data.site_xpos[self.end_effector_site_id].copy()

    def set_control(self, ctrl: np.ndarray) -> None:
        """设置 Panda 7 个关节的控制目标。"""
        control = np.asarray(ctrl, dtype=np.float64)
        if control.shape != (len(self.arm_actuator_ids),):
            raise ValueError(
                f"控制输入维度错误，期望 {(len(self.arm_actuator_ids),)}，实际 {control.shape}"
            )

        self.data.ctrl[self.arm_actuator_ids] = control
        if self.gripper_actuator_id is not None:
            self.data.ctrl[self.gripper_actuator_id] = self.gripper_ctrl

    def set_gripper_control(self, close: bool) -> None:
        """设置 Panda 手爪开合目标。"""
        if self.gripper_actuator_id is None:
            return
        self.gripper_ctrl = (
            self.config.MUJOCO_GRIPPER_CLOSE_CTRL
            if close
            else self.config.MUJOCO_GRIPPER_OPEN_CTRL
        )
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_ctrl

    def step(self) -> None:
        """执行一步物理仿真。"""
        mujoco.mj_step(self.model, self.data)

    def launch_viewer(self) -> None:
        """启动被动式查看器。"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def sync_viewer(self) -> None:
        """与被动查看器同步。"""
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def close_viewer(self) -> None:
        """关闭查看器窗口。"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def setup_renderer(self, width: int = 1280, height: int = 720) -> None:
        """初始化离线渲染器。"""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height, width)
            logger.info("离线渲染器已初始化，分辨率: %sx%s", width, height)

    def set_camera_view(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat: Optional[np.ndarray] = None,
    ) -> None:
        """更新离线渲染相机视角。"""
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = distance
        if lookat is not None:
            self.camera.lookat[:] = lookat

    def render_frame(self, camera: Union[str, mujoco.MjvCamera]) -> np.ndarray:
        """从指定相机渲染一帧 RGB 图像。"""
        if self.renderer is None:
            self.setup_renderer()

        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def render_all_custom_frames(self, camera_names: Tuple[str, ...]) -> Dict[str, np.ndarray]:
        """从多个命名相机渲染图像。"""
        frames: Dict[str, np.ndarray] = {}
        if self.renderer is None:
            self.setup_renderer()

        for cam_name in camera_names:
            try:
                frames[cam_name] = self.render_frame(camera=cam_name)
            except mujoco.FatalError:
                logger.warning("相机 '%s' 渲染失败，已跳过。", cam_name)
        return frames
