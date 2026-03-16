# mujoco_interface.py

import mujoco
import mujoco.viewer
import numpy as np
import logging
from typing import Tuple, Dict, Union, Optional
from config import ProjectConfig
from mujoco import MjvCamera

logger = logging.getLogger(__name__)

class MujocoInterface:
    """
    封装与MuJoCo仿真的所有底层交互的接口类。

    该类的职责是隔离上层控制逻辑与底层的MuJoCo API调用。它负责加载模型、
    管理仿真数据（`MjModel`, `MjData`）、步进仿真、获取传感器数据、
    设置控制器输入、管理约束以及处理可视化和离线渲染。
    """
    def __init__(self, config: ProjectConfig) -> None:
        """
        初始化MuJoCo仿真环境。

        Args:
            config (ProjectConfig): 包含所有配置参数的配置对象。

        Raises:
            Exception: 如果从XML文件加载模型失败。
        """
        self.config = config
        try:
            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(config.MJCF_PATH))
            
            if config.TIMESTEP is not None:
                self.model.opt.timestep = config.TIMESTEP
            
            self.data: mujoco.MjData = mujoco.MjData(self.model)
            logger.info(f"MuJoCo模型加载成功: {config.MJCF_PATH}")
            logger.info(f"仿真时间步长 (dt) 设置为: {self.model.opt.timestep}")
            
        except Exception as e:
            logger.exception(f"加载MuJoCo模型失败: {config.MJCF_PATH}")
            raise e
            
        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.renderer: Optional[mujoco.Renderer] = None

        # 存储从模型名称到ID的映射，避免在仿真循环中重复查找，提高效率。
        self._get_ids_from_config(config)

        # 创建一个MjvCamera实例来独立于viewer管理相机状态，这对于离线渲染至关重要。
        self.camera = MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)

    def _get_ids_from_config(self, config: ProjectConfig) -> None:
        """从配置文件中读取名称，并在MuJoCo模型中查找对应的ID进行缓存。"""
        self.arm_joint_names: Tuple[str, ...] = config.ARM_JOINT_NAMES
        self.end_effector_site_id: int = self.model.site(config.END_EFFECTOR_SITE_NAME).id

    def set_initial_joint_angles(self, initial_qpos: np.ndarray) -> None:
        """
        在仿真开始前，为机械臂关节设置一个特定的初始姿态。

        Args:
            initial_qpos (np.ndarray): 包含7个关节角度的目标数组（单位：弧度）。
        """
        try:
            joint_ids = [self.model.joint(name).id for name in self.arm_joint_names]
            qpos_addrs = self.model.jnt_qposadr[joint_ids]

            if len(initial_qpos) != len(qpos_addrs):
                logger.error(f"提供的初始角度数量({len(initial_qpos)})与机械臂关节数量({len(qpos_addrs)})不匹配。")
                return
            
            self.data.qpos[qpos_addrs] = initial_qpos
            mujoco.mj_forward(self.model, self.data) # 应用更改并更新模型状态
            logger.info(f"机械臂初始位置已设置为: {initial_qpos}")
        except KeyError as e:
            logger.error(f"设置初始关节角度失败: 找不到关节 {e}")
        except Exception as e:
            logger.error(f"设置初始关节角度时发生未知错误: {e}")
    
    def get_arm_joint_angles(self) -> np.ndarray:
        """高效地读取当前机械臂所有关节的角度值 (qpos)。"""
        joint_ids = [self.model.joint(name).id for name in self.arm_joint_names]
        qpos_addrs = self.model.jnt_qposadr[joint_ids]
        return self.data.qpos[qpos_addrs].copy()

    def get_arm_joint_velocities(self) -> np.ndarray:
        """高效地读取当前机械臂所有关节的速度值 (qvel)。"""
        joint_ids = [self.model.joint(name).id for name in self.arm_joint_names]
        dof_addrs = self.model.jnt_dofadr[joint_ids]
        return self.data.qvel[dof_addrs].copy()

    def get_base_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取基座 'base_link_site' 的全局位姿。

        Returns:
            一个元组 (position, quaternion)，其中 position 是 (3,) 的位置向量，
            quaternion 是 (4,) 的姿态四元数 (w, x, y, z)。如果site不存在则返回None。
        """
        try:
            site_id = self.model.site('base_link_site').id
            base_pos = self.data.site_xpos[site_id]
            base_xmat = self.data.site_xmat[site_id].reshape(3, 3)
            base_quat = np.zeros(4)
            mujoco.mju_mat2Quat(base_quat, base_xmat.flatten())
            return base_pos.copy(), base_quat.copy()
        except KeyError:
            logger.error("在MJCF模型中未找到名为 'base_link_site' 的site。")
            return None

    def get_joint_torques(self) -> np.ndarray:
        """
        获取施加在机械臂7个关节上的力矩（来自执行器）。
        """
        # 执行器名称为 'joint1_ctrl', 'joint2_ctrl', ...
        # 更稳健的方式是获取 actuator ID
        try:
            actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'joint{i+1}_ctrl') for i in range(7)]
            if -1 in actuator_ids:
                # 如果找不到，尝试从 data.ctrl 获取 (如果没有其他执行器的话)
                return self.data.ctrl[:7].copy()
            # data.actuator_force 存储了执行器产生的力
            return self.data.actuator_force[actuator_ids]
        except ValueError:
            return self.data.ctrl[:7].copy()

    def toggle_grasp_constraint(self, active: bool) -> None:
        """激活或禁用用于模拟抓取的等式约束。"""
        self.data.eq_active[self.grasp_constraint_id] = 1 if active else 0
        logger.info(f"抓取约束已{'激活' if active else '禁用'}。")

    def toggle_climb_constraint(self, active_base_origin: bool, active_ee_climbsite: bool) -> None:
        """分别激活或禁用用于攀爬的基座和末端约束。"""
        self.data.eq_active[self.base_origin_constraint_id] = 1 if active_base_origin else 0
        logger.info(f"基座原点约束已{'激活' if active_base_origin else '禁用'}。")
        self.data.eq_active[self.ee_climbsite_constraint_id] = 1 if active_ee_climbsite else 0
        logger.info(f"末端攀爬点约束已{'激活' if active_ee_climbsite else '禁用'}。")

    def set_actuator_gains_inverted(self, inverted: bool):
        """
        根据机器人是否处于倒置攀爬状态，动态修改7个关节的位置控制执行器增益。
        
        Args:
            inverted (bool): True表示切换到倒置增益，False表示恢复正常增益。
        """
        original_kp = np.array(self.config.POS_KP_GAINS)
        original_kv = np.array(self.config.POS_KV_GAINS)

        new_kp = original_kp[::-1] if inverted else original_kp
        new_kv = original_kv[::-1] if inverted else original_kv
        
        mode = "倒置" if inverted else "正常"
        logger.info(f"MuJoCo执行器增益已切换到 -> {mode}模式。")

        for i in range(7):
            try:
                actuator_id = self.model.actuator(f"joint{i+1}_ctrl").id
                self.model.actuator_gainprm[actuator_id, 0] = new_kp[i] # Kp
                self.model.actuator_gainprm[actuator_id, 1] = new_kv[i] # Kv
            except KeyError:
                logger.warning(f"查找执行器 'joint{i+1}_ctrl' 失败，无法设置增益。")

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器site的当前世界坐标。"""
        return self.data.site_xpos[self.end_effector_site_id]

    def get_object_position(self) -> np.ndarray:
        """获取目标物体site的当前世界坐标。"""
        return self.data.site_xpos[self.object_body_site_id]
        
    def set_control(self, ctrl: np.ndarray) -> None:
        """设置所有执行器的控制信号（力矩或位置）。"""
        self.data.ctrl[:] = ctrl

    def check_proximity(self) -> bool:
        """检查末端执行器与目标物体的距离是否小于阈值。"""
        dist = np.linalg.norm(self.get_end_effector_position() - self.get_object_position())
        return dist < self.proximity_threshold

    def step(self) -> None:
        """执行一个仿真步进。"""
        mujoco.mj_step(self.model, self.data)

    def launch_viewer(self) -> None:
        """如果尚未启动，则启动被动式查看器。"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def sync_viewer(self) -> None:
        """与被动查看器同步，更新显示画面。"""
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def close_viewer(self) -> None:
        """关闭查看器窗口。"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def setup_renderer(self, width: int = 1280, height: int = 720) -> None:
        """初始化用于离线渲染的渲染器。"""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height, width)
            logger.info(f"离线渲染器已初始化，分辨率: {width}x{height}")

    def set_camera_view(self, azimuth: float, elevation: float, distance: float, lookat: Optional[np.ndarray] = None) -> None:
        """
        更新内部 `MjvCamera` 实例的视角参数，用于离线渲染。
        
        Args:
            azimuth (float): 方位角 (度)。
            elevation (float): 仰角 (度)。
            distance (float): 相机到目标的距离。
            lookat (Optional[np.ndarray]): 相机观察点 [x, y, z]。
        """
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = distance
        if lookat is not None:
            self.camera.lookat[:] = lookat
        logger.debug(f"相机视角已设置: Azimuth={azimuth}, Elevation={elevation}, Distance={distance}")

    def render_frame(self, camera: Union[str, mujoco.MjvCamera]) -> np.ndarray:
        """
        使用离线渲染器从指定的相机渲染一帧画面。

        Args:
            camera: 相机标识符。可以是相机名称（字符串）或一个 `MjvCamera` 实例。

        Returns:
            一个 (H, W, 3) 的NumPy数组，格式为RGB。
        """
        if self.renderer is None:
            self.setup_renderer()
        
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()
    
    def render_all_custom_frames(self, camera_names: Tuple[str, ...]) -> Dict[str, np.ndarray]:
        """从所有指定的相机渲染画面，并返回一个字典。"""
        frames = {}
        if self.renderer is None:
            self.setup_renderer()

        for cam_name in camera_names:
            try:
                frames[cam_name] = self.render_frame(camera=cam_name)
            except mujoco.FatalError:
                 # ... (之前版本中的错误日志记录逻辑可以保留) ...
                 pass
        return frames