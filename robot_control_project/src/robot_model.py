# robot_model.py

import pinocchio as pin
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class RobotModel:
    """使用Pinocchio库进行机器人运动学和动力学建模。"""
    def __init__(self, urdf_path: str, end_effector_frame_name: str) -> None:
        try:
            self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
            self.data: pin.Data = self.model.createData()
            self.end_effector_frame_id: int = self.model.getFrameId(end_effector_frame_name)
            self.model.gravity.linear = np.array([0., 0., 0.])  #忽略重力影响
            logger.info("Pinocchio 动力学模型构建成功。")
        except Exception:
            logger.exception("从 URDF 构建 Pinocchio 模型失败。")
            raise

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行正向运动学计算。"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose: pin.SE3 = self.data.oMf[self.end_effector_frame_id]
        return pose.translation, pose.rotation

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """计算末端执行器框架的雅可比矩阵。"""
        pin.computeFrameJacobian(self.model, self.data, q, self.end_effector_frame_id)
        return self.data.J

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """执行逆动力学计算（使用RNEA算法）。"""
        return pin.rnea(self.model, self.data, q, v, a)
    
    