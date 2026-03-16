import logging
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)


class RobotModel:
    """使用 Pinocchio 库进行 Panda 运动学和动力学建模。"""

    def __init__(
        self,
        urdf_path: Path,
        end_effector_frame_name: str,
        urdf_xacro_path: Optional[Path] = None,
        xacro_mappings: Optional[Mapping[str, str]] = None,
        description_package_root: Optional[Path] = None,
    ) -> None:
        self.urdf_path = urdf_path
        materialized_urdf = self._materialize_urdf(
            urdf_path=urdf_path,
            urdf_xacro_path=urdf_xacro_path,
            xacro_mappings=xacro_mappings,
            description_package_root=description_package_root,
        )

        try:
            self.model: pin.Model = pin.buildModelFromUrdf(str(materialized_urdf))
            self.data: pin.Data = self.model.createData()

            if not self.model.existFrame(end_effector_frame_name):
                raise ValueError(f"URDF 中未找到末端坐标系: {end_effector_frame_name}")

            self.end_effector_frame_id: int = self.model.getFrameId(
                end_effector_frame_name
            )
            self.model.gravity.linear = np.zeros(3)
            logger.info("Pinocchio 动力学模型构建成功: %s", materialized_urdf)
        except Exception:
            logger.exception("从 URDF 构建 Pinocchio 模型失败。")
            raise

    def _materialize_urdf(
        self,
        urdf_path: Path,
        urdf_xacro_path: Optional[Path],
        xacro_mappings: Optional[Mapping[str, str]],
        description_package_root: Optional[Path],
    ) -> Path:
        if urdf_xacro_path is None:
            return urdf_path

        if description_package_root is None:
            raise ValueError("使用 xacro 时必须提供 description_package_root。")
        if not urdf_xacro_path.exists():
            raise FileNotFoundError(f"xacro 文件不存在: {urdf_xacro_path}")
        if not description_package_root.exists():
            raise FileNotFoundError(
                f"Franka 描述包目录不存在: {description_package_root}"
            )

        urdf_path.parent.mkdir(parents=True, exist_ok=True)
        should_regenerate = (
            not urdf_path.exists()
            or urdf_path.stat().st_mtime < urdf_xacro_path.stat().st_mtime
        )
        if not should_regenerate:
            return urdf_path

        try:
            import ament_index_python.packages as packages
            import xacro
        except ImportError as exc:
            raise RuntimeError(
                "生成 Panda URDF 需要安装 xacro 和 ament_index_python。"
            ) from exc

        original_get_package_share_directory = packages.get_package_share_directory

        def patched_get_package_share_directory(package_name: str) -> str:
            if package_name == "franka_description":
                return str(description_package_root)
            return original_get_package_share_directory(package_name)

        packages.get_package_share_directory = patched_get_package_share_directory
        try:
            doc = xacro.process_file(
                str(urdf_xacro_path), mappings=dict(xacro_mappings or {})
            )
        finally:
            packages.get_package_share_directory = original_get_package_share_directory

        urdf_text = doc.toprettyxml(indent="  ").replace(
            "package://franka_description", str(description_package_root)
        )
        urdf_path.write_text(urdf_text, encoding="utf-8")
        logger.info("已根据 xacro 生成 Panda URDF: %s", urdf_path)
        return urdf_path

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
        """执行逆动力学计算（使用 RNEA 算法）。"""
        return pin.rnea(self.model, self.data, q, v, a)
