import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrajectoryPlanner:
    """从 CSV 读取轨迹，缺省时回退到一条 Panda 安全测试轨迹。"""

    def __init__(
        self,
        filepath: Optional[Path],
        duration: float,
        timestep: Optional[float],
        default_qpos: Optional[np.ndarray] = None,
    ) -> None:
        self.traj_filepath = filepath
        self.duration = duration
        self.timestep = timestep if timestep is not None else 0.001
        self.default_qpos = (
            np.asarray(default_qpos, dtype=float).copy()
            if default_qpos is not None
            else np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        )

        if self.traj_filepath is None:
            logger.info("未提供轨迹文件，使用内置 Panda 测试轨迹。")
            self._build_default_trajectory()
        else:
            self.read_trajectory()
            self.trajectory_interpolation()

    def read_trajectory(self) -> None:
        """读取单个 CSV 轨迹，并将其处理成时间连续的轨迹。"""
        if self.traj_filepath is None:
            raise ValueError("traj_filepath 为空，无法读取轨迹文件。")

        try:
            trajectory_data = pd.read_csv(self.traj_filepath)
            logger.info("轨迹文件已成功加载: %s", self.traj_filepath)
        except FileNotFoundError:
            logger.warning("轨迹文件未找到: %s，回退到内置 Panda 测试轨迹。", self.traj_filepath)
            self._build_default_trajectory()
            return

        t_raw = trajectory_data.loc[:, "time"].to_numpy()
        q_raw = trajectory_data.loc[
            :,
            [
                "joint1_pos",
                "joint2_pos",
                "joint3_pos",
                "joint4_pos",
                "joint5_pos",
                "joint6_pos",
                "joint7_pos",
            ],
        ].to_numpy()
        v_raw = trajectory_data.loc[
            :,
            [
                "joint1_vel",
                "joint2_vel",
                "joint3_vel",
                "joint4_vel",
                "joint5_vel",
                "joint6_vel",
                "joint7_vel",
            ],
        ].to_numpy()
        a_raw = trajectory_data.loc[
            :,
            [
                "joint1_accel",
                "joint2_accel",
                "joint3_accel",
                "joint4_accel",
                "joint5_accel",
                "joint6_accel",
                "joint7_accel",
            ],
        ].to_numpy()

        t_continuous = np.copy(t_raw)
        last_valid_time = 0.0
        for i in range(1, len(t_continuous)):
            if t_raw[i] < t_raw[i - 1]:
                last_valid_time = t_continuous[i - 1]
            t_continuous[i] = t_raw[i] + last_valid_time

        non_increasing_indices = np.where(np.diff(t_continuous) <= 0)[0]
        mask = np.ones(len(t_continuous), dtype=bool)
        mask[non_increasing_indices + 1] = False

        self.traj_t = t_continuous[mask]
        self.traj_q = q_raw[mask]
        self.traj_v = v_raw[mask]
        self.traj_a = a_raw[mask]

        logger.info("轨迹处理完成。检测到 %s 个分段拼接点。", len(non_increasing_indices))
        logger.info(
            "原始数据点: %s, 处理后有效数据点: %s",
            len(t_raw),
            len(self.traj_t),
        )
        logger.info("生成连续时间序列，总时长: %.4fs", self.traj_t[-1])

    def _build_default_trajectory(self) -> None:
        """构造一条适用于 Panda 的小幅度安全测试轨迹。"""
        q0 = self.default_qpos.copy()
        qf = q0 + np.array([0.0, -0.25, 0.0, -0.15, 0.0, 0.15, 0.20])

        self.simulation_times = np.arange(0.0, self.duration + self.timestep, self.timestep)
        self.q_interp = np.zeros((len(self.simulation_times), len(q0)))
        self.v_interp = np.zeros_like(self.q_interp)
        self.a_interp = np.zeros_like(self.q_interp)

        for index, sim_time in enumerate(self.simulation_times):
            q, v, a = self.plan_trajectory(sim_time, q0=q0, qf=qf, tf=self.duration)
            self.q_interp[index] = q
            self.v_interp[index] = v
            self.a_interp[index] = a

        self.traj_t = self.simulation_times
        self.traj_q = self.q_interp
        self.traj_v = self.v_interp
        self.traj_a = self.a_interp
        logger.info("内置 Panda 测试轨迹已生成。")

    def trajectory_interpolation(self) -> None:
        """对轨迹数据进行线性插值。"""
        self.simulation_times = np.arange(0, self.traj_t[-1] + self.timestep, self.timestep)

        num_joints = self.traj_q.shape[1]
        self.q_interp = np.zeros((len(self.simulation_times), num_joints))
        self.v_interp = np.zeros((len(self.simulation_times), num_joints))
        self.a_interp = np.zeros((len(self.simulation_times), num_joints))

        for i in range(num_joints):
            self.q_interp[:, i] = np.interp(self.simulation_times, self.traj_t, self.traj_q[:, i])
            self.v_interp[:, i] = np.interp(self.simulation_times, self.traj_t, self.traj_v[:, i])
            self.a_interp[:, i] = np.interp(self.simulation_times, self.traj_t, self.traj_a[:, i])

        logger.info("轨迹插值完成。")

    def get_target_qva(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """根据当前时间 t 获取目标位置、速度和加速度。"""
        index = int(round(t / self.timestep))
        if 0 <= index < len(self.q_interp):
            return self.q_interp[index], self.v_interp[index], self.a_interp[index]

        logger.warning("请求的时间 %.4fs 超出轨迹范围 [0, %.4f]s。", t, self.duration)
        last_index = len(self.q_interp) - 1
        return (
            self.q_interp[last_index],
            self.v_interp[last_index],
            self.a_interp[last_index],
        )

    def plan_trajectory(
        self,
        t: float,
        q0: np.ndarray,
        qf: np.ndarray,
        tf: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用五次多项式规划一条平滑关节轨迹。"""
        clamped_t = np.clip(t, 0.0, tf)
        normalized_t = clamped_t / tf if tf > 0 else 0.0

        qt = np.zeros_like(q0)
        qt_dot = np.zeros_like(q0)
        qt_ddot = np.zeros_like(q0)

        for i in range(len(q0)):
            delta = qf[i] - q0[i]
            qt[i] = q0[i] + delta * (
                10 * normalized_t**3 - 15 * normalized_t**4 + 6 * normalized_t**5
            )
            qt_dot[i] = delta * (
                30 * normalized_t**2 - 60 * normalized_t**3 + 30 * normalized_t**4
            ) / tf
            qt_ddot[i] = delta * (
                60 * normalized_t - 180 * normalized_t**2 + 120 * normalized_t**3
            ) / (tf**2)

        return qt, qt_dot, qt_ddot
