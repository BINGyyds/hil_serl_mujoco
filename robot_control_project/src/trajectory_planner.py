# trajectory_planner.py

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class TrajectoryPlanner:
    """从Excel文件读取和管理预定义的运动轨迹。"""
    def __init__(self, filepath: str, duration: float, timestep: float) -> None:
        self.traj_filepath: str = filepath
        self.duration: float = duration
        self.timestep: float = timestep

        self.read_trajectory()
        self.trajectory_interpolation()

    def read_trajectory(self) -> None:
        """
        读取单个csv文件中纵向拼接的轨迹，并将其处理成一个时间连续的轨迹。
        """
        try:
            trajectory_data = pd.read_csv(self.traj_filepath)
            logger.info(f"轨迹文件已成功加载: {self.traj_filepath}")
        except FileNotFoundError:
            logger.error(f"轨迹文件未找到: {self.traj_filepath}")
            return

        # 1. 提取原始数据为Numpy数组
        t_raw = trajectory_data.loc[:, "time"].to_numpy()
        q_raw = trajectory_data.loc[:, ["joint1_pos", "joint2_pos", "joint3_pos","joint4_pos", 
                                       "joint5_pos", "joint6_pos","joint7_pos"]].to_numpy()
        v_raw = trajectory_data.loc[:, ["joint1_vel", "joint2_vel", "joint3_vel","joint4_vel", 
                                       "joint5_vel", "joint6_vel","joint7_vel"]].to_numpy()
        a_raw = trajectory_data.loc[:, ["joint1_accel", "joint2_accel", "joint3_accel","joint4_accel", 
                                       "joint5_accel", "joint6_accel","joint7_accel"]].to_numpy()

        # 2. 修正时间序列，使其连续
        t_continuous = np.copy(t_raw)
        last_valid_time = 0.0 # 用于记录上一段轨迹的结束时间点

        for i in range(1, len(t_continuous)):
            # 始终基于原始时间序列 t_raw 来检测时间重置点
            if t_raw[i] < t_raw[i-1]:
                # 当检测到重置时，更新偏移量。
                # 偏移量等于前一个点的、已经被修正过的连续时间。
                last_valid_time = t_continuous[i-1]
            
            # 将当前原始时间点加上正确的偏移量
            t_continuous[i] = t_raw[i] + last_valid_time
        
        # 3. 移除因拼接产生的重复时间点
        #    np.diff(t) 计算相邻元素的差值。如果差值 <= 0，说明时间没有严格增加
        #    我们找到这些非严格增加点的索引
        non_increasing_indices = np.where(np.diff(t_continuous) <= 0)[0]
        
        # 创建一个布尔掩码(mask)，初始全为True
        mask = np.ones(len(t_continuous), dtype=bool)
        # 将需要移除的点的下一个点标记为False（保留第一个，移除第二个）
        mask[non_increasing_indices + 1] = False
        
        # 4. 应用掩码，过滤掉所有重复数据点
        self.traj_t = t_continuous[mask]
        self.traj_q = q_raw[mask]
        self.traj_v = v_raw[mask]
        self.traj_a = a_raw[mask]

        logger.info(f"轨迹处理完成。检测到 {len(non_increasing_indices)} 个分段拼接点。")
        logger.info(f"原始数据点: {len(t_raw)}, 处理后有效数据点: {len(self.traj_t)}")
        logger.info(f"生成连续时间序列，总时长: {self.traj_t[-1]:.4f}s")

    def trajectory_interpolation(self) -> None:
        """对轨迹数据进行插值处理 采用线性插值"""
        # 生成仿真时间序列
        self.simulation_times = np.arange(0, self.traj_t[-1] + self.timestep, self.timestep)

        # 获取关节数量 (从traj_q数组的列数)
        num_joints = self.traj_q.shape[1]

        # 初始化用于存储所有关节插值结果的空数组
        self.q_interp = np.zeros((len(self.simulation_times), num_joints))
        self.v_interp = np.zeros((len(self.simulation_times), num_joints))
        self.a_interp = np.zeros((len(self.simulation_times), num_joints))

        # 遍历每一个关节（每一列）并分别进行插值 ---
        for i in range(num_joints):
            self.q_interp[:, i] = np.interp(self.simulation_times, self.traj_t, self.traj_q[:, i])
            self.v_interp[:, i] = np.interp(self.simulation_times, self.traj_t, self.traj_v[:, i])
            self.a_interp[:, i] = np.interp(self.simulation_times, self.traj_t, self.traj_a[:, i])
        
        logger.info("轨迹插值完成。")

    def get_target_qva(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """根据当前时间t获取对应的目标位置、速度和加速度"""
        # 1.根据时间t计算其在插值结果数组中的整数索引 round()函数可以很好地处理t中累积的微小浮点误差
        index = int(round(t / self.timestep))

        # 2.增加边界检查，防止仿真时间超出范围导致索引越界
        if 0 <= index < len(self.q_interp):
            q = self.q_interp[index]
            v = self.v_interp[index]
            a = self.a_interp[index]
            return q, v, a
        else:
            # 如果时间超出预定轨迹范围，可以选择报错或返回最后一个有效值
            logger.warning(f"请求的时间 {t:.4f}s 超出轨迹范围 [0, {self.duration}]s。")
            last_index = len(self.q_interp) - 1
            q = self.q_interp[last_index]
            v = self.v_interp[last_index]
            a = self.a_interp[last_index]
            return q, v, a

    def plan_trajectory(self, t) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """自行规划轨迹（用于测试）"""
        tf = 50.0
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qf = np.array([0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5])
        qt = np.zeros(7)
        qt_dot = np.zeros(7)
        qt_ddot = np.zeros(7)
        for i in range(7):
            qt[i] = q0[i] + (qf[i] - q0[i]) * (10 * pow(t/tf, 3) - 15 * pow(t/tf, 4) + 6 * pow(t/tf, 5))
            qt_dot[i] = (qf[i] - q0[i]) * (10 * 3/tf * pow(t/tf, 2) - 15 * 4/tf * pow(t/tf, 3) + 6 * 5/tf * pow(t/tf, 4))
            qt_ddot[i] = (qf[i] - q0[i]) * (10 * 3/tf * 2/tf * pow(t/tf, 1) - 15 * 4/tf * 3/tf * pow(t/tf, 2) + 6 * 5/tf * 4/tf * pow(t/tf, 3))
        return qt, qt_dot, qt_ddot
        
    