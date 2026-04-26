# IK 与导纳控制重构建议

本文档记录当前 MuJoCo Franka 插入任务控制链的评估结论与后续重构建议。本文档只作为方案记录，不代表已经实施代码修改。

- [IK 与导纳控制重构建议](#ik-与导纳控制重构建议)
  - [当前 IK 问题](#当前-ik-问题)
  - [IK 迁移评估](#ik-迁移评估)
  - [推荐 IK 设计](#推荐-ik-设计)
  - [SO(3) 本体导纳评估](#so3-本体导纳评估)
  - [推荐导纳控制设计](#推荐导纳控制设计)
  - [训练后仿真过快问题评估](#训练后仿真过快问题评估)
  - [推荐评估与实时播放设计](#推荐评估与实时播放设计)
  - [推荐实施顺序](#推荐实施顺序)
  - [风险与测试](#风险与测试)

## 当前 IK 问题

当前 IK 路径存在一些工程问题，会直接影响 reset 质量、轨迹规划质量和控制稳定性：

- 没有零空间目标。Panda 是 7 自由度机械臂，而任务是 6D 末端位姿控制，当前求解器没有主动利用冗余自由度来维持更舒适的肘部姿态。
- 关节限位只做 clip。更新后再裁剪容易让解卡在关节边界，也可能在边界附近产生震荡。
- 初值依赖强。初始 `q` 不合适时，即使目标可达，也可能收敛到不理想的分支，或者直接失败。
- 位置误差和姿态误差缺少灵活权重。很难明确调节 reset 时到底更优先 socket 对齐、插入轴姿态，还是整体 6D 位姿精度。
- 阻尼固定。单一阻尼参数很难同时适配正常构型、近奇异构型和近关节限位构型。
- 没有 line search 或 trust-region 机制。某一步更新过大时，可能反而增大残差并破坏迭代稳定性。
- 没有 multi-seed 策略。当前求解器可能错过更好的解分支，尤其是在 7 自由度冗余机械臂上。

## IK 迁移评估

`iiwa-RNEA-program/iiwa14_rne_dynamics/ik_solver.py` 里的 IK 方法值得迁移。它的核心思路比当前简单 DLS 循环更完整：

- 将 IK 表述为非线性最小二乘问题。
- residual 由位置误差、四元数姿态误差和正则化项共同组成。
- 正则化项会把冗余机械臂拉向期望舒适姿态，相当于利用了零空间自由度。
- 关节限位通过 optimizer bounds 表达，而不是更新后简单 clip。
- 姿态误差通过 `radius` 参数缩放，便于调节位置精度和姿态精度之间的权重。
- 使用 MuJoCo analytic site Jacobian 和四元数误差导数逻辑，而不是有限差分。

但是，这份代码不能直接复制到当前仓库。

当前 `sim` 环境固定使用 `mujoco==3.6.0`，而当前 MuJoCo Python 包没有暴露 `mujoco.minimize` 接口。因此，参考 IK 代码里的求解 backend 无法原样运行。迁移时应保留它的 residual 和 analytic Jacobian 设计，但把 backend 替换为 `scipy.optimize.least_squares`。SciPy 已经是当前项目依赖，不需要额外引入新的优化库。

## 推荐 IK 设计

建议实现一套共享的 MuJoCo site IK solver，供 env reset、planned-mode 目标位姿求解，以及后续离线目标位姿求解共同使用。

核心设计如下：

- 使用 `scipy.optimize.least_squares` 作为非线性最小二乘求解器。
- 只求解 Panda 7 个 arm joints，不求解完整 `model.nq`，避免以后模型中出现 socket/free joint 或任务物体自由度时被 IK 意外修改。
- 对非 arm 自由度保留固定的 base `qpos` 快照，只在 IK `MjData` 中写入候选 arm joint。
- residual 建议采用如下结构：

```text
residual(q) = [
  w_pos * (p_current(q) - p_target),
  w_rot * orientation_error(current, target),
  w_reg * (q - q_regularization_target)
]
```

- bounds 直接来自 MuJoCo 关节限位。
- 支持多 seed，包括当前 `q`、home `q`、regularization target，以及可选用户指定 seed。
- 用最终 weighted residual 选择最优解，同时设置明确的位置误差和姿态误差验收阈值。
- 返回诊断信息，包括是否收敛、迭代次数、最终位置误差、最终姿态误差、选中的 seed index、optimizer cost。

实现时需要注意：

- 四元数顺序必须显式处理。MuJoCo 使用 `wxyz`，SciPy 使用 `xyzw`。
- 如果采用 MuJoCo quaternion error 逻辑，优先继续使用 MuJoCo 的四元数工具函数，避免误差定义前后不一致。
- 不建议把这套非线性 IK 放进每个 teleop control step。它适合 reset 和规划；在线速度控制仍应保持 Jacobian-based，以保证实时性和手感。

## SO(3) 本体导纳评估

`master_thesis_project/admittance_framework/algorithms.py` 中的 `AlgorithmIB` 与目标控制思路基本一致：

- 采用解耦的 `R^3 x SO(3)` 导纳模型。
- 平移部分在欧氏空间中处理。
- 旋转部分在 `SO(3)` 上处理。
- 旋转部分采用本体系、右不变形式。
- 旋转误差定义为：

```text
phi_b = Log(R_d^T R_c)
```

- 旋转速度为本体角速度 `omega_b`。
- 力矩输入为本体系力矩 `tau_b`。
- 李群更新采用右乘：

```text
R_c_next = R_c @ Exp(omega_b_next * dt)
```

这比当前的 6D world-frame rotvec admittance 更符合几何一致性，因为旋转柔顺轴可以固定在工具本体系上，而不是固定在世界坐标轴上。

## 推荐导纳控制设计

建议保留当前 SpaceMouse 名义目标的积分方式，只替换 admittance 虚拟末端层。

名义目标约定保持不变：

- SpaceMouse 平移输入继续在 world frame 中积分。
- SpaceMouse 旋转输入继续在末端 local frame 中积分。
- 内层跟踪继续使用 local pose error 和 local Jacobian DLS。

导纳层约定如下：

- 平移导纳使用 world frame：

```text
M_p xddot_w + D_p xdot_w + K_p x_w = force_world
p_virtual = p_nominal + x_w
```

- 旋转导纳使用 body-frame right-invariant `SO(3)`：

```text
phi_b = Log(R_nominal^T R_virtual)
M_r omega_dot_b + D_r omega_b + K_r phi_b = torque_body
R_virtual = R_nominal @ R_compliance
R_compliance_next = R_compliance @ Exp(omega_b_next * dt)
```

Wrench 处理方式：

- 从 `tool_tip_site` 的 force/torque sensor 读取末端接触力和力矩。
- 在 sensor local frame 中减去 tare bias。
- 平移导纳使用 world-frame force：

```text
force_world = R_tip @ force_body
```

- 旋转导纳保留 body-frame torque：

```text
torque_body = torque_body_sensor
```

滤波和安全保护：

- `force_world` 和 `torque_body` 分别做低通滤波。
- force 和 torque 分别做 deadband。
- 限制平移 offset、平移速度、SO(3) rotvec 幅值和本体角速度。
- teleop reference reset 时清空所有导纳状态。

## 训练后仿真过快问题评估

当前训练结果曲线和 checkpoint 本身看上去已经可以收敛，但用训练好的模型做仿真评估时，10 个 episode 平均约 0.17 秒就成功完成一轮任务，这个现象需要拆成两个问题分别看：

1. 评估统计中的“成功时间”到底是 wall-clock time，还是 MuJoCo simulated time。
2. 如果 simulated time 也确实只有 0.17 秒，那么任务本身可能允许策略以不真实的速度完成插入。

当前第一步应先修正时间语义，而不是马上改 reward 或 success condition。原因是：

- wall-clock time 反映的是程序跑得多快，不反映仿真里的真实物理时间。
- MuJoCo 可以远快于实时运行；如果用 `time.time()` 或 `time.perf_counter()` 直接统计成功耗时，得到的 0.17 秒可能只是计算耗时。
- checkpoint eval 中应该优先报告 simulated success time，也就是 `raw_env.data.time - episode_start_sim_time`。
- 如果需要看“真实时间一一对应”的仿真效果，应该在 eval/playback 阶段按 MuJoCo simulated time 主动 sleep，把可视化播放同步到 1x real time。

需要明确区分：

- `success_wall_time`：从 Python 程序开始 rollout 到 success 的真实墙钟耗时，主要用于性能分析。
- `success_sim_time`：MuJoCo 仿真世界中的任务耗时，才是评估策略动作是否真实的核心指标。
- `video_playback_time`：导出视频的播放时长，由采样帧率和 simulated time 共同决定。

如果修正后发现 `success_sim_time` 仍然只有十几步，例如 `control_dt=0.01` 且 17 step 成功，那么问题就不只是统计口径，而是任务定义太容易或成功判定太早。可能原因包括：

- reset 初始位姿距离 socket 太近，策略只需要少量动作即可满足 success。
- success threshold 只检查瞬时成功，没有要求稳定保持一段时间。
- action/reference velocity limit 对训练策略来说仍然偏大。
- reward 的 success bonus 很高，策略倾向于最快触发 success，而不是形成接近真实装配节奏的动作。
- failure/contact penalty 不足，策略可能学到“猛插也能成功”的行为。

这部分不建议和 IK、导纳重构同时修改。因为一旦改变 success dwell time、动作速度限制或 reward，已有 checkpoint 的评估结果就不再可直接比较，并且很可能需要重新训练。

## 推荐评估与实时播放设计

建议先做一个最小、低风险的 eval 修正：

- checkpoint eval 同时记录 `success_sim_time` 和 `success_wall_time`。
- 默认打印 `Average simulated success time`，避免误把程序运行速度当成任务完成时间。
- 新增实时同步参数，例如：

```text
--eval_realtime_factor 1.0
```

- `eval_realtime_factor=1.0` 时，eval 每一步后根据 MuJoCo simulated time 和 wall-clock time 对齐播放。
- `eval_realtime_factor=0.0` 时，关闭 sleep，恢复最快速批量评估。
- 导出视频时继续按 simulated time 采样帧，保证视频时长和仿真时间一致。

推荐 eval 输出类似：

```text
Episode 1: success=True
  success_sim_time: 0.84s
  success_wall_time: 0.17s
Final success rate: 10/10
Average simulated success time: 0.91s
Average wall-clock rollout time: 0.18s
```

只有当 `success_sim_time` 也明显过短时，再进入第二阶段任务定义修改：

- 增加 `success_dwell_steps`，要求连续满足 success 条件若干步后才 terminate。
- 增加插入速度上限或 reference velocity 上限。
- 在 reward 中加入过大速度、过大接触力或过快插入的惩罚。
- 增大 reset 初始扰动范围，使任务不是从近乎完成的位置开始。
- 将 evaluation 中的每步动作和末端速度记录下来，用数据判断到底是“显示快”还是“策略动作快”。

## 推荐实施顺序

建议按以下顺序推进：

1. 先把 reset/planning IK 路径替换为共享的 least-squares site IK solver。
2. 加入 IK 诊断信息和 reset 压测，再修改 controller 行为。
3. 将当前 world-frame rotational admittance 替换为 body/right-invariant `SO(3)` admittance。
4. 分别验证 admittance 关闭和开启时的 teleop 行为。
5. 修正 checkpoint eval 的时间统计和 1x 实时播放同步，确认 `success_sim_time` 和 `success_wall_time` 的差异。
6. 如果 simulated success time 仍然不真实，再单独修改 success dwell time、速度限制或 reward，并重新训练。

评估时间问题应单独处理。第一步应先把 checkpoint eval 中的成功时间改成 MuJoCo simulated time，并可选支持实时播放同步。奖励函数或 success dwell time 的修改应放在后面，因为这些改动会影响已有 checkpoint 的可比性，并且可能需要重新训练。

## 风险与测试

主要风险：

- 四元数顺序错误会导致 IK 数值上看似收敛，但实际姿态是错的。
- 如果求解完整 `model.nq`，可能意外移动任务物体或未来加入的 free joint。
- IK 正则化过强会改善姿态舒适度，但牺牲末端位姿精度。
- IK 正则化过弱仍然可能产生不利于插入任务的肘部姿态。
- force/torque sensor frame 搞错会直接导致柔顺方向反向或错轴。
- 把 body torque 和 world angular velocity 混用，或者把 right-invariant error 和 left-multiply update 混用，会破坏 SO(3) 导纳语义。
- 导纳增益过大或显式欧拉积分过粗，可能在接触时产生振荡。

建议测试：

- 连续运行多次 insertion env reset，记录最大位置误差和姿态误差。
- 在 socket 周围设置多个可达目标，测试 IK multi-seed 是否选择更合理的解。
- 人工设置不可达目标，确认 IK 失败时能输出清晰诊断。
- admittance 关闭时，确认 teleop 输出和当前 controller 行为基本一致。
- wrench 为零时，确认 `virtual_transform == target_transform`。
- 注入恒定 world force，确认平移柔顺沿世界方向退让。
- 注入恒定 body torque，确认旋转柔顺沿工具本体轴退让。
- 改变工具姿态后重复 body torque 测试，确认旋转柔顺轴跟随本体系。
- 确认 wrench tare 可以清除 bias，reset 可以清空所有柔顺状态。
- 在修改 reward 或 success threshold 前，先区分 simulated success time 和 wall-clock runtime。
- checkpoint eval 输出应同时包含 simulated success time 和 wall-clock rollout time。
- `--eval_realtime_factor=1.0` 时，viewer 或视频观察到的动作节奏应接近 1x 实时。
- 如果 `success_sim_time` 仍明显过短，再用 episode step count、末端速度、action 范数和接触力数据判断是否需要改任务定义。
