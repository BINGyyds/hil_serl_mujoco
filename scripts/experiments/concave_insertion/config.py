import os
from pathlib import Path
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    GripperCloseEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from scripts.experiments.config import DefaultTrainingConfig
from scripts.experiments.concave_insertion.wrapper import ConcaveInsertionEnv


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import tomllib as toml
    except ModuleNotFoundError:
        import tomli as toml
    with path.open("rb") as f:
        return toml.load(f)


def _to_array(value, default, dtype=float):
    if value is None:
        value = default
    return np.array(value, dtype=dtype)


_TOML_PATH = Path(__file__).with_name("config.toml")
_TOML = _load_toml(_TOML_PATH)
_ENV = _TOML.get("env", {})
_REALSENSE = _TOML.get("realsense", {})


# Defaults
_DEFAULT_TARGET = np.array([0.55, 0.00, 0.25, np.pi, 0.0, 0.0])
_DEFAULT_THRESHOLD = np.array([0.005, 0.005, 0.005, 0.02, 0.02, 0.02])


class EnvConfig(DefaultEnvConfig):
    SERVER_URL = _ENV.get("server_url", "http://127.0.0.2:5000/")

    use_dummy_camera = _REALSENSE.get("use_dummy", True)
    if use_dummy_camera:
        REALSENSE_CAMERAS = {
            "dummy": {"dummy": True}
        }
        IMAGE_CROP = {}
    else:
        # Placeholder for real camera setup. Provide camera config in code if needed.
        REALSENSE_CAMERAS = {}
        IMAGE_CROP = {}

    TARGET_POSE = _to_array(_ENV.get("target_pose"), _DEFAULT_TARGET)
    REWARD_THRESHOLD = _to_array(_ENV.get("reward_threshold"), _DEFAULT_THRESHOLD)

    RESET_POSE = _to_array(
        _ENV.get("reset_pose"),
        TARGET_POSE + np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
    )

    ABS_POSE_LIMIT_LOW = _to_array(
        _ENV.get("abs_pose_limit_low"),
        TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.1, 0.1, 0.4]),
    )
    ABS_POSE_LIMIT_HIGH = _to_array(
        _ENV.get("abs_pose_limit_high"),
        TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.1, 0.1, 0.4]),
    )

    RANDOM_RESET = bool(_ENV.get("random_reset", True))
    RANDOM_XY_RANGE = float(_ENV.get("random_xy_range", 0.01))
    RANDOM_RZ_RANGE = float(_ENV.get("random_rz_range", 0.03))

    ACTION_SCALE = tuple(_ENV.get("action_scale", [0.005, 0.05, 1.0]))

    DISPLAY_IMAGE = bool(_ENV.get("display_image", False))
    MAX_EPISODE_LENGTH = int(_ENV.get("max_episode_length", 100))

    COMPLIANCE_PARAM = _TOML.get("compliance", {})
    PRECISION_PARAM = _TOML.get("precision", {})


class TrainConfig(DefaultTrainingConfig):
    # Dummy image key to keep the pixel pipeline intact without RealSense.
    image_keys = ["dummy"]
    classifier_keys = []
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    # Use sparse reward from pose threshold
    use_reward_classifier = False

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = ConcaveInsertionEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env
