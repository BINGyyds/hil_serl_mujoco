from setuptools import setup, find_packages

setup(
    name="hil_serl_mujoco",
    version="0.1.0",
    description="HIL-SERL with MuJoCo simulation for Franka Panda",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "gymnasium==0.29.1",
        "mujoco>=3.0.0",
        "numpy>=1.24.3",
        "scipy>=1.11.0",
        "typing_extensions",
        "opencv-python",
        "lz4",
    ],
    extras_require={
        "train": [
            "jax[cuda12]",
            "flax>=0.8.0",
            "distrax>=0.1.2",
            "chex>=0.1.85",
            "optax>=0.1.5",
            "wandb>=0.12.14",
            "tqdm>=4.60.0",
            "absl-py>=0.12.0",
            "natsort",
            "tensorflow>=2.15.0",
            "tensorflow_probability>=0.23.0",
        ],
        "sb3": [
            "stable-baselines3",
        ],
        "hil": [
            "zmq",
            "easyhid",
            "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d",
        ],
    },
    python_requires=">=3.9",
    zip_safe=False,
)
