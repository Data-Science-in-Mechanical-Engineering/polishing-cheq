from setuptools import find_packages, setup


setup(
    name="polishing_robot",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.13.3,<2.0.0",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco==3.1.4",
        "Pillow",
        "opencv-python",
        "pynput",
        "termcolor",
        "agentlace@git+https://github.com/youliangtan/agentlace.git@e35c9c5ef440d3cc053a154c47b842f9c12b4356",
        "gymnasium",
        "pink-noise-rl",
        "stable-baselines3",
        "tqdm",
        "pandas",
        "wandb",
        "hydra-core",
        "Flask",
        "psutil",
        "ml-collections",
        "omegaconf",
        "matplotlib",
        "plotly",
        "dm-robotics-transformations",
        "imageio",
        "PyOpenGL==3.1.1a1" #or maybe another
    ],
)