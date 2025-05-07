from setuptools import setup, find_packages

setup(
    name="autodrive",
    version="0.1.0",
    packages=find_packages(),
    description="Robust Autonomous Driving System Using Deep Learning and Generative AI",
    author="ME5920 Team",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "albumentations",
        "pillow",
        "tqdm",
        "tensorboard",
    ],
    python_requires=">=3.8",
) 