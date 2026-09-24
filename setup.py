from setuptools import setup, find_packages

setup(
    name="alert",
    version="0.1.0",
    description="ALERT — Adaptive Landmark Eye-state & Reaction-Time fatigue detection",
    author="Prem",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mediapipe>=0.10.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pygame>=2.5.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "alert-run=scripts.run_alert:main",
            "alert-train=scripts.train:main",
        ],
    },
)
