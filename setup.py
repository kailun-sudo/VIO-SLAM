#!/usr/bin/env python3
"""
VIO-SLAM: Visual-Inertial Odometry SLAM
A comprehensive SLAM implementation with loop closure detection
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    with open(os.path.join("src", "vio_slam", "__init__.py"), "r") as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read long description from README.md
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Remove comments from line
                req = line.split('#')[0].strip()
                if req:
                    requirements.append(req)
        return requirements

# Development requirements
dev_requirements = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "flake8>=3.8.0",
    "pre-commit>=2.13.0",
    "mypy>=0.812",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
]

# Optional requirements for different features
extras_require = {
    "dev": dev_requirements,
    "gpu": ["torch>=1.9.0", "torchvision>=0.10.0"],
    "ros": ["rospy", "geometry_msgs", "sensor_msgs"],
    "deep": ["tensorflow>=2.5.0", "keras>=2.4.0"],
    "all": dev_requirements + ["torch>=1.9.0", "tensorflow>=2.5.0"],
}

setup(
    name="vio-slam",
    version="0.1.0",  # Will be replaced by get_version() when __init__.py exists
    author="Your Name",
    author_email="your.email@example.com",
    description="Visual-Inertial Odometry SLAM with loop closure detection",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vio-slam",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/vio-slam/issues",
        "Documentation": "https://vio-slam.readthedocs.io/",
        "Source": "https://github.com/yourusername/vio-slam",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.60.0",
        "scikit-learn>=0.24.0",
        "Pillow>=8.0.0",
        "pandas>=1.2.0",
        "PyYAML>=5.4.0",
    ],
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "vio-slam=vio_slam.cli:main",
            "slam-eval=vio_slam.evaluation:main",
            "slam-viz=vio_slam.visualization:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vio_slam": [
            "config/*.yaml",
            "data/sample/*",
            "assets/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "slam",
        "visual-inertial-odometry",
        "computer-vision",
        "robotics",
        "localization",
        "mapping",
        "sensor-fusion",
        "loop-closure",
        "pose-estimation",
    ],
    # Additional metadata
    platforms=["any"],
    license="MIT",
    maintainer="Your Name",
    maintainer_email="your.email@example.com",
    download_url="https://github.com/yourusername/vio-slam/archive/v0.1.0.tar.gz",
)