#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
import os

# 读取README文件作为长描述
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="gym_hil",
    version="0.1.8",
    description="Human-in-the-loop gymnasium environments for robotics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="The HuggingFace Inc. team",
    author_email="thomaswolfcontact@gmail.com",
    url="https://github.com/huggingface/lerobot",
    license="Apache-2.0",
    packages=find_packages(),
    package_data={
        "gym_hil": [
            "assets/*.xml",
            "assets/*.json",
            "controller_config.json",
        ],
    },
    include_package_data=True,
    install_requires=[
        "gymnasium==0.29.1",
        "mujoco>=2.3.0",
        "numpy>=1.21.0",
        "opencv-python-headless>=4.9.0",
        "pynput>=1.7.7",
        "pygame>=2.5.1",  # 用于游戏手柄支持
        "hidapi>=0.14.0",  # 用于游戏手柄支持
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.0",
            "pytest-timeout>=2.4.0",
            "pytest-cov>=5.0.0",
        ],
        "full": [
            "transformers>=4.50.3",
            "protobuf>=5.29.3",
            "grpcio==1.71.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=["robotics", "gymnasium", "human-in-the-loop", "mujoco"],
    zip_safe=False,
)