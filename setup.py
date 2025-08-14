"""
Setup script for Universal NN Latent Upscaler
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version from __init__.py
def get_version():
    version_file = os.path.join("__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="wan-nn-latent-upscaler",
    version=get_version(),
    author="denrakeiw",
    author_email="your-email@example.com",
    description="Universal Neural Network Latent Upscaler for ComfyUI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wan_nn_latent",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/wan_nn_latent/issues",
        "Source": "https://github.com/yourusername/wan_nn_latent",
        "Documentation": "https://github.com/yourusername/wan_nn_latent/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "matplotlib>=3.5.0",
            "tqdm>=4.64.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "training": [
            "matplotlib>=3.5.0",
            "tqdm>=4.64.0",
            "scipy>=1.8.0",
            "scikit-image>=0.19.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
        "models": ["*.pt"],
    },
    keywords=[
        "comfyui",
        "latent",
        "upscaling",
        "neural-network",
        "stable-diffusion",
        "flux",
        "wan2.2",
        "image-processing",
        "ai",
        "machine-learning",
    ],
    zip_safe=False,
)
