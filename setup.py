import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List, Dict, Any

# Define constants
PROJECT_NAME = "enhanced_cs"
VERSION = "1.0.0"
DESCRIPTION = "A Multimodal Neural Network for Recognizing Subjective Self-Disclosure"
AUTHOR = "Henry Powell, Guy Laban, Emily S. Cross"
EMAIL = "henry.powell@example.com, guy.laban@example.com, emily.cross@example.com"
URL = "https://github.com/example/enhanced_cs"

# Define dependencies
DEPENDENCIES: List[str] = [
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "opencv-python",
]

# Define optional dependencies
OPTIONAL_DEPENDENCIES: Dict[str, List[str]] = {
    "dev": ["pytest", "flake8", "mypy"],
    "docs": ["sphinx", "sphinx-rtd-theme"],
}

# Define package data
PACKAGE_DATA: Dict[str, List[str]] = {
    "": ["*.txt", "*.md", "*.json"],
}

# Define entry points
ENTRY_POINTS: Dict[str, List[str]] = {
    "console_scripts": ["enhanced_cs=enhanced_cs.main:main"],
}

# Define classifiers
CLASSIFIERS: List[str] = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

# Define long description
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

class CustomInstallCommand(install):
    """Custom install command to handle additional installation tasks."""

    def run(self):
        """Run the custom installation tasks."""
        install.run(self)
        print("Custom installation tasks completed.")

class CustomDevelopCommand(develop):
    """Custom develop command to handle additional development tasks."""

    def run(self):
        """Run the custom development tasks."""
        develop.run(self)
        print("Custom development tasks completed.")

class CustomEggInfoCommand(egg_info):
    """Custom egg info command to handle additional egg info tasks."""

    def run(self):
        """Run the custom egg info tasks."""
        egg_info.run(self)
        print("Custom egg info tasks completed.")

def main():
    """Main function to setup the package."""
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=find_packages(),
        install_requires=DEPENDENCIES,
        extras_require=OPTIONAL_DEPENDENCIES,
        package_data=PACKAGE_DATA,
        entry_points=ENTRY_POINTS,
        classifiers=CLASSIFIERS,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
    )

if __name__ == "__main__":
    main()