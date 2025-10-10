#!/usr/bin/env python
"""Minimal setup.py for package installation."""

from setuptools import setup, find_packages

# Read version from pyproject.toml (simple parsing)
import re

with open("pyproject.toml", "r") as f:
    content = f.read()
    version_match = re.search(r'^version = "(.+)"', content, re.MULTILINE)
    version = version_match.group(1) if version_match else "0.2.0"

setup(
    name="causal-judge-evaluation",
    version=version,
    description="Causal Judge Evaluation - Unbiased LLM evaluation framework",
    author="Eddie Landesberg",
    author_email="eddie@fondutech.com",
    packages=find_packages(),
    python_requires=">=3.9,<3.13",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "scipy>=1.11.0",
        "pydantic>=2.0",
        "typing-extensions>=4.0",
    ],
    extras_require={
        # Documentation now hosted on cimolabs.com
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
