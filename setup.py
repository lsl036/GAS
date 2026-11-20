"""Setup scripts."""

from setuptools import find_namespace_packages, setup

setup(
    name="gas-attention-research",
    version="0.1",
    author="Chaofan Lin",
    package_dir={"": "."},
    packages=find_namespace_packages(include=["gas_attention", "gas_attention.*"]),
)
