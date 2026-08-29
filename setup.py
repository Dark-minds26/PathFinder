from setuptools import find_packages, setup

setup(
    name="learning-path-recommender",
    version="0.1.0",
    packages=find_packages(include=["src*", "api*"]),
    include_package_data=True,
    package_data={"api": ["static/*"]},
    python_requires=">=3.11",
)
