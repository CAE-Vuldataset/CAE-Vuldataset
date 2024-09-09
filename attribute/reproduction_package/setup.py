from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name="dq_analysis",
    version="0.1",
    install_requires=install_requires,
    packages=find_packages()
)
