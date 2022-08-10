from setuptools import setup, find_packages
setup(
    name="MegaSAM",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "torch",
        "numpy",
        "sklearn",
        "tqdm"
    ]
)