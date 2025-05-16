from setuptools import setup, find_packages

setup(
    name="regime_mamba",
    version="0.1.0",
    author="Anonymous",
    author_email="Anonymous@gmail.com",
    description="Mamba 기반 시장 레짐 식별 시스템",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://anonymous.4open.science/r/RegimeMamba-998F",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CC BY-SA 4.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
        "bayesian-optimization>=1.2.0",
        "scipy>=1.5.0",
        "mamba-ssm>=1.0.1",
        "causal-conv1d>=1.4.0"
    ],
)