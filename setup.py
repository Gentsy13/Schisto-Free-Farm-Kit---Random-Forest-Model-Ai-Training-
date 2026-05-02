from setuptools import setup, find_packages

setup(
    name="schisto-free-farm-kit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "schisto-train=src.train:_cli",
            "schisto-predict=src.predict:_cli",
        ]
    },
    python_requires=">=3.10",
)
