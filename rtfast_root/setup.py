from setuptools import setup, find_packages

setup(
    name="rtfast",
    version="2.0.1",
    description="RTFAST: a machine learning emulator for the reltrans model",
    author="Ben Ricketts, Tin Hadži Veljković",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={
        "rtfast": [
            "models/*",
            "scalers/*",
            "fortran/*",
        ],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "pandas",
        "tqdm",
        "glob",
        "joblib"
    ],
    python_requires=">=3.7",
)