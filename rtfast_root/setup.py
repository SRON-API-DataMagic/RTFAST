from setuptools import setup, find_packages

setup(
    name="rtfast_root",
    version="0.1.0",
    description="RTFast: Relativistic Transfer Function Training Package",
    author="Your Name",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={
        "rtfast": [
            "../models/*",
            "../scalers/*",
            "../fortran/*",
        ],
    },
    include_package_data=True,
    install_requires=[
        # Add your dependencies here, e.g.:
        # "numpy",
        # "torch",
    ],
    python_requires=">=3.7",
)