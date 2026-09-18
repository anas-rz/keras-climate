from setuptools import setup, find_packages

setup(
    name="keras_climate",
    version="0.1.0",
    description="Keras 3 framework for climate modeling and Earth observation",
    packages=find_packages(),
    install_requires=["keras>=3.0", "numpy"],
    python_requires=">=3.9",
)
