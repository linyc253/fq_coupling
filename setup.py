from setuptools import setup, find_packages

setup(
    name="fq_coupling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "qutip", "scipy"],
)
