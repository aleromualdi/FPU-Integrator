from setuptools import find_packages, setup

packages = find_packages()

setup(
    name="FPUT",
    version="0.1.0",
    description="Fermi-Pasta-Ulam integrator",
    url="https://github.com/aleromualdi/fpu",
    author="Alessandro Romualdi and Gionni Marchetti",
    author_email="alessandro.romu@gmail.com",
    license="",
    packages=packages,
    install_requires=[],
    classifiers=[],
)
