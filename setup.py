import os
import sys

import setuptools


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file.
with open(os.path.join(here, "README.md"), encoding="utf-8") as md:
    long_description = md.read()


version_path = os.path.join(os.path.dirname(__file__), "deeplookup")
sys.path.append(version_path)

from version import __version__


setuptools.setup(
    name="deeplookup",
    version=__version__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Deep Lookup - Deep Learning for Domain Name System",
    url="https://github.com/ybubnov/deep-lookup",
    author="Yasha Bubnov",
    author_email="girokompass@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=setuptools.find_packages(exclude=["tests"]),
    package_data={"deeplookup": ["weights/*.h5"]},
    tests_require=[
        "pytest>=6.0.0",
        "pytest-benchmark>=3.4.0",
    ],
    install_requires=[
        "nltk",
        "scipy",
        "sklearn",
        "gym",
        "pandas",
        "matplotlib",
        "numpy",
        "keract",
        "h5py",
        "Keras>=2.4.3",
        "tensorflow>=2.4.1",
        "tensorflow-datasets>=4.4.0",
        "pyts>=0.11.0",
        "keras-rl2>=1.0.4",
        "wandb>=0.12.0",
        "dnspython>=2.1.0",
    ],
)
