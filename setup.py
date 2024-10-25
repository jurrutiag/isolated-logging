from glob import glob
from os.path import basename, splitext
from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
HERE = Path(__file__).parent
long_description = (HERE / "README.md").read_text()

setup(
    name="isolated-logging",
    version="0.0.3",
    author="Juan Urrutia",
    author_email="juan.urrutia.gandolfo@gmail.com",
    description="A Python library for tracking and logging function and loop execution times with stats and color-coded logs for easy performance monitoring and optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jurrutiag/isolated-logging",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[],
    extras_require={
        "testing": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "tox>=3.24",
        ]
    },
)
