from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    name="isolated-logging",
    version="0.0.1",
    description="Utility to log in an isolated way",
    author="Juan Urrutia",
    author_email="juan.urrutia.gandolfo@gmail.com",
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
