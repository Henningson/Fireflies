from setuptools import setup, find_packages
from os import path


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="Fireflies",
    version="1.0",
    description="A module for randomizing mitsuba scenes and their parameters originally created for Structured Light Endoscopy.",
    author="Jann-Ole Henningson",
    author_email="jann-ole.henningson@fau.de",
    url="https://github.com/Henningson/Fireflies",
    packages=["fireflies"],
    install_requires=install_requires,
    packages=find_packages(),
)
