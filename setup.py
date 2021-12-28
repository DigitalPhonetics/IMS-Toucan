from pathlib import Path
from typing import List

from setuptools import setup, find_packages

project_root = Path(__file__).parent

install_requires: List[str] = []

print(find_packages())

setup(name="ims_toucan", version="0.0.1", packages=find_packages(), python_requires=">=3.8",
        install_requires=install_requires, )
