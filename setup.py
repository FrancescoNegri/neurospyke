import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'neurospyke'
AUTHOR = 'Francesco Negri'
AUTHOR_EMAIL = 'francesco.negri@outlook.com'
URL = 'https://github.com/FrancescoNegri/neurospyke'

LICENSE = 'MIT License'
DESCRIPTION = 'A work-in-progress Python3 library for neural signal analysis and visualization.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'matplotlib',
      'numpy',
      'scipy'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )