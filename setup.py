#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

setup(name = 'MnuLFI',
      version = __version__,
      python_requires='>3.5.2',
      description = 'Mnu Likelihood-Free Inference',
      requires = ['numpy', 'matplotlib', 'scipy'],
      provides = ['mnulfi'],
      packages = ['mnulfi']
      )
