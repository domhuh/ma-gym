import setuptools
from setuptools import setup
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='ma_gym',
      version='0.0.0',
      url='https://github.com/domhuh/ma-gym',
      author='Dom Huh',
      author_email='dhuh@ucdavis.edu',
      packages=setuptools.find_packages(),
      python_requires='>=3.6',
      install_requires=requirements,
      )
