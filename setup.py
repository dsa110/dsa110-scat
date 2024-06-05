from setuptools import setup

"""
try:
    version = get_git_version()
    assert version is not None
except (AttributeError, AssertionError):
    version = '1.0.0'
"""

setup(name='dsa-110_scat-dev',
      version='1.0.0',
      description='DSA-110 Scattering and Scintillation Utilities',
      packages=['scattering','scintillation'],
      package_dir={
        "": ".",
        "scattering": "./scattering",
        "scintillation": "./scintillation",
        }
      install_requires=[
          'numpy',
          'matplotlib',
          'sigpyproc',
          'lmfit',
          'bilby',
          'astropy',
          'scipy',
          'pandas',
          'tqdm'
          ],

     )