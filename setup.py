from setuptools import setup

setup(name='sithcon',
      version='0.1',
      description='SithCon: Temporal Convolutions Through Compressed Time',
      url='https://github.com/compmem/SITHCon',
      license='Free for non-commercial use',
      author='Computational Memory Lab',
      author_email='bgj5hk@virginia.edu',
      packages=['sithcon'],
      install_requires=[
          'torch>=1.1.0',
      ],
      zip_safe=False)
