from setuptools import setup

setup(name='tctct',
      version='0.1',
      description='TCTCT: Temporal Convolutions Through Compressed Time',
      url='https://github.com/beegica/TCTCT',
      license='Free for non-commercial use',
      author='Computational Memory Lab',
      author_email='bgj5hk@virginia.edu',
      packages=['tctct'],
      install_requires=[
          'torch>=1.1.0',
          
      ],
      zip_safe=False)