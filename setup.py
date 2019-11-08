from setuptools import setup, find_packages

setup(
  name='lfv_pdnn',
  version='0.0.1',
  packages=find_packages(),
  install_requires=[
          'numpy>1.16',
          'matplotlib>3.0',
          'tensorflow>1.14',
          'keras>2.3',
          'scikit-learn>0.20',
          'ConfigParser>3.6',
          'reportlab>3.5'
      ],
  entry_points={
      'console_scripts': [
        'execute_pdnn_job=share.execute:main'
      ],
    },
)
