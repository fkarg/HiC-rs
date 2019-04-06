from setuptools import setup, find_packages

setup(name='stochastic-matrix-balancing',
      version='0.1',
      description='Fast stochastic matrix balancing in Rust',
      url='http://github.com/fkarg/HiC-rs',
      author='Felix Karg',
      author_email='kargf@informatik.uni-freiburg.de',
      license='GPLv3',
      packages=find_packages(),
      include_package_data=True,
      package_data={'': ['stochastic-matrix-balancing/target/debug/*.so',
                         'stochastic-matrix-balancing/target/debug/*.so']},
      zip_safe=False)

