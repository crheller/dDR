from setuptools import setup

setup(name='dDR',
      version='0.1.0',
      description='Dimensionality reduction for neural decoding',
      url='https://github.com/crheller/dDR.git',
      author='Charlie Heller',
      author_email='charlieheller95@gmail.com',
      license='MIT',
      install_requires=['numpy', 'matplotlib'],
      extras_require={
        'demos': ['jupyter']
      },
      packages=['dDR', 'utils'],
      zip_safe=False)
