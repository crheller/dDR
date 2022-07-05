from setuptools import setup, find_packages

setup(name='dDR',
      version='0.1.0',
      description='Dimensionality reduction for neural decoding',
      url='https://github.com/crheller/dDR.git',
      author='Charlie Heller',
      author_email='charlieheller95@gmail.com',
      license='MIT',
      python_requires='>=3.8.3',
      install_requires=['numpy', 'matplotlib'],
      extras_require={
        'extras': ['seaborn', 'scipy']
      },
      packages=find_packages(include=['dDR', 'dDR.*']),
      zip_safe=False)
