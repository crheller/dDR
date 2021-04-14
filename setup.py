from setuptools import setup, find_packages

setup(name='dDR',
      version='0.1.0',
      description='Dimensionality reduction for neural decoding',
      url='https://github.com/crheller/dDR.git',
      author='Charlie Heller',
      author_email='charlieheller95@gmail.com',
      license='MIT',
      install_requires=['numpy'],
      extras_require={
        'demos|figures': ['jupyter', 'seaborn', 'pickle', 'matplotlib']
      },
      packages=find_packages(include=['dDR', 'dDR.*']),
      zip_safe=False)
