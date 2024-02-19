#create a setup.py file to install the package
from setuptools import setup, find_packages


setup(
    name="RnaBench",
    version="0.1",
    description="RNA benchmarking tools and utilities.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'forgi==2.0.2',
        "ViennaRNA",
        "biopython",
        "dask",
        "nltk==3.8.1",
        "regex==2022.10.31",
        "Distance==0.1.3",
        "distlib==0.3.6",
        "gdown==4.6.4",
        "google-auth==2.19.1",
        "google-auth-oauthlib==1.0.0",
        "google-pasta==0.2.0",
        "GraKeL==0.1.9",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "pyaml==21.10.1",
        "pynisher==1.0.2",
        "pytest",
        "PyYAML",
        "torch==1.13.1",
        "torchvision==0.14.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        # Add other relevant classifiers.
    ],
# Optional
    author='Karim Farid',
    author_email='faridk@informatik.uni-freiburg.de',
    license='MIT',
    keywords='RNA benchmarking',
    url='https://github.com/Rungetf/RnaBench.git'
)