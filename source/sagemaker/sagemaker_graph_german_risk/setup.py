from setuptools import setup, find_packages


setup(
    name='sagemaker_graph_german_risk',
    version='1.0',
    description='A package to organize code in the solution',
    packages=find_packages(exclude=('test',))
)