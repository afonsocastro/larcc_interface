#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='utilsfunctionslib',
    packages=find_packages(include=['utilspythonlib']),
    version='0.1.0',
    description='Python library to communicate with robot\'s controllers',
    author='Joel Baptista',
    license='Universidade de Aveiro',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',

)
