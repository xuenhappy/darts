#!/usr/bin/env python3

'''
File: setup.py
Project: python
File Created: Monday, 27th December 2021 10:57:03 am
Author: Xu En (xuen@mokar.com)
-----
Last Modified: Wednesday, 5th January 2022 5:32:10 pm
Modified By: Xu En (xuen@mokahr.com)
-----
Copyright 2021 - 2022 Your Company, Moka
'''

from setuptools.command.build_py import build_py as _build_py
from setuptools import setup, find_packages
import codecs
import os


def long_description():
    with codecs.open('../README.md', 'r', 'utf-8') as f:
        long_description = f.read()
    return long_description


class build_py(_build_py):
    """Custom build command."""

    def build_package_data(self):
        super().build_package_data()
        build_dir = os.path.join(*([self.build_lib]))
        import shutil
        scripts_dir = os.path.split(os.path.realpath(__file__))[0]
        shutil.copytree(os.path.join(scripts_dir, '../data'), os.path.join(build_dir, 'darts/data'))


setarg = setup(
    name='darts',
    author='Xu En',
    author_email='xuen@mokahr.com',
    description='darts python wrapper',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    version="2.0.1",
    packages=find_packages(include=['darts', 'darts.*']),
    url='https://github.com/xuenhappy/darts',
    license='Apache',
    platforms=['Unix', 'MacOS'],
    classifiers=[
        'Development Status :: 5 - Production/Stable', 'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix', 'Programming Language :: Python',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    cmdclass={
        'build_py': build_py,
    },
    package_data={'': ['*.*'], },
    include_package_data=True,
    python_requires='>=3.8, <=3.10',
    scripts=['darts/bin/darts']
)
