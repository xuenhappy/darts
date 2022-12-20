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

import codecs
import os
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution

_scripts_dir = os.path.split(os.path.realpath(__file__))[0]


def long_description():
    reamdme = os.path.join(_scripts_dir, '../README.md')
    with codecs.open(reamdme, 'r', 'utf-8') as fp:
        long_description = fp.read()
    return long_description


def copy_follow_symlinks(src, dstDir):
    shutil.copy(src, dstDir, follow_symlinks=False)
    if os.path.islink(src):
        return copy_follow_symlinks(os.path.realpath(src), dstDir)


def wheel_name(**kwargs):
    fuzzlib = Extension('fuzzlib', ['fuzz.pyx'])  # the files don't need to exist
    # create a fake distribution from arguments
    dist = Distribution(attrs={**kwargs, 'ext_modules': [fuzzlib]})
    # finalize bdist_wheel command
    bdist_wheel_cmd = dist.get_command_obj('bdist_wheel')
    bdist_wheel_cmd.ensure_finalized()
    return {'plat_name': bdist_wheel_cmd.plat_name, 'python_tag': bdist_wheel_cmd.python_tag}


class build_py(_build_py):
    """Custom build command."""

    def build_package_data(self):
        super().build_package_data()
        build_dir = os.path.join(*([self.build_lib]))
        shutil.copytree(os.path.join(_scripts_dir, '../data'), os.path.join(build_dir, 'darts/data'))
        shutil.copytree('/opt/onnxruntime/lib/', os.path.join(build_dir, 'darts/'), dirs_exist_ok=True)


setup_kwargs = dict(name='darts',
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
    'Development Status :: 5 - Production/Stable', 'Environment :: Console', 'Intended Audience :: Developers',
    'Intended Audience :: Science/Research', 'License :: OSI Approved :: Apache Software License',
    'Operating System :: Unix', 'Programming Language :: Python', 'Topic :: Text Processing :: Linguistic',
    'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    cmdclass={
    'build_py': build_py,
    },
    package_data={
    '': ['*.*'],
    },
    include_package_data=True,
    python_requires='>=3.8, <=3.10')

setup(**setup_kwargs, options={'bdist_wheel': wheel_name(**setup_kwargs)})
