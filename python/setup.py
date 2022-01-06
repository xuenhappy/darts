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

from setuptools import setup, Extension
import codecs
import os
import sys
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


cflags = ["-Wall", "-Wextra", "-std=c++17"]
if sys.platform.lower() == 'darwin':
    cflags.extend(["-stdlib=libc++", "-mmacosx-version-min=10.14.6"])

extensions = [
    Extension("cdarts",
              sources=["darts/cdarts.pyx"],
              include_dirs=["../build/dist/include"],
              library_dirs=["../build/dist/lib"],
              libraries=["cdarts", ],
              language="c++",
              extra_compile_args=cflags,
              extra_link_args=cflags,
              extra_objects=[],),
]

if cythonize is not None:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)


def long_description():
    with codecs.open('../README.md', 'r', 'utf-8') as f:
        long_description = f.read()
    return long_description


setup(
    name='darts',
    author='Xu En',
    author_email='xuen@mokahr.com',
    description='SentencePiece python wrapper',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    version="",
    package_dir={'': 'darts'},
    url='https://github.com/google/sentencepiece',
    license='Apache',
    platforms='Unix',
    py_modules=[
        'sentencepiece/__init__', 'sentencepiece/sentencepiece_model_pb2',
        'sentencepiece/sentencepiece_pb2'
    ],
    ext_modules=extensions,
    classifiers=[
        'Development Status :: 5 - Production/Stable', 'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix', 'Programming Language :: Python',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
