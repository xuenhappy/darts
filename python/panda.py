'''
File: panda.py
Project: devel
File Created: Thursday, 31st December 2020 10:49:59 pm
Author: enxu (xuen@mokar.com)
-----
Last Modified: Thursday, 31st December 2020 10:50:03 pm
Modified By: enxu (xuen@mokahr.com)
-----
Copyright 2021 - 2020 Your Company, Moka
'''
import ctypes
import json
import os
__py_dir = os.path.split(os.path.realpath(__file__))[0]
lib = ctypes.cdll.LoadLibrary(os.path.join(__py_dir, 'panda.so'))

# token function
freeStr = lib.freeCStr
freeStr.argtypes = ctypes.c_void_p,
freeStr.restype = None


buildFileTireDict = lib.buildFileTireDict
buildFileTireDict.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
buildFileTireDict.restype = None


basicToken = lib.basicToken
basicToken.argtypes = [ctypes.c_char_p, ctypes.c_int]
basicToken.restype = ctypes.c_void_p

develQSample = lib.develQSample
develQSample.argtypes = ctypes.c_char_p,
develQSample.restype = ctypes.c_void_p


def __point2json__(point):
    go_str = str(ctypes.c_char_p(point).value, encoding="utf-8")
    freeStr(point)
    if go_str:
        return json.loads(go_str)
    return None


def bToken(content, piceEng=False):
    if content is None:
        return []
    goResult = basicToken(content.encode("utf-8"), int(piceEng))
    return __point2json__(goResult)


def toSample(content):
    if content is None:
        return []
    goResult = develQSample(content.encode("utf-8"))
    return __point2json__(goResult)


def buildDict(dictfile, outfile):
    buildFileTireDict(dictfile.encode("utf-8"), outfile.encode("utf-8"))


str = toSample("中华任命hello words!")
print(str)
