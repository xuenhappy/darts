def __bootstrap__():
    import os
    __py_dir = os.path.split(os.path.realpath(__file__))[0]
    os.environ['LD_LIBRARY_PATH'] = "%s:%s" % (__py_dir, os.environ.get('LD_LIBRARY_PATH', ""))
    os.environ['PYTHONPATH'] = "%s:%s" % (__py_dir, os.environ.get('PYTHONPATH', ""))
    from .cdarts import *


__bootstrap__()
