'''
File: reader.py
Project: devel
File Created: Saturday, 8th January 2022 9:12:42 pm
Author: Xu En (xuen@mokar.com)
-----
Last Modified: Saturday, 8th January 2022 9:13:03 pm
Modified By: Xu En (xuen@mokahr.com)
-----
Copyright 2021 - 2022 Your Company, Moka
'''


class LineSampleReader():

    def __init__(self, filep):
        self.samplefile = filep

    def process(self, line):
        raise Exception("not impl")

    def sample_iter(self):
        with open(self.samplefile, encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                yield line

    def __iter__(self):
        if self.processNum < 1:
            for line in self.sample_iter():
                result = self.process(line)
                if not result:
                    continue
                yield result
        else:
            for result in MutiProcessIter(self.sample_iter(), self.process, self.processNum):
                if not result:
                    continue
                yield result
