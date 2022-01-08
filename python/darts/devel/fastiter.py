from multiprocessing import cpu_count, Process, Queue, Event


class _getter(Process):
    def __init__(self, args, argqueue, worknum, done):
        Process.__init__(self)
        self.worknum = worknum
        self.args = args
        self.argqueue = argqueue
        self.done = done

    def run(self):
        for arg in self.args:
            if arg is None:
                continue
            self.argqueue.put(arg)
        for _ in range(self.worknum):
            self.argqueue.put(None)
        self.done.wait()


class _worker(Process):
    def __init__(self, process, argqueue, resqueue, done):
        Process.__init__(self)
        self.process = process
        self.argqueue = argqueue
        self.resqueue = resqueue
        self.done = done

    def run(self):
        while True:
            arg = self.argqueue.get()
            if arg is None:
                break
            if not isinstance(arg, tuple):
                res = self.process(arg)
            else:
                res = self.process(*arg)
            if res is not None:
                self.resqueue.put(res)
        self.resqueue.put(None)
        self.done.wait()


class MutiProcessIter():
    def __init__(self, argIter, process, work_num=0):
        if work_num < 1:
            work_num = cpu_count()
        self.worknum = work_num
        self.process = process
        self.args = argIter
        self.argqueue = Queue(10*work_num+1)
        self.resqueue = Queue(10*work_num+1)
        self.startFlag = False

    def _start(self):
        if self.startFlag:
            return
        self.done = Event()
        self.getter = _getter(self.args, self.argqueue, self.worknum, self.done)
        self.workers = [_worker(self.process, self.argqueue, self.resqueue, self.done) for _ in range(self.worknum)]
        self.getter.start()
        for worker in self.workers:
            worker.start()
        self.startFlag = True

    def _close(self):
        if not self.startFlag:
            return
        self.done.set()
        self.getter.join()
        for worker in self.workers:
            worker.join()
        self.resqueue.close()
        self.argqueue.close()

    def __iter__(self):
        self._start()
        noneNum = self.worknum
        while True:
            if noneNum < 1:
                break
            res = self.resqueue.get()
            if res is None:
                noneNum -= 1
                continue
            yield res
        self._close()
