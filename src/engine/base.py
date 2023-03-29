import abc
from utils.timer import Timer
class Base(object):
    def __init__(self):
        __metaclass__ = abc.ABCMeta
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

    @abc.abstractmethod
    def _make_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return