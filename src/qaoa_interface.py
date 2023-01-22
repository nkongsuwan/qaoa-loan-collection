from abc import ABC, abstractmethod

class QaoaInterface(ABC):
    @abstractmethod
    def __init__(self):
        self.xxx = 1

    @abstractmethod
    def test(self):
        pass