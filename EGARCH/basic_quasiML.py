from abc import ABC, abstractmethod
import numpy as np


class BasicCalibrator(ABC):

    def __init__(self, price_series):
        self.returns = np.diff(np.log(price_series))  # "Y" is "returns" here
        self.s0 = price_series[0]
        self.T = len(self.returns)

    @abstractmethod
    def calibrate(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_paths(self, *args, **kwargs):
        pass