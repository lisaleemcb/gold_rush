import time
import numpy as np
import matplotlib.pyplot as plt

import zeus21

from abc import ABC, abstractmethod

#set up the CLASS cosmology
from classy import Class

class Anomaly(ABC):
    def __init__(self, k, z, strength, modelparams, verbose=False):
        self.k = k
        self.z = z
        self.strength = strength
        self.modelparams = modelparams
        self.verbose = verbose

        self.scalefactor = 1

    def growthfactor(self):
            return np.ones_like(self.z)

    @abstractmethod
    def model(self):
        pass

    def signal(self):
        signal = self.model()
        signal *= self.strength
        signal = signal[None,:] * self.growthfactor()[:, None]

        return signal


class Compact(Anomaly):
    def __init__(self, k, z, strength, modelparams, verbose=False):
        super().__init__(k, z, strength, modelparams, verbose=False)


    def model(self):
        kindex = np.argmin(np.abs(self.k - self.modelparams.kscale))

        model = np.zeros_like(self.k)
        model[kindex] = self.modelparams.amplitude

        return model
    
class Infinite(Anomaly):
    def __init__(self, k, z, strength, modelparams, verbose=False):
        super().__init__(k, z, strength, modelparams, verbose=False)


    def model(self):
        model = self.modelparams.amplitude * np.sin(self.modelparams.omega * self.k)

        return model
    
class Gaussian(Anomaly):
    def __init__(self, k, z, strength, modelparams, verbose=False):
        super().__init__(k, z, strength, modelparams, verbose=False)


    def model(self):
        model = (1.0 / np.sqrt(2.0 * np.pi * self.modelparams.sigma**2)) 
    
        model = np.exp(-(self.k - self.modelparams.mu)**2.0 / self.modelparams.sigma**2.0)

        return model