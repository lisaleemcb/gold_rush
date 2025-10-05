import numpy as np
import matplotlib.pyplot as plt
import corner

import zeus21
import gold_rush.fitting

from gold_rush.model import Signal21cm
from gold_rush.fitting import *

z = 13

omegab = 0.0223828
omegac = 0.1201075
h_fid = 0.6781
As = 2.100549e-09
ns = 0.9660499

params = np.array([omegab, omegac, h_fid, As, ns])

H_0_Planck = 67.66
H_0_sigma = 0.42
h_Planck = H_0_Planck / 100
h_sigma = H_0_sigma / 100


def model(params):
    return Signal21cm(params, verbose=False).gen_PS21()


# k, data = np.load('../../docs/notebooks/zeus21_data_fiducial.npy')
# k, data = np.load('/jet/home/emcbride/packages/gold_rush/data/zeus21_data_fiducial.npy')
k = Signal21cm(params, verbose=False).klist
data = model(params)

print(f"k is {k}")
print(f"data is {data}")

sampler = gold_rush.fitting.start_mcmc(
    params,
    data,
    model,
    0.01 * data,
    nwalkers=params.size * 2,
    nsteps=1000,
    burn_in=50,
    backend=None,
    progress=True,
    priors="Planck",
)

np.save("test_samples", sampler.get_chain())
