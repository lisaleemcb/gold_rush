import time
import numpy as np
import emcee

from multiprocessing import Pool

def log_prior(params):
    omegab = params[0]
    omegac = params[1]
    h_fid = params[2]
    As = params[3]
    ns = params[4]
    #tau_fid = params[5]

    if 1.8e-09 < As < 2.5e-09:
        return 0.0
    # if -5.0 < omegab < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
    #     return 0.0
    return -np.inf

def log_likelihood(params, data, model, sigmas):
    log_likelihood = -(data - model(params))**2 / sigmas**2
    return log_likelihood.sum()

def log_probability(params, data, model, sigmas):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    t0 = time.time()
    lklhd = log_likelihood(params, data, model, sigmas)
    tf = time.time()
    #print(f'{(tf-t0):.3f}', lp + lklhd)
    print(f'likelihood evaluation took {(tf-t0):.3f} seconds with value {lp + lklhd}')

    return lp + lklhd

def start_mcmc(truths, data, model, sigmas, backend=None,
                    nwalkers=36, nsteps=1e5, burn_in=50):
    t0 = time.time()
    ndim = truths.size
    nsteps = int(nsteps)

    print(f'ndim type is {type(ndim)}')
    print(f'ndim type is {type(nsteps)}')

    print(f'running emcee for {nsteps} steps with {nwalkers} walkers and {burn_in} burn in...')
    pos = truths * np.ones((nwalkers, ndim)) + 1e-10 * np.random.normal(scale=truths * np.ones((nwalkers, ndim)), size=(nwalkers, ndim))

    if backend:
        print(f'saving backend as {backend}')
        backend = emcee.backends.HDFBackend(backend)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            backend=backend, args=(data, model, sigmas)
        )

    else:
        print('running without backend')
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(data, model, sigmas)
        )
    state = sampler.run_mcmc(pos, burn_in)
    sampler.reset()
    sampler.run_mcmc(state, nsteps)

    sampler.run_mcmc(pos, nsteps, progress=True)

    tf = time.time()
    print(f'mcmc sampling took {(tf-t0)/60/60:.3f} hours')

    return sampler
