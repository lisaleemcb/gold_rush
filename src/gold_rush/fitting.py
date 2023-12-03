import time
import numpy as np
import emcee

from multiprocessing import Pool

def gaussian(x, mean, sigma, normed=False):
    N = 1

    if normed is True:
        N = 1 / np.sqrt(2 * np.pi * sigma**2)

    return N * np.exp(-(x - mean)**2 / (2 * sigma**2))

def log_prior(params, truths, priors, priors_width):
    omegab = params[0]
    omegac = params[1]
    h_fid = params[2]
    As = params[3]
    ns = params[4]
    #tau_fid = params[5]

    if priors == 'uniform':
        for i in range(params.size):
            if np.abs(truths[i]) * .9  < params[i] < np.abs(truths[i]) * 1.1:
                return 0.0
            return -np.inf

    if priors == 'Planck':
        omegabh2_Planck = 0.02237
        omegabh2_sigma = 0.00014

        omegach2_Planck = 0.11933
        omegach2_sigma = 0.00091

        H_0_Planck = 67.66
        H_0_sigma = .42
        h_Planck = H_0_Planck / 100
        h_sigma = H_0_sigma / 100

        ln1010As_Planck = 3.047
        ln1010As_sigma = 0.014

        ns_Planck = 0.9665
        ns_sigma = 0.0038

        if (omegabh2_Planck - omegabh2_sigma) > omegab * h_fid**2 > (omegabh2_Planck + omegabh2_sigma):
            print(omegab * h_fid**2)
            return -np.inf
        if (omegach2_Planck - omegach2_sigma) > omegac * h_fid**2 > (omegach2_Planck + omegach2_sigma):
            print(omegac * h_fid**2)
            return -np.inf
        if (ln1010As_Planck - ln1010As_sigma) > np.log(10**10 * As) > (ln1010As_Planck + ln1010As_sigma):
            print(np.log(10**10 * As))
            return -np.inf
        if (h_Planck - h_sigma) > h_fid > (h_Planck + h_sigma):
            print(f'h_fid was outside the priors at h_fid={h_fid}')
            return -np.inf
        if (ns_Planck - ns_sigma) > ns > (ns_Planck + ns_sigma):
            print(ns)
            return -np.inf

        return 0.0

    if priors == 'gaussian':
        priors_vals = np.zeros(params.size)

        for i in range(params.size):
            priors_vals[i] = np.log(gaussian(params[i], truths[i], truths[i] * priors_width))
            print(f'prior val is: {priors_vals[i]}')

    return priors_vals.sum()

def log_likelihood(params, data, model, sigmas):
    log_likelihood = -(data - model(params))**2 / sigmas**2
    return log_likelihood.sum()

def log_probability(params, truths, data, model, sigmas, priors, priors_width):
    lp = log_prior(params, truths, priors, priors_width)
    print(f'the prior contribution was {lp}')
    if not np.isfinite(lp):
        return -np.inf

    t0 = time.time()
    lklhd = log_likelihood(params, data, model, sigmas)
    tf = time.time()
    #print(f'{(tf-t0):.3f}', lp + lklhd)
    print(f'likelihood evaluation took {(tf-t0):.3f} seconds with value {lp + lklhd}')
    print(f'params for this evaluation was: {params}')

    return lp + lklhd

def start_mcmc(truths, data, model, sigmas,
                priors='gaussian', priors_width=.1,
                backend=None, nwalkers=36, nsteps=1e5, burn_in=50):

    ndim = truths.size
    nsteps = int(nsteps)

    print(f'params are {truths}')
    print(f'data / model is {data / model(truths)}')

    print(f'running emcee for {nsteps} steps with {nwalkers} walkers and {burn_in} burn in...')
    print(f'assuming {priors} priors...')
    print('The log posterior of the truth is:', log_probability(truths, truths, data, model, .01 * data, priors, .1))
    pos = truths * np.ones((nwalkers, ndim)) + 1e-10 * np.random.normal(scale=truths * np.ones((nwalkers, ndim)), size=(nwalkers, ndim))

    if backend:
        print(f'saving backend as {backend}')
        backend = emcee.backends.HDFBackend(backend)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            backend=backend, args=(truths, data, model, sigmas, priors, priors_width)
        )

    else:
        print('running without backend')
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(truths, data, model, sigmas, priors, priors_width)
        )

    t0 = time.time()
    state = sampler.run_mcmc(pos, burn_in)
    print('burn in complete. setting up for full run...')
    tf = time.time()
    print(f'mcmc burn in took {(tf-t0)/60/60:.3f} hours')

    sampler.reset()
    print('beginning run...')

    t0 = time.time()
    sampler.run_mcmc(state, nsteps, progress=True)
    tf = time.time()
    print(f'mcmc sampling took {(tf-t0)/60/60:.3f} hours')

    return sampler
