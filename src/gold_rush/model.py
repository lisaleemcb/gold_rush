import time
import numpy as np
import matplotlib.pyplot as plt

import zeus21

#set up the CLASS cosmology
from classy import Class

class Model:
    def __init__(self, params, z=13, verbose=False):
        self.omegab = params[0]
        self.omegac = params[1]
        self.h_fid = params[2]
        self.As = params[3]
        self.ns = params[4]
        #self.tau_fid = params[5]

        # create cosmoparams object to pass to classy
        self.CosmoParams_input = zeus21.Cosmo_Parameters_Input(
                                            omegab=self.omegab,
                                            omegac=self.omegac,
                                            h_fid=self.h_fid,
                                            As=self.As,
                                            ns=self.ns,
            #                                tau_fid=self.tau_fid,
                                            kmax_CLASS=300.0)
        self.z = z
        self.verbose = verbose

        # Now all the initialisations
        self.ClassCosmo = Class()
        self.ClassCosmo.compute()
        self.ClassyCosmo = zeus21.runclass(self.CosmoParams_input)
        if verbose:
            t0 = time.time()
            print('CLASS has run, we store the cosmology.')

        self.CosmoParams = zeus21.Cosmo_Parameters(self.CosmoParams_input,
                                                   self.ClassyCosmo)
        self.CorrFClass = zeus21.Correlations(self.CosmoParams, self.ClassyCosmo)
        if verbose:
            print('Correlation functions saved.')
        self.HMFintclass = zeus21.HMF_interpolator(self.CosmoParams, self.ClassyCosmo)
        if verbose:
            print('HMF interpolator built. This ends the cosmology part -- moving to astrophysics.')

        #set up your astro parameters too, here the peak of f*(Mh) as an example
        self.AstroParams = zeus21.Astro_Parameters(self.CosmoParams)

        ZMIN = 10.0 #down to which z we compute the evolution
        self.CoeffStructure = zeus21.get_T21_coefficients(self.CosmoParams, self.ClassyCosmo,
                                        self.AstroParams, self.HMFintclass, zmin=ZMIN)
        if verbose:
            print('SFRD and coefficients stored. Move ahead.')

        self.zlist = self.CoeffStructure.zintegral
        RSDMODE = 1 #which RSD mode you want, 0 is no RSDs (real space), 1 is spherical (as simulations usually take), 2 is mu~1 (outside the wedge, most relevant for observations)
        self.PS21 = zeus21.Power_Spectra(self.CosmoParams, self.ClassyCosmo, self.CorrFClass,
         self.CoeffStructure, RSD_MODE = RSDMODE)
        self.klist = self.PS21.klist_PS

        if verbose:
            print('Computed the 21-cm power spectrum. Initialisation complete!')
            tf = time.time()
            print(f'took {(tf-t0):.3f} seconds')

    def gen_PS21(self):
        _iz = min(range(len(self.zlist)), key=lambda i: np.abs(self.zlist[i]-self.z))

        return self.PS21.Deltasq_T21[_iz]


    def plot_PS21(self, choice, k_or_z='k'):
        fig, ax = plt.subplots(figsize=(5,3))

        if k_or_z == 'k':
            #choose a k to plot
            kchoose=choice # 0.3;
            _ik = min(range(len(self.klist)), key=lambda i: np.abs(self.klist[i]-kchoose))

            ax.semilogy(self.zlist, self.PS21.Deltasq_T21[:,_ik], color='k', linewidth=2.0)
            ax.semilogy(self.zlist, self.PS21.Deltasq_T21_lin[:,_ik], color='gray', linewidth=1.2)

            ax.set_xlabel(r'$z$');
            ax.set_ylabel(r'$\Delta^2_{21}\,\rm[mK^2]$');
            ax.legend([r'Full Zeus21', r'Linear'])

            ax.set_xlim([10, 25])
            ax.set_ylim([1,200])

        if k_or_z == 'z':
            #choose a z to plot
            zchoose=choice # 13.;
            _iz = min(range(len(self.zlist)), key=lambda i: np.abs(self.zlist[i]-zchoose))

            ax.loglog(self.klist, self.PS21.Deltasq_T21[_iz], color='k', linewidth=2.0)
            ax.loglog(self.klist, self.PS21.Deltasq_T21_lin[_iz], color='gray', linewidth=1.2)

            ax.set_xlabel(r'$k\,\rm [Mpc^{-1}]$');
            ax.set_ylabel(r'$\Delta^2_{21}\,\rm[mK^2]$');
            ax.legend([r'Full Zeus21', r'Linear'])

            ax.set_xlim([1e-2,1])
            ax.set_ylim([1,200])
