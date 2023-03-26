import numpy as np
import argparse
import pandas as pd
from scipy import interpolate, integrate, optimize, stats
import pycbc.conversions as conv
import time
import emcee
import glob
import sys
from schwimmbad import MPIPool

from power_law_funcs import *


parser = argparse.ArgumentParser(description='Estimate the likelihood hyper-posterior for a given population of binary merger detections drawn from a power law in m1, and uniform in q.')

parser.add_argument('--N_MCMC', default="100", type=int,  help='number of MCMC steps to take (default: 100)')
#parser.add_argument('--burnin', default="50", type=int,  help='number of burn-in steps for MCMC (default: 50)')

parser.add_argument('--mcmc_params', default="0 1 2", type=int, nargs='+',  help='indices of parameters to run MCMC over, in the list [alpha, mmin, mmax] (default: 0 1 2)')

priors_mcmc_low_all = [-4.2, 4.98, 55.0]
priors_mcmc_high_all = [-3.8, 5.02, 65.0]

parser.add_argument('--N_events', default="100", type=int,  help='number of events to estimate likelihoods for (default: 100)')


parser.add_argument('--alpha_prior_low', default="-5.0",  type=float, help='lower limit on alpha prior (default: -5.0)')
parser.add_argument('--alpha_prior_high', default="-2.0",  type=float, help='upper limit on alpha prior (default: -2.0)')

parser.add_argument('--mmin_prior_low', default="4.5",  type=float, help='lower limit on mmin prior (default: 4.5)')
parser.add_argument('--mmin_prior_high', default="5.5",  type=float, help='upper limit on mmin prior (default: 5.5)')

parser.add_argument('--mmax_prior_low', default="40.0",  type=float, help='lower limit on mmax prior (default: 40.0)')
parser.add_argument('--mmax_prior_high', default="200.",  type=float, help='upper limit on mmax prior (default: 200.0)')


parser.add_argument('--n_m1_int', default="800", type=int,  help='number of integration bins over m1 (default: 800)')
parser.add_argument('--n_m2_int', default="300", type=int,  help='number of integration bins over m2 (default: 300)')



parser.add_argument('-i', '--inputdir',  default="../output/powerlaw_3.5_lams_m2_lim", type=str,  help='directory of input networks (default: ../output/powerlaw_3.5_lams_m2_lim)')

parser.add_argument('-o', '--outputdir',  default="../mcmc/powerlaw_3.5_m2_lim", type=str,  help='directory of mcmc sampler output (default: ../mcmc/powerlaw_3.5_m2_lim)')


parser.add_argument('--alpha_inj', default="-3.5",  type=float, help='slope of m1 power law (default: -3.5)')
parser.add_argument('--mmin_inj', default="5.0",  type=float, help='minimum primary mass in Solar Mass (default: 5.0)')
parser.add_argument('--mmax_inj', default="60.0",  type=float, help='maximum mass in Solar Mass (default: 60.0)')
parser.add_argument('--beta_inj', default="0.0",  type=float, help='slope of m2 power law (default: 0.0)')





args = vars(parser.parse_args())
# print(args)

N_MCMC = args["N_MCMC"]
#burnin = args["burnin"]
N_events = args["N_events"]

mcmc_params = args["mcmc_params"]

alpha_prior_low = args["alpha_prior_low"]
alpha_prior_high = args["alpha_prior_high"]
mmin_prior_low = args["mmin_prior_low"]
mmin_prior_high = args["mmin_prior_high"]
mmax_prior_low = args["mmax_prior_low"]
mmax_prior_high = args["mmax_prior_high"]

n_m1_int = args["n_m1_int"]
n_m2_int = args["n_m2_int"]

input_dir = args["inputdir"]
output_dir = args["outputdir"]

alpha_inj = args["alpha_inj"]
mmin_inj = args["mmin_inj"]
mmax_inj = args["mmax_inj"]
beta_inj = args["beta_inj"]


# Read the event data

files_mc = glob.glob(input_dir + '/*_Mc_*.csv')

df1 = pd.DataFrame()
for fi in files_mc:
    df_temp = pd.read_csv(fi)
    df1 = df1.append(df_temp)
# drop the zero rows that resulted from NoneType Networks
df_mc_raw = df1.loc[~(df1==0).all(axis=1)]
df_mc_raw = df_mc_raw[df_mc_raw["Mc"] > 0]
df_mc_raw = df_mc_raw[(df_mc_raw["m1"]/(1+df_mc_raw["z"])) <= mmax_inj]


files_eta = glob.glob(input_dir + '/*_eta_*.csv')

df1 = pd.DataFrame()
for fi in files_eta:
    df_temp = pd.read_csv(fi)
    df1 = df1.append(df_temp)
# drop the zero rows that resulted from NoneType Networks
df_eta_raw = df1.loc[~(df1==0).all(axis=1)]
df_eta_raw = df_eta_raw[df_eta_raw["Mc"] > 0]
df_eta_raw = df_eta_raw[(df_eta_raw["m1"]/(1+df_eta_raw["z"])) <= mmax_inj]

# select all data
df_mc = df_mc_raw
df_eta = df_eta_raw


# Sample m1 and m2 from the data
m1_mu_sampled, m2_mu_sampled,  m1_variance, m2_variance, m1_m2_covariance = sample_m1_m2_events(df_mc[:N_events], df_eta[:N_events], injected=True)
covariances = np.zeros((len(m1_mu_sampled), 2,2))

# compute joint posteriors on m1, m2
for i in range(len(m1_mu_sampled)):
    covariances[i] =  [[m1_variance[i], m1_m2_covariance[i]], [m1_m2_covariance[i], m2_variance[i]]]

Ns = len(m1_mu_sampled)
Nt = Ns

m1_min_int = mmin_prior_low
m1_max_int = mmax_prior_high
m_int_range = np.geomspace(m1_min_int, m1_max_int, n_m1_int)

#posteriors_m2 = np.zeros((Nt, n_m1_int, n_m1_int))
#for i in range(Nt):
#    for j in range(n_m1_int):
#        posteriors_m2[i,j] = bivariate_normal_dist(m_int_range[j], m_int_range, m1_mu_sampled[i], m2_mu_sampled[i], covariances[i])


def lnprob(hyper):
    alpha = hyper[0]
    m1_min_pow = hyper[1]
    m1_max_pow = hyper[2]
    beta = beta_inj

    integrand_m2 = np.zeros((Nt, n_m1_int))
    prior_m1 = power(m_int_range, alpha, m1_min_pow, m1_max_pow)
    prior_m1 = prior_m1 / integrate.trapezoid(prior_m1, m_int_range)


    for j in range(n_m1_int):
        m2_int_range = np.linspace(m1_min_int, m_int_range[j], n_m2_int)
        priors_m2 = power(m2_int_range, beta, m1_min_pow, m_int_range[j])
        for i in range(Nt):
            posteriors_m2 = bivariate_normal_dist(m_int_range[j], m2_int_range, m1_mu_sampled[i], m2_mu_sampled[i], covariances[i])
            integrand_m2[i,j] = integrate.trapezoid(priors_m2 * posteriors_m2, m2_int_range)    

    integrands = prior_m1 * integrand_m2
    integrals = integrate.trapezoid(integrands, m_int_range, axis=1)
    integrals = integrals[integrals!=0]
    
    return np.sum(np.log(integrals))


def population_posterior(hyper):

    # set only the 'mcmc_params' from the hyper array,
    # leave everything else to 'injected' value.
    hyper_mcmc = [alpha_inj, mmin_inj, mmax_inj]

    for i, mcmc_param in enumerate(mcmc_params):
        hyper_mcmc[mcmc_param] = hyper[i]

    alpha = hyper_mcmc[0]
    m1_min = hyper_mcmc[1]
    m1_max = hyper_mcmc[2]


    if (alpha>alpha_prior_high) or (alpha<alpha_prior_low):
        return -np.inf
    if (m1_min>mmin_prior_high) or (m1_min<mmin_prior_low):
        return -np.inf
    if (m1_max>mmax_prior_high) or (m1_max<mmax_prior_low):
        return -np.inf
    # if (beta>1.0) or (beta<-2.5):
    #     return -np.inf

    else:
        return lnprob(hyper_mcmc)
    

mcmc_params_list = ['alpha', 'm1_min', 'm1_max']
truths_all=[alpha_inj, mmin_inj, mmax_inj]
#priors_mcmc_low_all = [-4.2, 4.98, 55.0]
#priors_mcmc_high_all = [-3.8, 5.02, 65.0]
labels_all=[r"$\alpha$", r"$m_{\rm min}$", r"$m_{\rm max}$"]

# set which parameters to run MCMC for
fname = '/mcmc_'
for i in mcmc_params:
    fname += f'{mcmc_params_list[i]}_'

mcmc_file = output_dir + fname + f'N_{N_events}.h5'


priors_mcmc_low = []
priors_mcmc_high = []
truths = []
labels = []

for mcmc_param in mcmc_params:
    priors_mcmc_low.append(priors_mcmc_low_all[mcmc_param])
    priors_mcmc_high.append(priors_mcmc_high_all[mcmc_param])
    truths.append(truths_all[mcmc_param])
    labels.append(labels_all[mcmc_param])

ndim, nwalkers = len(mcmc_params), 2*len(mcmc_params)+1
#backend = emcee.backends.HDFBackend(mcmc_file)

#if reset_mcmc:
#    backend.reset(nwalkers, ndim)


p0 = np.random.uniform(low=priors_mcmc_low, high=priors_mcmc_high, size=(nwalkers,ndim))

reset_mcmc = True


with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
   
    backend = emcee.backends.HDFBackend(mcmc_file)

    if reset_mcmc:    
        backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, population_posterior, pool=pool, backend=backend)
    start = time.time()
    sampler.run_mcmc(p0, N_MCMC, progress=True)
    end = time.time()
    print()
    print(f"MCMC computed in {end - start:.2f} seconds")


