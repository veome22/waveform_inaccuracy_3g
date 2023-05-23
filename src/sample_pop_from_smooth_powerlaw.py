import numpy as np
import argparse
from gwbench import injections
from gwbench import network
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
from gwbench import basic_relations as br
from scipy import interpolate, integrate
import dill
import sys
from mpi4py import MPI
import os
import time

import gwbench_network_funcs as gwnet



def power(m1, alpha, m_min, m_max):
    '''
    BBH merger primary mass PDF.
    '''
    if alpha != -1:
        N1 = 1 / ((m_max**(alpha+1) - m_min**(alpha+1))/(alpha+1))
    else:
        N1 = 1/(np.log(m_max/m_min))

    return np.piecewise(m1, [(m1 < m_min), (m1 >= m_min)*(m1 < m_max), (m1 >= m_max)],
                        [0, lambda m1: N1*m1**alpha, 0])



def butterworth(m1, m0, eta):
    norm = integrate.trapezoid((1+ (m0/m1)**eta)**(-1), m1)

    return (1+ (m0/m1)**eta)**(-1) / norm


def inv_cdf_analytic(c, alpha, mmin, mmax):
    '''
    Analytically computed inverse CDF for the power law (for sampling m1)
    alpha is defined as in the pdf above, such that p=m1^alpha
    '''
    term1 = np.power(mmax, 1+alpha) - np.power(mmin, 1+alpha)
    term2 = np.power(mmin, 1+alpha)

    return np.power((c*term1 + term2), (1/(1+alpha)))


def beta(chi, a, b):
    func = chi**(a-1) * (1-chi)**(b-1)
    norm = integrate.trapezoid(func, chi)
    return func/norm


def p_z_madau_fragos(z, z_min, z_max):
    '''
    Return normalized z-distribution from the Madau Fragos (2017)
    star formation rate density.
    '''
    term_1 = (1+z)**(2.6)
    term_2 = 1 + ((1+z)/3.2)**(6.2)

    psi = 0.01 * term_1/term_2
    norm = np.sum(psi)
    psi = psi/norm

    return psi

parser = argparse.ArgumentParser(description='Generate a list of binaries sampled from a power law in m1, and uniform in q.')

parser.add_argument('-N', default="48", type=int,  help='number of merger events to sample (default: 48)')
parser.add_argument('-o', '--outputdir',  default="../data/powerlaw_3.5", type=str,  help='directory of output networks (default: ../data/powerlaw_3.5)')

parser.add_argument('--mmin', default="5.0",  type=float, help='minimum mass in Solar Masses (default: 5.0)')
parser.add_argument('--mmin_lim', default="3.0",  type=float, help='minimum mass for smoothed power-law in Solar Masses (default: 3.0)')

parser.add_argument('--mmax', default="60.0",  type=float, help='maximum mass in Solar Mass (default: 60.0)')

parser.add_argument('--eta', default="50.0",  type=float, help='Butterworth filter smoothing parameter, to be applied at the low end of the primary mass population (default: 50.0)')

parser.add_argument('--zmin', default="0.02",  type=float, help='minimum redshift (default: 0.02)')
parser.add_argument('--zmax', default="50.0",  type=float, help='maximum redshift (default: 50.0)')

parser.add_argument('--alpha', default="-3.5",  type=float, help='power law exponent for m1 distribution (default: -3.5 bsaed on LIGO GWTC3 PP model)')


parser.add_argument('--qmin', default="0.01",  type=float, help='minimum mass ratio (q) (default: 0.01)')
parser.add_argument('--qmax', default="0.99",  type=float, help='maximum mass ratio (q) (default: 0.99)')

parser.add_argument('--chi_alpha', default="2.0",  type=float, help='alpha parameter for beta distribution for spins (default: 2.0 bsaed on LIGO GWTC3 DEFAULT model)')

parser.add_argument('--chi_beta', default="7.0",  type=float, help='beta parameter for beta distribution for spins (default: 7.0 bsaed on LIGO GWTC3 DEFAULT model)')

parser.add_argument('--offset', default="0",  type=int, help='starting index offset')

parser.add_argument('--net_key', default="3G",  type=str, help='network to compute bias over (default: 3G)')

args = vars(parser.parse_args())
# print(args)

num_injs = args["N"]
output_path = args["outputdir"]

m_min = args["mmin"]
m_min_lim = args["mmin_lim"]
m_max = args["mmax"]
m_eta = args["eta"]

z_min = args["zmin"]
z_max = args["zmax"]

alpha = args["alpha"]

q_min = args["qmin"]
q_max = args["qmax"]

chi_alpha = args["chi_alpha"]
chi_beta = args["chi_beta"]

offset = args["offset"]
net_key = args["net_key"]

seed=42


# SAMPLE REDSHIFTS
## old behavior: sample redshifts uniormly in [0.02, 50] based on Bohranian & Sathyaprakash (2022)
#redshifts = np.random.uniform(z_min, z_max, num_injs) 
#DLs = Planck18.luminosity_distance(redshifts).value

# sample redshifts from Madau Fragos (2017) pdf
z_range = np.linspace(z_min, z_max, 100000)
pdf_z = p_z_madau_fragos(z_range, z_min, z_max)
cdf_z = integrate.cumulative_trapezoid(pdf_z, z_range, initial=0)
inv_cdf_z = interpolate.interp1d(cdf_z / cdf_z[-1], z_range)
redshifts = inv_cdf_z(np.random.rand(num_injs))
DLs = Planck18.luminosity_distance(redshifts).value


# SAMPLE M1
m1 = np.geomspace(m_min_lim, m_max, 1000000)
pdf_m1 = butterworth(m1, m_min, m_eta) * power(m1, alpha, m_min_lim, m_max)
pdf_m1 = pdf_m1/integrate.trapezoid(pdf_m1, m1) # make sure that the pdf is normalized
cdf_m1 = integrate.cumulative_trapezoid(pdf_m1, m1, initial=0)
inv_cdf_m1 = interpolate.interp1d(cdf_m1 / cdf_m1[-1], m1)
mass1 = inv_cdf_m1(np.random.rand(num_injs))


# SAMPLE M2
mass2 = np.random.uniform(low=m_min_lim, high=mass1)
    
Mcs = (mass1*mass2)**(3/5) / (mass1+mass2)**(1/5) 
etas = (mass1*mass2) / (mass1+mass2)**2

# SAMPLE Chi_1, Chi_2 (aligned)
chi_range = np.linspace(0, 1, 100000)
pdf_chi = beta(chi_range, chi_alpha, chi_beta) 
cdf_chi = integrate.cumulative_trapezoid(pdf_chi, chi_range, initial=0)
inv_cdf_chi = interpolate.interp1d(cdf_chi / cdf_chi[-1], chi_range)
chi1 = inv_cdf_chi(np.random.rand(num_injs))
chi2 = inv_cdf_chi(np.random.rand(num_injs))

# Sample spin orientations uniformly, but only select the z component
chi1z = chi1 * np.random.uniform(low=-1.0, high=1.0, size=num_injs)
chi2z = chi2 * np.random.uniform(low=-1.0, high=1.0, size=num_injs)

# Sample angles
iotas, ras, decs, psis = injections.angle_sampler(num_injs, seed)


# Convert source frame masses to detector frame masses
Mcs = Mcs * (1+redshifts)
mtotals = (mass1+mass2) * (1+redshifts)



f_highs = np.round(4*br.f_isco_Msolar(mtotals))


deriv_symbs_string = 'Mc eta DL chi1z chi2z iota ra dec psi'
param_list = deriv_symbs_string.split()


if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.time()
    
    if rank==0:
        sys.stdout.write("\n Simulating population into" + output_path + "\n\n")

    for i, task in enumerate(range(num_injs)):
        
        if i%size!=rank: continue
        
        inj_params = {
            'Mc':    Mcs[i],
            'eta':   etas[i],
            'chi1x': 0.,
            'chi2x': 0.,
            'chi1y': 0.,
            'chi2y': 0.,
            'chi1z': chi1z[i],
            'chi2z': chi2z[i],
            'DL':    DLs[i],
            'tc':    0,
            'phic':  0,
            'iota':  iotas[i],
            'ra':    ras[i],
            'dec':   decs[i],
            'psi':   psis[i],
            'gmst0': 0
            } 

        sys.stdout.write("\n Event number %d (%d) being simulated by processor %d of %d\n" % (i, task, rank, size))

        sys.stdout.write(f"Mc: {Mcs[i]:.2f}, eta: {etas[i]:.2f}")
        
        net2 = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant='IMRPhenomD', network_key=net_key)

        if net2.cov is None:
            sys.stdout.write(f"Matrix not invertible for Mc={Mcs[i]:.2f}, eta={etas[i]:.2f}, writing empty file\n.")
            with open(f'{output_path}/{offset+i}_xas_net', "wb") as fi:
                dill.dump(None, fi)
            with open(f'{output_path}/{offset+i}_d_net', "wb") as fi:
                dill.dump(None, fi)
        else:        
            net1 = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant='IMRPhenomXAS', network_key=net_key)
            net1.save_network(f'{output_path}/{offset+i}_xas_net')
            net2.save_network(f'{output_path}/{offset+i}_d_net')    
    
    end = time.time()

    comm.Barrier()
    if rank == 0:
      print('Done')
      print("Execution time {} seconds".format(end - start))


