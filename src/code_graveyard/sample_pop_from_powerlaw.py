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


def p_m1(m1, alpha, m_min, m_max):
    '''
    BBH merger primary mass PDF.
    '''
    if alpha != -1:
        N1 = 1 / ((m_max**(alpha+1) - m_min**(alpha+1))/(alpha+1))
    else:
        N1 = 1/(np.log(m_max/m_min))

    return np.piecewise(m1, [(m1 < m_min), (m1 >= m_min)*(m1 < m_max), (m1 >= m_max)],
                        [0, lambda m1: N1*m1**alpha, 0])

def inv_cdf_analytic(c, alpha, mmin, mmax):
    '''
    Analytically computed inverse CDF for the power law (for sampling m1)
    alpha is defined as in the pdf above, such that p=m1^alpha
    '''
    term1 = np.power(mmax, 1+alpha) - np.power(mmin, 1+alpha)
    term2 = np.power(mmin, 1+alpha)

    return np.power((c*term1 + term2), (1/(1+alpha)))



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
parser.add_argument('--mmax', default="100.0",  type=float, help='maximum mass in Solar Mass (default: 100.0)')

parser.add_argument('--zmin', default="0.02",  type=float, help='minimum redshift (default: 0.02)')
parser.add_argument('--zmax', default="50.0",  type=float, help='maximum redshift (default: 50.0)')

parser.add_argument('--alpha', default="-3.5",  type=float, help='power law exponent for m1 distribution (default: -3.5 bsaed on LIGO GWTC3 PP model)')


parser.add_argument('--qmin', default="0.01",  type=float, help='minimum mass ratio (q) (default: 0.01)')
parser.add_argument('--qmax', default="0.99",  type=float, help='maximum mass ratio (q) (default: 0.99)')


parser.add_argument('--offset', default="0",  type=int, help='starting index offset')

args = vars(parser.parse_args())
# print(args)

num_injs = args["N"]
output_path = args["outputdir"]

m_min = args["mmin"]
m_max = args["mmax"]

z_min = args["zmin"]
z_max = args["zmax"]

alpha = args["alpha"]

q_min = args["qmin"]
q_max = args["qmax"]

offset = args["offset"]
   
seed=42


# SAMPLE REDSHIFTS
## old behavior: sample redshifts uniormly in [0.02, 50] based on Bohranian & Sathyaprakash (2022)
#redshifts = np.random.uniform(z_min, z_max, num_injs) 
#DLs = Planck18.luminosity_distance(redshifts).value

# sample redshifts from Madau Fragos (2017) pdf
z_range = np.linspace(z_min, z_max, 10000)
pdf_z = p_z_madau_fragos(z_range, z_min, z_max)
cdf_z = integrate.cumulative_trapezoid(pdf_z, z_range, initial=0)
inv_cdf_z = interpolate.interp1d(cdf_z / cdf_z[-1], z_range)
redshifts = inv_cdf_z(np.random.rand(num_injs))
DLs = Planck18.luminosity_distance(redshifts).value


# SAMPLE M1
# sample mass 1 from numerical power law pdf with pre-defined bins
m1 = np.geomspace(m_min, m_max, 1000000)
pdf_m1 = p_m1(m1, alpha, m_min, m_max)
cdf_m1 = integrate.cumulative_trapezoid(pdf_m1, m1, initial=0)
inv_cdf_m1 = interpolate.interp1d(cdf_m1 / cdf_m1[-1], m1)
mass1 = inv_cdf_m1(np.random.rand(num_injs))

# # sample m1 from analytic formula
# mass1 = inv_cdf_analytic(np.random.rand(num_injs), alpha, m_min, m_max)


# SAMPLE M2
# sample q uniformly between q_min and q_max
#q = np.random.uniform(q_min, q_max, num_injs)
#mass2 = mass1 * q
mass2 = np.random.uniform(low=m_min, high=mass1)
    
Mcs = (mass1*mass2)**(3/5) / (mass1+mass2)**(1/5) 
etas = (mass1*mass2) / (mass1+mass2)**2

# Sample angles
iotas, ras, decs, psis = injections.angle_sampler(num_injs, seed)


# Convert source frame masses to detector frame masses
Mcs = Mcs * (1+redshifts)
mtotals = (mass1+mass2) * (1+redshifts)



f_highs = np.round(4*br.f_isco_Msolar(mtotals))


deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'
param_list = deriv_symbs_string.split()


if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.time()

    for i, task in enumerate(range(num_injs)):
        
        if i%size!=rank: continue
        
        inj_params = {
            'Mc':    Mcs[i],
            'eta':   etas[i],
            'chi1x': 0.,
            'chi2x': 0.,
            'chi1y': 0.,
            'chi2y': 0.,
            'chi1z': 0.,
            'chi2z': 0.,
            'DL':    DLs[i],
            'tc':    0,
            'phic':  0,
            'iota':  iotas[i],
            'ra':    ras[i],
            'dec':   decs[i],
            'psi':   psis[i],
            'gmst0': 0
            } 

        sys.stdout.write("Event number %d (%d) being simulated by processor %d of %d\n" % (i, task, rank, size))

        sys.stdout.write(f"Mc: {Mcs[i]:.2f}, eta: {etas[i]:.2f}")
        
        net2 = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant='IMRPhenomD')

        if net2.cov is None:
            sys.stdout.write(f"Matrix not invertible for Mc={Mcs[i]:.2f}, eta={etas[i]:.2f}, writing empty file\n.")
            with open(f'{output_path}/{offset+i}_xas_net', "wb") as fi:
                dill.dump(None, fi)
            with open(f'{output_path}/{offset+i}_d_net', "wb") as fi:
                dill.dump(None, fi)
        else:        
            net1 = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant='IMRPhenomXAS')
            net1.save_network(f'{output_path}/{offset+i}_xas_net')
            net2.save_network(f'{output_path}/{offset+i}_d_net')    
    
    end = time.time()

    comm.Barrier()
    if rank == 0:
      print('Done')
      print("Execution time {} seconds".format(end - start))


