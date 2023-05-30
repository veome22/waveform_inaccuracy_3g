import numpy as np
import argparse
from scipy import interpolate, integrate
from astropy.cosmology import Planck18
import distributions as dist

parser = argparse.ArgumentParser(description='Generate a list of binaries sampled from a power law in m1, and uniform in q.')

parser.add_argument('-N', default="100000", type=int,  help='number of merger events to sample (default: 10000)')
parser.add_argument('-o', '--outfile',  default="~/scratch16-berti/3G_systematics_veome/data/smooth_powerlaw_pop.npz", type=str,  help='output file (default: ~/scratch16-berti/3G_systematics_veome/data/smooth_powerlaw_pop.npz)')

parser.add_argument('--mmin', default="5.0",  type=float, help='minimum mass in Solar Masses (default: 5.0)')
parser.add_argument('--mmin_lim', default="3.0",  type=float, help='minimum mass for smoothed power-law in Solar Masses (default: 3.0)')

parser.add_argument('--mmax', default="60.0",  type=float, help='maximum mass in Solar Mass (default: 60.0)')

parser.add_argument('--eta', default="50.0",  type=float, help='Butterworth filter smoothing parameter, to be applied at the low end of the primary mass population (default: 50.0)')

parser.add_argument('--zmin', default="0.02",  type=float, help='minimum redshift (default: 0.02)')
parser.add_argument('--zmax', default="50.0",  type=float, help='maximum redshift (default: 50.0)')

parser.add_argument('--alpha', default="-3.5",  type=float, help='power law exponent for m1 distribution (default: -3.5 bsaed on LIGO GWTC3 PP model)')

parser.add_argument('--chi_alpha', default="2.0",  type=float, help='alpha parameter for beta distribution for spins (default: 2.0 bsaed on LIGO GWTC3 DEFAULT model)')
parser.add_argument('--chi_beta', default="7.0",  type=float, help='beta parameter for beta distribution for spins (default: 7.0 bsaed on LIGO GWTC3 DEFAULT model)')


args = vars(parser.parse_args())

num_injs = args["N"]
outfile = args["outfile"]

m_min = args["mmin"]
m_min_lim = args["mmin_lim"]
m_max = args["mmax"]
m_eta = args["eta"]

z_min = args["zmin"]
z_max = args["zmax"]

alpha = args["alpha"]

chi_alpha = args["chi_alpha"]
chi_beta = args["chi_beta"]

seed=42


# SAMPLE M1
m1 = np.geomspace(m_min_lim, m_max, 1000000)
pdf_m1 = dist.butterworth(m1, m_min, m_eta) * dist.power(m1, alpha, m_min_lim, m_max)
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
pdf_chi = dist.beta(chi_range, chi_alpha, chi_beta) 
cdf_chi = integrate.cumulative_trapezoid(pdf_chi, chi_range, initial=0)
inv_cdf_chi = interpolate.interp1d(cdf_chi / cdf_chi[-1], chi_range)
chi1 = inv_cdf_chi(np.random.rand(num_injs))
chi2 = inv_cdf_chi(np.random.rand(num_injs))

# Sample spin orientations uniformly, but only select the z component
chi1z = chi1 * np.random.uniform(low=-1.0, high=1.0, size=num_injs)
chi2z = chi2 * np.random.uniform(low=-1.0, high=1.0, size=num_injs)



# SAMPLE REDSHIFTS
## old behavior: sample redshifts uniormly in [0.02, 50] based on Bohranian & Sathyaprakash (2022)
#redshifts = np.random.uniform(z_min, z_max, num_injs)
#DLs = Planck18.luminosity_distance(redshifts).value

# sample redshifts from Madau Fragos (2017) pdf
z_range = np.linspace(z_min, z_max, 1000000)
pdf_z = dist.p_z_madau_fragos(z_range, z_min, z_max)
cdf_z = integrate.cumulative_trapezoid(pdf_z, z_range, initial=0)
inv_cdf_z = interpolate.interp1d(cdf_z / cdf_z[-1], z_range)
redshifts = inv_cdf_z(np.random.rand(num_injs))
DLs = Planck18.luminosity_distance(redshifts).value



# Sample angles
iotas = np.arccos(np.random.uniform(low=-1, high=1, size=num_injs))
ras   = np.random.uniform(low=0., high=2.*np.pi, size=num_injs)
decs  = np.arccos(np.random.uniform(low=-1, high=1, size=num_injs)) - np.pi/2
psis  = np.random.uniform(low=0., high=2.*np.pi, size=num_injs)


# Convert source frame masses to detector frame masses
Mcs = Mcs * (1+redshifts)
mtotals = (mass1+mass2) * (1+redshifts)

np.savez(outfile, Mcs=Mcs, etas=etas, chi1z=chi1z, chi2z=chi2z, DLs=DLs, iotas=iotas, ras=ras, decs=decs, psis=psis)

print(f"{num_injs} binary parameters generated into {outfile}.")


