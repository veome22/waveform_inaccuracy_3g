
from __future__ import division, print_function
import os
import sys

import numpy as np
from gwbench import basic_relations as br


bilby_path = os.environ['HOME']+'/bilby_3g'
if os.path.isdir(bilby_path):
  import sys
  sys.path.append(bilby_path)
  print('Added path: ', bilby_path)

import bilby

import bilby.gw.utils as gwutils

nlive=60
nact=1
maxmcmc=500
npool=56

m1src = 120.0
m2src= 110.0

theta_jn_deg=0.0
ra_deg=60.0
dec_deg=60.0
psi_deg=60.0
redshift=0.2
duration=np.round(4*br.f_isco_Msolar(m1src+m2src))
fsample=2048.0
fmin=20.
fref=20.

waveform='IMRPhenomXAS'
waveform_det='IMRPhenomXAS'

outdir=os.environ['WORK']+'/bilby_PE_waveform_systematics/'+f'm1_{m1src:.0f}_m2_{m2src:.0f}_{waveform}_{waveform_det}'

if not os.path.exists(outdir)
   # Create a new directory because it does not exist
   os.makedirs(outdir)

margPhase = False
seed = 1290643798
# ifo_list= ['CE', 'CES', 'ET1', 'ET2', 'ET3']
ifo_list= ['CE']



# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
sampling_frequency = fsample
sampler = 'dynesty'

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(seed)


# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
bilby.gw.cosmology._set_default_cosmology()
cosmo = bilby.gw.cosmology.DEFAULT_COSMOLOGY
mtotal = m1src+m2src
q=m2src/m1src
# print(q)
tilt_1=0.
tilt_2=0.
chi_1=0.
chi_2=0.
phi_12=0.
phi_jl=0.
z=redshift
dL=cosmo.luminosity_distance(z).value
theta_jn_rad = theta_jn_deg*np.pi/180.0
ra_rad = ra_deg*np.pi/180.0
dec_rad = dec_deg*np.pi/180.0
psi_rad = psi_deg*np.pi/180.0
phase=0.0
gps_time=1577491218.0
m1_det = m1src*(1+z)
m2_det = m2src*(1+z)
Mc_det = bilby.gw.conversion.component_masses_to_chirp_mass(m1_det, m2_det)
total_mass_det = m1_det+m2_det
approx = waveform
approx_det = waveform_det
label = 'mtotal%.1f_q%.1f_z%.1f_iota%.1f_%s' %(mtotal, 1./q, redshift, theta_jn_deg, approx)
bilby.core.utils.setup_logger(outdir=outdir, label=label)
fref=fref
fmin=fmin



# injection/simulation of a waveform
injection_parameters = dict(
	chirp_mass = Mc_det, mass_1=m1_det, mass_2=m2_det, chi_1=chi_1, chi_2=chi_2, #tilt_1=tilt_1, tilt_2=tilt_2,
	#phi_12=phi_12, phi_jl=phi_jl, 
	luminosity_distance=dL, theta_jn=theta_jn_rad, psi=psi_rad,
	phase=phase, geocent_time=gps_time, ra=ra_rad, dec=dec_rad, reference_frequency=fref, minimum_frequency=fmin)


# Fixed arguments passed into the source model
# approx: IMRPhenomPv2: standard, (2,2), IMRPhenomXPHM (l,m)=(2,2),...(4,4)
waveform_arguments = dict(waveform_approximant=approx, reference_frequency=fref, minimum_frequency=fmin)

waveform_generator = bilby.gw.WaveformGenerator(
	duration=duration, sampling_frequency=sampling_frequency,
	frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
	parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
	waveform_arguments=waveform_arguments,start_time=injection_parameters['geocent_time'] - duration + 2.0)


deltaT = gwutils.calculate_time_to_merger(
            frequency=1,
            mass_1=m1_det,
            mass_2=m2_det,
        )

deltaT = np.round(deltaT, 1)


# Set detection waveform model
waveform_arguments = dict(waveform_approximant=approx_det, reference_frequency=fref, minimum_frequency=fmin)


waveform_generator = bilby.gw.WaveformGenerator(
	duration=duration, sampling_frequency=sampling_frequency,
	frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
	parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
	waveform_arguments=waveform_arguments,start_time=injection_parameters['geocent_time'] - duration + 2.0)


# Set up interferometers.
ifos = bilby.gw.detector.InterferometerList(ifo_list)
for ifo in ifos:
	#ifo.set_strain_data_from_power_spectral_density(
	ifo.set_strain_data_from_zero_noise(
		sampling_frequency=sampling_frequency, duration=duration,
		start_time=injection_parameters['geocent_time'] - duration + 2.0)
	ifo.inject_signal_from_waveform_generator(parameters=injection_parameters, waveform_generator=waveform_generator)
	# ifo.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

# For this analysis, we implemenet the standard BBH priors defined, except for
# the definition of the time prior, which is defined as uniform about the
# injected value.
# Furthermore, we decide to sample in total mass and mass ratio, due to the
# preferred shape for the associated posterior distributions.
q_min=0.1
q_max=1.0
total_mass_min = total_mass_det*(0.25)
total_mass_max = total_mass_det*(2.5)
if approx=="IMRPhenomXPHM":
	if q>=0.8:
		chirp_mass_min = Mc_det*(0.75)
		chirp_mass_max = Mc_det*(1.25)
		total_mass_min = total_mass_det*(0.75)
		total_mass_max = total_mass_det*(1.25)
	if (q>=0.4)*(q<0.8):
		chirp_mass_min = Mc_det*(0.75)
		chirp_mass_max = Mc_det*(1.5)
		total_mass_min = total_mass_det*(0.75)
		total_mass_max = total_mass_det*(1.5)
	if q<0.4:
		chirp_mass_min = Mc_det*(0.5)
		chirp_mass_max = Mc_det*(2.0)
		total_mass_min = total_mass_det*(0.5)
		total_mass_max = total_mass_det*(2.0)
dist_min = dL/10.0
dist_max = dL*5.0
# We first output the prior ranges in a text file as a record tracker
BBHprior = \
"""mass_ratio = PowerLaw(name='mass_ratio', minimum=%f, maximum=%f, alpha=-2.0)
total_mass = Uniform(name='total_mass', minimum=%f, maximum=%f)
luminosity_distance = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=%f, maximum=%f)
dec = Cosine(name='dec')
ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
theta_jn = Sine(name='theta_jn')
psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
delta_phase = Uniform(name='delta_phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
chi_1 = Uniform(name='chi_1', minimum=0, maximum=0.99)
chi_2 = Uniform(name='chi_2', minimum=0, maximum=0.99)
geocent_time = Uniform(name='geocent_time', minimum=%f, maximum=%f)
""" %(q_min, q_max, total_mass_min, total_mass_max, dist_min, dist_max, gps_time-0.1, gps_time+0.1)

with open(outdir+'/%s.prior' %label,'w') as f_prior:
	f_prior.write(BBHprior)

priors = bilby.gw.prior.BBHPriorDict(outdir+'/%s.prior' %label)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator, as well the priors.
# The explicit time, distance, and phase marginalizations are turned on to
# improve convergence, and the parameters are recovered by the conversion
# function.
likelihood = bilby.gw.GravitationalWaveTransient(
	interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
	distance_marginalization=True, phase_marginalization=margPhase, time_marginalization=True, distance_marginalization_lookup_table=outdir+'/%s_dist_lookup.npz' %label)

result = bilby.run_sampler(
	likelihood=likelihood, priors=priors, sampler=sampler, nlive=nlive, nact=nact, maxmcmc=maxmcmc, npool=npool,
	injection_parameters=injection_parameters, outdir=outdir,
	label=label,
	conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)
