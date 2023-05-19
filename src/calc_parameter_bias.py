import numpy as np
import argparse
from gwbench import network
from gwbench import snr
import antenna_pattern_np as gw_ap
from pycbc.types import FrequencySeries
from pycbc.filter import match
from pycbc import pnutils
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

import matplotlib.pyplot as plt
import dill
import sys
import glob

import bias_funcs as bf
import gwbench_network_funcs as gwnet

parser = argparse.ArgumentParser(description='Compute the values of lambda and faithfulness above which Cutler-Vallisneri bias exceeds statistical error from Fisher.')

parser.add_argument('-N', default=None, type=int,  help='number of gwbench network objects to read (default: None, i.e. ALL the objects in inputdir)')
parser.add_argument('--offset', default=0, type=int,  help='offset for indices of gwbench network objects to read (default: 0)')

parser.add_argument('-o', '--outfile', default="../output/uniform_statistics_Mc.csv",  type=str, help='file to output the binary properties, threshold lambda, and threshold faithfulness (default: ../output/bias_uniform_statistics_Mc.csv)')
  
parser.add_argument('-i', '--inputdir', default="../data/uniform_networks_f_max",  type=str, help='input directory of network files (default: ../data/uniform_networks_f_max')

parser.add_argument('--suffix1', default="xas",  type=str, help='suffix name of network files for approximant 1 (default: xas')
parser.add_argument('--suffix2', default="d",  type=str, help='suffix name of network files for approximant 2 (default: d')


args = vars(parser.parse_args())

n_events = args["N"]
offset = args["offset"]
outfile = args["outfile"]
inputdir = args["inputdir"]

inputdir = args["inputdir"]
suffix1 = args["suffix1"]
suffix2 = args["suffix2"]

n_networks_all = len(glob.glob1(inputdir,f"*_{suffix1}_net"))

if n_events is None:
    n_events = n_networks_all

n_events = min(n_events, n_networks_all-offset)

print(f"Reading {n_events} network objects, starting from network {offset}")

deriv_symbs_string = 'Mc eta DL chi1z chi2z ra dec psi'
#deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'
param_list = deriv_symbs_string.split()


print("Calculating bias thresholds for parameters " + str(param_list))

# set up the empty arrays to store information for each event
inj_Mc = np.zeros(n_events)
inj_eta = np.zeros(n_events)
inj_m1 = np.zeros(n_events)
inj_m2 = np.zeros(n_events)
inj_chi1z = np.zeros(n_events)
inj_chi2z = np.zeros(n_events)
inj_DL = np.zeros(n_events)
inj_z = np.zeros(n_events)
inj_mtotal = np.zeros(n_events)
inj_q = np.zeros(n_events)
snrs = np.zeros(n_events)

inspiral_t = np.zeros(n_events)

stat_errs = np.zeros((n_events, len(param_list)))
full_biases = np.zeros((n_events, len(param_list)))
full_faiths = np.zeros(n_events)
full_inner_prods = np.zeros(n_events)

# MAIN LOOP
for i in range(n_events):
    print(f"Calculating bias for network {i+offset} ({i+1} of {n_events})")
    
    try:
        with open(inputdir + f'/{i+offset}_{suffix1}_net', "rb") as fi:
            net1 = dill.load(fi)
            fi.close() 

        with open(inputdir + f'/{i+offset}_{suffix2}_net', "rb") as fi:
            net2 = dill.load(fi)
            fi.close()
    except:
        print(inputdir + f'/{i+offset}_{suffix1}_net not found, skipping.')
        continue

    if net2 is None:
        print("NoneType object found, skipping.")
        continue

    # calculate Mtotal, q (calculations from pycbc)
    # (https://github.com/gwastro/pycbc/blob/master/pycbc/conversions.py)
    mchirp = net1.inj_params["Mc"]
    eta = net1.inj_params["eta"]
    mtotal = mchirp / (eta**(3./5.))
    m1 = 0.5 * mtotal * (1.0 + (1.0 - 4.0 * eta)**0.5)
    m2 = 0.5 * mtotal * (1.0 - (1.0 - 4.0 * eta)**0.5)
    q = max(m1, m2)/ min(m1, m2)

    inj_Mc[i] = mchirp
    inj_eta[i] = eta
    inj_m1[i] = m1
    inj_m2[i] = m2
    inj_chi1z[i] = net1.inj_params["chi1z"]
    inj_chi2z[i] = net1.inj_params["chi2z"]
    inj_DL[i] = net1.inj_params["DL"]
    inj_z[i] = z_at_value(Planck18.luminosity_distance, net1.inj_params["DL"] * u.Mpc)
    inj_mtotal[i] = mtotal
    inj_q[i] = q
    
    snrs[i] = net2.snr

   
    stat_errs[i] = list(net2.errs.values())[:-1] 
   
   # Calculate the Theoretical Bias in Parameters based on Cutler-Vallisneri formalism
    full_biases[i] = bf.compute_wf_bias(net1, net2, param_list)

    
    
    # Calculate Faithfulness using PyCBC

    delta_f = net1.f[1] - net1.f[0]
    psd = FrequencySeries(net1.detectors[1].psd, delta_f=delta_f) # caluclate mismatch using any one detector PSD
    
    hp1_pyc = FrequencySeries(net1.hfp, delta_f=delta_f)
    hp2_pyc = FrequencySeries(net2.hfp, delta_f=delta_f)
    full_faith, index = match(hp1_pyc, hp2_pyc, psd=psd, low_frequency_cutoff=net1.f[0])

   ## print("getting hybrid waveform")
   # hp_hyb, hc_hyb = get_hyb_wf(net1.hfp, net1.hfc, net2.hfp, net2.hfc, max_lam)
   # hp_hyb_pyc = FrequencySeries(hp_hyb, delta_f=delta_f)
    

   ## print("calculating faithfulness the pycbc way")
   # min_faith, index = match(hp1_pyc, hp_hyb_pyc, psd=psd, low_frequency_cutoff=net1.f[0])
    
   # Compute the inner product (unoptimized faithfulness)   
    hp1_norm = np.sum((hp1_pyc * np.conjugate(hp1_pyc) / psd).data)
    hp2_norm = np.sum((hp2_pyc * np.conjugate(hp2_pyc) / psd).data)
    #hyb_norm = np.sum((hp_hyb_pyc * np.conjugate(hp_hyb_pyc) / psd).data)

    full_inner_prod = np.abs(np.sum((hp1_pyc * np.conjugate(hp2_pyc)/psd).data)) / np.abs(np.sqrt(hp1_norm*hp2_norm))

   # Compute the inspiral time in band using pycbc
    f_low = net1.f[0]
    ts_5hz,fs_5hz = pnutils.get_inspiral_tf(0.,inj_m1[i],inj_m2[i],inj_chi1z[i],inj_chi2z[i],f_low)
    inspiral_t[i] = -ts_5hz[0]

    full_faiths[i] = full_faith
    full_inner_prods[i] = full_inner_prod


print("Writing output to " + outfile)

# Store all the data in a DataFrame
df =  pd.DataFrame()

df['Mc'] = inj_Mc
df['eta'] = inj_eta

df['m1'] = inj_m1
df['m2'] = inj_m2
df['M_tot'] = inj_mtotal

df['q'] = inj_q

df['chi1z'] = inj_chi1z
df['chi2z'] = inj_chi2z

df['DL'] = inj_DL
df['z'] = inj_z

df['snr'] = snrs

df['inspiral_t'] = inspiral_t

param_bias_cols = [str(param)+'_full_bias' for param in param_list]
param_stat_err_cols = [str(param)+'_stat_err' for param in param_list]

df[param_bias_cols] = full_biases
df[param_stat_err_cols] = stat_errs

df['full_faith'] = full_faiths
df['full_inner_prod'] = full_inner_prods

df.to_csv(outfile, index=False)

