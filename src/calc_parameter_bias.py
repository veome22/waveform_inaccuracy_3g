import numpy as np
import argparse
from gwbench import network
from gwbench import snr
import antenna_pattern_np as gw_ap
from pycbc.types import FrequencySeries
from pycbc.filter import match
import pandas as pd

import matplotlib.pyplot as plt
import dill
import sys
import glob

parser = argparse.ArgumentParser(description='Compute the values of lambda and faithfulness above which Cutler-Vallisneri bias exceeds statistical error from Fisher.')

parser.add_argument('-N', default=None, type=int,  help='number of gwbench network objects to read (default: None, i.e. ALL the objects in inputdir)')
parser.add_argument('--offset', default=0, type=int,  help='offset for indices of gwbench network objects to read (default: 0)')

parser.add_argument('-p', '--parameter', default="0", type=int,  help='parameter to compute errors for (default: 0 : Mc)')

parser.add_argument('-o', '--outfile', default="../output/uniform_statistics_Mc.csv",  type=str, help='file to output the binary properties, threshold lambda, and threshold faithfulness (default: ../output/bias_uniform_statistics_Mc.csv)')
  
parser.add_argument('-i', '--inputdir', default="../data/uniform_networks",  type=str, help='input directory of network files (default: ../data/uniform_networks')


args = vars(parser.parse_args())
# print(args)

n_events = args["N"]
offset = args["offset"]
outfile = args["outfile"]
param_index = args["parameter"]
inputdir = args["inputdir"]

n_networks_all = len(glob.glob1(inputdir,"*_xas_net"))

if n_events is None:
    n_events = n_networks_all

n_events = min(n_events, n_networks_all-offset)

print(f"Reading {n_events} network objects, starting from network {offset}")

deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'
param_list = deriv_symbs_string.split()
network_spec = ['CE2-40-CBO_C', 'CE2-20-CBO_S', 'ET_ET1', 'ET_ET2', 'ET_ET3']

print("Calculating bias thresholds for parameter " + param_list[param_index])

# Hybrid Waveform
def get_hyb_wf(hp1, hc1, hp2, hc2, lam, opt_phi=False):

    if (lam > 1. or lam < 0):
        raise Exception('lambda should be between 0.0 and 1.0. The value of lambda was: {}'.format(lam))

    # h_plus components
    a1_p = np.abs(hp1)
    phi1_p = np.unwrap(np.angle(hp1))

    a2_p = np.abs(hp2)
    phi2_p = np.unwrap(np.angle(hp2))

    # h_cross components
    a1_c = np.abs(hc1)
    phi1_c = np.unwrap(np.angle(hc1))

    a2_c = np.abs(hc2)
    phi2_c = np.unwrap(np.angle(hc2))

    # Construct Hybrid waveforms
    a_hyb_p = a1_p*(1-lam) + a2_p*(lam)
    phi_hyb_p = phi1_p*(1-lam) + phi2_p*(lam)

    a_hyb_c = a1_c*(1-lam) + a2_c*(lam)
    phi_hyb_c = phi1_c*(1-lam) + phi2_c*(lam)


    hp_hyb =  a_hyb_p * np.exp(1.j * phi_hyb_p)
    hc_hyb =  a_hyb_c * np.exp(1.j * phi_hyb_c)

    return hp_hyb, hc_hyb

def waveform_to_det_response(hp, hc, inj_params, detector):
    Mc = inj_params["Mc"]
    tc = inj_params["tc"]
    ra = inj_params["ra"]
    dec = inj_params["dec"]
    psi = inj_params["psi"]
    gmst0 = inj_params["gmst0"]
    use_rot = 0

    f = detector.f
    loc = detector.loc

    Fp, Fc, Flp = gw_ap.antenna_pattern_and_loc_phase_fac(f,Mc,tc,ra,dec,psi,gmst0,loc,use_rot)
    return Flp * (Fp * hp + Fc * hc)


# Calculate Cutler-Vallisneri Biases in chirp mass across Lambda
# Interpolate the 'True' waveform for ease of calculation

#param_index = 0 # Mc
param = param_list[param_index]

inj_Mc = np.zeros(n_events)
inj_eta = np.zeros(n_events)
inj_DL = np.zeros(n_events)
inj_mtotal = np.zeros(n_events)
inj_q = np.zeros(n_events)
snrs = np.zeros(n_events)
stat_errs = np.zeros(n_events)
full_bias = np.zeros(n_events)
max_lams = np.zeros(n_events)
min_faiths = np.zeros(n_events)


# lams = np.linspace(0., 0.15, 100)
lams = np.logspace(-4., -0.5, 100)

errors_th_lam = np.zeros((n_events, len(lams)))

for i in range(n_events):
    print(f"Calculating bias for network {i+offset+1}")
    with open(inputdir + f'/{i+offset}_xas_net', "rb") as fi:
        net1 = dill.load(fi)
        fi.close() 

    with open(inputdir + f'/{i+offset}_d_net', "rb") as fi:
        net2 = dill.load(fi)
        fi.close()
    
    if net2.cov is None:
        print("NoneType object found, skipping.")
        continue

    snrs[i] = net1.snr

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
    inj_DL[i] = net1.inj_params["DL"]
    inj_mtotal[i] = mtotal
    inj_q[i] = q

    cov_ap = net2.cov
    
    
    # Calculate the Theoretical Bias in Parameters based on Cutler-Vallisneri formalism
    inner_prod = np.zeros(len(param_list))

    for d in range(len(network_spec)):
        del_h_ap_all = net2.detectors[d].del_hf
        del_params_j = list(del_h_ap_all.keys())

        h_tr = net1.detectors[d].hf
        h_ap = net2.detectors[d].hf
        psd = net2.detectors[d].psd
        freq_range = net2.detectors[d].f
        df = freq_range[1] - freq_range[0]

        for j, parameter_j in enumerate(del_params_j):
            del_h_ap_j = del_h_ap_all[parameter_j]
            # Inner Product
            inner_prod[j] += snr.scalar_product_freq_array(del_h_ap_j, h_tr - h_ap, psd, freq_range, df)


        # Calculate the theoretical bias between IMRPhenomD and IMRPhenomXAS
        full_bias[i] = np.dot(cov_ap, inner_prod)[param_index]


    for l, lam in enumerate(lams):
        inner_prod = np.zeros(len(param_list))

        for d in range(len(network_spec)):
            del_h_ap_all = net2.detectors[d].del_hf
            del_params_j = list(del_h_ap_all.keys())

            hp_hyb, hc_hyb = get_hyb_wf(net2.hfp, net2.hfc, net1.hfp, net1.hfc, lam)

            h_tr = waveform_to_det_response(hp_hyb, hc_hyb, net1.inj_params, net1.detectors[d])
            h_ap = net2.detectors[d].hf

            psd = net2.detectors[d].psd
            freq_range = net2.detectors[d].f
            df = freq_range[1] - freq_range[0]

            for j, parameter_j in enumerate(del_params_j):
                del_h_ap_j = del_h_ap_all[parameter_j]  
                # Inner Product
                inner_prod[j] += snr.scalar_product_freq_array(del_h_ap_j, h_tr - h_ap, psd, freq_range, df)
            
        # Calculate the theoretical bias for chosen parameter
        errors_th_lam[i,l] = np.dot(cov_ap, inner_prod)[param_index]
        
        # Stop calculating biases for higher lambda if Statistical error has already been exceeded.
        if errors_th_lam[i,l] > net2.errs[param]:
            errors_th_lam[i, l:] = errors_th_lam[i,l]
            break

    sigma_mc = np.abs(net2.errs[param])
    bias_mc = np.abs(errors_th_lam[i,:])

    try:
        max_lam = lams[np.where(bias_mc <= sigma_mc)[0][-1]]
    except:
        max_lam = lams[0]

    #print(f"Max lambda for M={mtotal:.1f}, q={q:.1f} : {max_lam:.3f}")

    delta_f = net1.f[1] - net1.f[0]
    
   # print("psd step")
    psd = FrequencySeries(net1.detectors[1].psd, delta_f=delta_f) # caluclate mismatch using any one detector PSD
    
    #print("getting pycbc waveform 1")
    hp1_pyc = FrequencySeries(net1.hfp, delta_f=delta_f)

   # print("getting hybrid waveform")
    hp_hyb, hc_hyb = get_hyb_wf(net1.hfp, net1.hfc, net2.hfp, net2.hfc, max_lam)
    hp_hyb_pyc = FrequencySeries(hp_hyb, delta_f=delta_f)
    

   # print("calculating faithfulness the pycbc way")
    min_faith, index = match(hp1_pyc, hp_hyb_pyc, psd=psd, low_frequency_cutoff=net1.f[0])
    

    stat_errs[i] = sigma_mc
    max_lams[i] = max_lam
    min_faiths[i] = min_faith

    #print(f"Min faithfulness for M={mtotal:.1f}, q={q:.1f} = {min_faith:.6f}\n")

print("Writing output to " + outfile)


df =  pd.DataFrame()

df['Mc'] = inj_Mc
df['eta'] = inj_eta
df['DL'] = inj_DL
df['M_tot'] = inj_mtotal
df['q'] = inj_q
df['full_bias'] = full_bias
df['snr'] = snrs
df[param+'_stat_err'] = stat_errs
df[param+'_max_lam'] = max_lams
df[param+'_min_faith'] = min_faiths

df.to_csv(outfile, index=False)

