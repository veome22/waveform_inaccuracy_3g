import numpy as np
import argparse
from gwbench import injections
from gwbench import network
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
from gwbench import basic_relations as br
from gwbench import fisher_analysis_tools as fat


from scipy import interpolate, integrate
from scipy.optimize import minimize

import dill
import sys
from mpi4py import MPI
import os
import time

from pycbc.types import FrequencySeries
from pycbc.filter import optimized_match

import gwbench_network_funcs as gwnet

parser = argparse.ArgumentParser(description='Simulate binaries from a list and compute the Waveform Biases.')

parser.add_argument('-N', default="48", type=int,  help='number of merger events to sample (default: 48)')
parser.add_argument('-i', '--input',  default="../data/smooth_powerlaw_pop.npz", type=str,  help='input binary parameters file (default: ../data/smooth_powerlaw_pop.npz)')

parser.add_argument('-o', '--outputdir',  default="../output/powerlaw_smooth_hybrid_3G/", type=str,  help='directory of output (default: ../output/powerlaw_smooth_hybrid_3G)')

parser.add_argument('--offset', default="0",  type=int, help='starting index offset')

parser.add_argument('--approx1', default="IMRPhenomXAS",  type=str, help='waveform approximant to use as true waveform (default: IMRPhenomXAS)')
parser.add_argument('--approx2', default="IMRPhenomD",  type=str, help='waveform approximant to use as approximate waveform (default: IMRPhenomD)')

parser.add_argument('--hybr', default="0.0",  type=float, help='hybrid waveform tuning parameter, defined from 0 (approximate) to 1 (true) (default: 0.0)')

parser.add_argument('--net_key', default="3G",  type=str, help='network to compute bias over (default: 3G)')

parser.add_argument('--align_waveforms', default="True",  type=bool, help='Align the coalescence time and phase of the waveforms to maximize overlap before computing biases?')

parser.add_argument('--spin_priors', default="True",  type=bool, help='Impose prior ranges on the spin parameters?')

args = vars(parser.parse_args())

num_injs = args["N"]
input_file = args["input"]
output_dir = args["outputdir"]

offset = args["offset"]

approx1 = args["approx1"]
approx2= args["approx2"]
hybr = args["hybr"]

net_key = args["net_key"]

align_wfs = args["align_waveforms"]
spin_priors = args["spin_priors"]

output_path = output_dir + f'/hybr_{hybr}/' 

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)


def cutler_vallisneri_overlap_vec(del_hf, delta_hf, psd, freqs):
    return np.array([ inner_prod_simps(del_hf[deriv], delta_hf, psd, freqs) for deriv in del_hf ])

def inner_prod_simps_normed(h1, h2, Sn, f, h1_norm = None, h2_norm=None):
    if h1_norm is None:
        h1_norm = 4*np.real(integrate.simpson(y= h1*np.conjugate(h1) / Sn, x=f))
    if h2_norm is None:
        h2_norm = 4*np.real(integrate.simpson(y= h2*np.conjugate(h2) / Sn, x=f))
    return  2*np.real(integrate.simpson(y= (h1*np.conjugate(h2) + h2*np.conjugate(h1)) / Sn, x=f)) / (np.sqrt(h1_norm * h2_norm))

def inner_prod_simps(h1, h2, Sn, f):
    return  2*np.real(integrate.simpson(y= (h1*np.conjugate(h2) + h2*np.conjugate(h1)) / Sn, x=f))



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.time()
    
    # Read the binary list ONLY ONCE, broadcast to the other processes
    if rank==0:
        with np.load(input_file, mmap_mode='r') as binaries:
            Mcs = np.array(binaries['Mcs'])
            etas = np.array(binaries['etas'])
            chi1z = np.array(binaries['chi1z'])
            chi2z = np.array(binaries['chi2z'])
            DLs = np.array(binaries['DLs'])
            iotas = np.array(binaries['iotas'])
            ras = np.array(binaries['ras'])
            decs = np.array(binaries['decs'])
            psis = np.array(binaries['psis'])

        print("Binaries successfully loaded from", input_file)

        sys.stdout.write(f"\n Simulating binaries {offset} to {offset+num_injs}" + output_path + "\n\n")
        
    else:
        Mcs = None
        etas = None
        chi1z = None
        chi2z = None
        DLs = None
        iotas = None
        ras = None
        decs = None
        psis = None

        
    Mcs = comm.bcast(Mcs, root=0)
    etas = comm.bcast(etas, root=0)
    chi1z = comm.bcast(chi1z, root=0)
    chi2z = comm.bcast(chi2z, root=0)
    DLs = comm.bcast(DLs, root=0)
    iotas = comm.bcast(iotas, root=0)
    ras = comm.bcast(ras, root=0)
    decs = comm.bcast(decs, root=0)
    psis = comm.bcast(psis, root=0)


    # Split binaries over the processes
    for i, task in enumerate(range(num_injs)):
        
        if i%size!=rank: continue
           
        outfile = output_path +  f"hybr_{hybr:.3f}_bin_{i+offset}"
        
        print("\n Binary number %d (%d) being simulated by processor %d of %d\n" % (i+offset, task, rank, size))
        print(f"Mc: {Mcs[i+offset]:.2f}, eta: {etas[i+offset]:.2f}\n")
    
        mtotals = br.M_of_Mc_eta(Mcs, etas)
        f_highs = np.round(4*br.f_isco_Msolar(mtotals))

        inj_params = {
            'Mc':    Mcs[i+offset],
            'eta':   etas[i+offset],
            'chi1x': 0.,
            'chi2x': 0.,
            'chi1y': 0.,
            'chi2y': 0.,
            'chi1z': chi1z[i+offset],
            'chi2z': chi2z[i+offset],
            'DL':    DLs[i+offset],
            'tc':    0,
            'phic':  0,
            'iota':  iotas[i+offset],
            'ra':    ras[i+offset],
            'dec':   decs[i+offset],
            'psi':   psis[i+offset],
            'gmst0': 0,
            'hybr': hybr
            } 

        
        net_true = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i+offset],      
                approximant=approx1, network_key=net_key, calc_detector_responses=True, calc_derivs=False, calc_fisher=False)


        net_ap_hyb = gwnet.get_hybrid_network_response(inj_params=inj_params, f_max=f_highs[i+offset], 
            approximant1=approx1, approximant2=approx2, 
            network_key=net_key, calc_detector_responses=True, calc_derivs=True, calc_fisher=True)

        if spin_priors:
            chi_prior_range = 1.0

            fisher_bounds = np.zeros_like(net_ap_hyb.fisher)
            fisher_bounds[chi1_idx,chi1_idx] = 1.0/chi_prior_range**2
            fisher_bounds[chi2_idx,chi2_idx] = 1.0/chi_prior_range**2
    
            net_ap_hyb_fisher = net_ap_hyb.fisher + fisher_bounds

            net_ap_hyb_cov = np.linalg.inv(net_ap_fisher_reg)
            net_ap_hyb_cov   = (net_ap_hyb_cov + net_ap_hyb_cov.T) / 2

            net_ap_hyb_errs = fat.get_errs_from_cov(net_ap_hyb_cov, net_ap_hyb.deriv_variables)
        
        else:
            net_ap_hyb_fisher = net_ap_hyb.fisher
            net_ap_hyb_cov = net_ap_hyb.cov
            net_ap_hyb_errs = net_ap_hyb.errs

        overlap_vecs_network = np.zeros((len(net_ap_hyb.detectors), len(net_ap_hyb.deriv_variables)))

        try:
            if align_wfs:
                # Find optimal phic and tc using matched filter
                # Limit the time window to precisely search for t_0
                time_arr_d = np.linspace(-0.2, 0.8, 21001)

                for d in range(len(net_ap_hyb.detectors)):
                    ## set up initial waveforms
                    h1 = net_true.detectors[d].hf
                    h2 = net_ap_hyb.detectors[d].hf
                    Sn = net_ap_hyb.detectors[d].psd
                    f = net_ap_hyb.detectors[d].f
                    network_spec_d = [net_ap_hyb.detectors[d].det_key]

                    # Set up Matched Filter 
                    x_t0_re_d = np.zeros(len(time_arr_d))
                    x_t0_im_d = np.zeros(len(time_arr_d))

                    for i_t in range(len(time_arr_d)):
                        t0 = time_arr_d[i_t]
                        x_t0_d = 4*(integrate.simpson(h1 * np.conjugate(h2) * np.exp(2*np.pi*1j*f*t0)/ Sn, x=f))
                        x_t0_re_d[i_t] = np.real(x_t0_d)
                        x_t0_im_d[i_t] = np.imag(x_t0_d)

                    # Find time that maximizes overlap
                    max_idx = np.argmax(x_t0_re_d**2 + x_t0_im_d**2)
                    time_shift_d = time_arr_d[max_idx]
                    phase_shift_d = np.unwrap(np.angle(x_t0_re_d + 1j*x_t0_im_d))[max_idx]

                    inj_params_opt_d = inj_params.copy()
                    inj_params_opt_d['tc'] = time_shift_d
                    inj_params_opt_d['phic'] = phase_shift_d 

                    net_tr_opt_d = gwnet.get_network_response(inj_params=inj_params_opt_d, f_max=f_highs[i+offset], 
                    approximant=approx1,
                    network_spec=network_spec_d, calc_detector_responses=True, calc_derivs=False, calc_fisher=False)

                    print("Inner product:")
                    print(inner_prod_simps_normed(net_tr_opt_d.detectors[0].hf, h2, Sn, f))

                    if d==0: # save overlap for the first detector
                        faith = inner_prod_simps_normed(net_tr_opt_d.detectors[0].hf, h2, Sn, f)

                    ## Compute CV overlap vector for this detector
                    delta_hf = net_tr_opt_d.detectors[0].hf - h2
                    overlap_vecs_network[d] = cutler_vallisneri_overlap_vec(net_ap_hyb.detectors[d].del_hf, delta_hf, Sn, f)

                cv_bias = np.matmul(net_ap_hyb_cov, np.sum(overlap_vecs_network, axis=0))


            else:
                for d in range(len(net_ap_hyb.detectors)):
                    ## set up initial waveforms
                    h1 = net_true.detectors[d].hf
                    h2 = net_ap_hyb.detectors[d].hf
                    Sn = net_ap_hyb.detectors[d].psd
                    f = net_ap_hyb.detectors[d].f

                    if d==0: # save overlap for the first detector
                        faith = inner_prod_simps_normed(h1, h2, Sn, f)

                    ## Compute CV overlap vector for this detector
                    delta_hf = h1 - h2
                    overlap_vecs_network[d] = cutler_vallisneri_overlap_vec(net_ap_hyb.detectors[d].del_hf, delta_hf, Sn, f)


                cv_bias = np.matmul(net_ap_hyb_cov, np.sum(overlap_vecs_network, axis=0))

        
        except:
            print(f"\n\n Error while computing network {i+offset}, with inj_params = {inj_params}, Skipping... \n\n")
            continue

        d = 0
        h1 = net_true.detectors[d].hf
        h2 = net_ap_hyb.detectors[d].hf
        Sn = net_ap_hyb.detectors[d].psd
        f = net_ap_hyb.detectors[d].f
        network_spec_d = [net_ap_hyb.detectors[d].det_key]
        inner_prod = inner_prod_simps_normed(h1, h2, Sn, f)

        print(cv_bias)

        # Save binary parameters, statistical errors, waveform bias, mismatch, inner product
        np.savez(outfile, inj_params=net_ap_hyb.inj_params, errs=net_ap_hyb_errs, cov=net_ap_hyb.cov, cv_bias=cv_bias, snr=net_ap_hyb.snr, faith=faith, inner_prod=inner_prod,\
                #z_inj=z_inj, z_err=z_err, z_bias=z_bias, \
                index=i+offset)


    end = time.time()

    comm.Barrier()
    
    if rank == 0:
      print('Done')
      print("Execution time {} seconds".format(end - start))


