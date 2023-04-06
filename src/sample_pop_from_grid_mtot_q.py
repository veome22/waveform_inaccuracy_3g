import numpy as np
import argparse
from gwbench import injections
from gwbench import network
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
from gwbench import basic_relations as br

import dill
import sys
from mpi4py import MPI
import os
import time

def get_network_response(inj_params, f_max=1024., network_spec = ['CE2-40-CBO_C', 'CE2-20-CBO_S', 'ET_ET1', 'ET_ET2', 'ET_ET3'], approximant='IMRPhenomXAS', deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'):
    
    # initialize the network with the desired detectors
    net = network.Network(network_spec)

    # choose the desired waveform 
    wf_model_name = 'lal_bbh'
    # pass the chosen waveform to the network for initialization
    net.set_wf_vars(wf_model_name=wf_model_name, wf_other_var_dic = {'approximant': approximant})

    # pick the desired frequency range
    f_min = 5.
    #f_max = 1024.
    d_f = 2**-4
    f = np.arange(f_min, f_max, d_f)

    # choose whether to take Earth's rotation into account
    use_rot = 0

    # pass all these variables to the network
    net.set_net_vars(
        f=f, inj_params=inj_params,
        deriv_symbs_string=deriv_symbs_string,
        use_rot=use_rot
        )

    # compute the WF polarizations
    net.calc_wf_polarizations()
    # compute the WF polarizations and their derivatives
    net.calc_wf_polarizations_derivs_num()

    # setup antenna patterns, location phase factors, and PSDs
    net.setup_ant_pat_lpf_psds()

    # compute the detector responses
    net.calc_det_responses()
    # compute the detector responses and their derivatives
    net.calc_det_responses_derivs_num()

    # calculate the network and detector SNRs
    net.calc_snrs()

    # calculate the network and detector Fisher matrices, condition numbers,
    # covariance matrices, error estimates, and inversion errors
    net.calc_errors()

    # calculate the 90%-credible sky area (in deg)
    net.calc_sky_area_90()

    return net


def get_network_snr(inj_params, network_spec = ['aLIGO_H','aLIGO_L','aLIGO_V'], approximant='IMRPhenomXAS', deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi', cond_num=1e25):
    
    # initialize the network with the desired detectors
    net = network.Network(network_spec)

    # choose the desired waveform 
    wf_model_name = 'lal_bbh'
    # pass the chosen waveform to the network for initialization
    net.set_wf_vars(wf_model_name=wf_model_name, wf_other_var_dic = {'approximant': approximant})

    # pick the desired frequency range
    f_min = 5.
    f_max = 512.
    d_f = 2**-4
    f = np.arange(f_min, f_max, d_f)

    # choose whether to take Earth's rotation into account
    use_rot = 0

    # pass all these variables to the network
    net.set_net_vars(
        f=f, inj_params=inj_params,
        deriv_symbs_string=deriv_symbs_string,
        use_rot=use_rot
        )
    
    # setup antenna patterns, location phase factors, and PSDs
    net.setup_ant_pat_lpf_psds()

    # compute the detector responses
    net.calc_det_responses()

    # calculate the network and detector SNRs
    net.calc_snrs()

    return net

parser = argparse.ArgumentParser(description='Generate a list of binaries sampled from a uniform grid in Mc and eta.')

parser.add_argument('-N', default="10", type=int,  help='number of bins in the mass-ratio (q) space (default: 10)')
parser.add_argument('-o', '--outputdir',  default="../data/uniform_grid_m1_q", type=str,  help='directory of output networks (default: ../data/uniform_grid_m1_q)')

parser.add_argument('--mtot', default="10.0",  type=float, help='total binary mass in Solar Mass (default: 10.0)')
parser.add_argument('--qmin', default="0.01",  type=float, help='minimum mass ratio (q) (default: 0.01)')
parser.add_argument('--qmax', default="0.99",  type=float, help='maximum mass ratio (q) (default: 0.99)')

parser.add_argument('--DL', default="400.0",  type=float, help='luminosity distance of event in Mpc (default: 400.0, modeled after GW150914)')
parser.add_argument('--SNR', default="None",  type=float, help='SNR of event in Mpc. Overrides the DL argument if specified. (default: None')

parser.add_argument('--offset', default="0",  type=int, help='starting index offset')

parser.add_argument('--approximant1', default="IMRPhenomXAS",  type=str, help='Approximant to use for reference waveform')
parser.add_argument('--approximant2', default="IMRPhenomD",  type=str, help='Approximant to use for other waveform')

parser.add_argument('--suffix1', default="xas",  type=str, help='filename suffix to use for reference waveform')
parser.add_argument('--suffix2', default="d",  type=str, help='filename suffix to use for other waveform')


args = vars(parser.parse_args())
# print(args)

n_q = args["N"]
output_path = args["outputdir"]

m_tot = args["mtot"]
q_min = args["qmin"]
q_max = args["qmax"]

DL = args["DL"]
target_snr = args["SNR"]

offset = args["offset"]

approximant1 = args["approximant1"]
approximant2 = args["approximant2"]

suffix1 = args["suffix1"]
suffix2 = args["suffix2"]

seed=42

#redshift = z_at_value(Planck18.luminosity_distance, DL * u.Mpc)

q_range = np.geomspace(q_min, q_max, num=n_q)
mass1 = m_tot/(q_range+1.0)
mass2 = m_tot * (q_range/(q_range+1.0))
    
Mcs = (mass1*mass2)**(3/5) / (mass1+mass2)**(1/5) 
etas = (mass1*mass2) / (mass1+mass2)**2

## Convert source frame masses to detector frame masses
#Mcs = Mcs * (1+redshift)
mtotals = (mass1+mass2) #* (1+redshift)

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

    for i, task in enumerate(range(n_q)):
        
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
            'DL':    DL,
            'tc':    0,
            'phic':  0,
            'iota':  np.pi/3,
            'ra':    np.pi/3,
            'dec':   np.pi/3,
            'psi':   np.pi/3,
            'gmst0': 0
            } 

        sys.stdout.write("Event number %d (%d) being simulated by processor %d of %d\n" % (i, task, rank, size))

        sys.stdout.write(f"Mc: {Mcs[i]:.2f}, eta: {etas[i]:.2f}\n")
        
        # Make sure the distance is set to achieve target SNR
        if target_snr is not None:
            # get the fiducial snr at DL
            net1_snr = get_network_snr(inj_params=inj_params, network_spec=network_spec, approximant=approximant1)
            
            # calculate DL required to hit target_snr
            new_DL = DL * (net1_snr.snr / target_snr)
            
            # adjust injected DL as needed
            inj_params['DL'] = new_DL


        net2 = get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant=approximant2)

        if net2.cov is None:
            sys.stdout.write(f"Matrix not invertible for Mc={Mcs[i]:.2f}, eta={etas[i]:.2f}, writing empty file\n.")
            with open(f'{output_path}/{offset+i}_{suffix1}_net', "wb") as fi:
                dill.dump(None, fi)
            with open(f'{output_path}/{offset+i}_{suffix2}_net', "wb") as fi:
                dill.dump(None, fi)
        else:
            net1 = get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant=approximant1)
            net1.save_network(f'{output_path}/{offset+i}_{suffix1}_net')
            net2.save_network(f'{output_path}/{offset+i}_{suffix2}_net')    
    
    end = time.time()

    comm.Barrier()
    if rank == 0:
      print('Done')
      print("Execution time {} seconds".format(end - start))


