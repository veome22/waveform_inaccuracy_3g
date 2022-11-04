import numpy as np
import argparse
from gwbench import injections
from gwbench import network

import sys
from mpi4py import MPI
import os
import time

def get_network_response(inj_params, network_spec = ['CE2-40-CBO_C', 'CE2-20-CBO_S', 'ET_ET1', 'ET_ET2', 'ET_ET3'], approximant='IMRPhenomXAS', deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'):
    
    # initialize the network with the desired detectors
    net = network.Network(network_spec)

    # choose the desired waveform 
    wf_model_name = 'lal_bbh'
    # pass the chosen waveform to the network for initialization
    net.set_wf_vars(wf_model_name=wf_model_name, wf_other_var_dic = {'approximant': approximant})

    # pick the desired frequency range
    f_min = 5.
    f_max = 1024.
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




parser = argparse.ArgumentParser(description='Generate a list of binaries sampled from a power-law distribution.')

parser.add_argument('-N', default="10", type=int,  help='number of merger events to sample (default: 10)')
parser.add_argument('-o', '--outputdir',  default="../data/uniform_networks", type=str,  help='directory of output networks (default: ../data/uniform_networks)')
parser.add_argument('--mmin', default="1.0",  type=float, help='minimum mass in Solar Masses (default: 1.0)')
parser.add_argument('--mmax', default="100.0",  type=float, help='maximum mass in Solar Mass (default: 100.0)')
parser.add_argument('--offset', default="0",  type=int, help='starting index offset')

args = vars(parser.parse_args())
# print(args)

N_events = args["N"]
output_path = args["outputdir"]
m_min = args["mmin"]
m_max = args["mmax"]
offset = args["offset"]
   
seed=42
num_injs = N_events

mass_dict = {'dist': 'uniform', 'mmin': m_min, 'mmax':m_max}
spin_dict = {'geom': 'spherical', 'dim':3, 'chi_lo':0., 'chi_hi':1.}
cosmo_dict = {'sampler': 'uniform_comoving_volume_rejection', 'zmin':0., 'zmax':3.}

data = injections.injections_CBC_params_redshift(cosmo_dict,mass_dict,spin_dict,num_injs=num_injs,seed=seed, redshifted=0)
Mcs, etas, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DLs, iotas, ras, decs, psis, zs = data

#Mcs, etas = injections.mass_sampler(mass_dict,num_injs,seed)
#iotas, ras, decs, psis = injections.angle_sampler(num_injs,seed)
#zs, DLs = injections.redshift_lum_distance_sampler(cosmo_dict,num_injs,seed)


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

        net1 = get_network_response(inj_params=inj_params, approximant='IMRPhenomXAS')
        net2 = get_network_response(inj_params=inj_params, approximant='IMRPhenomD')

        net1.save_network(f'{output_path}/{offset+i}_xas_net')
        net2.save_network(f'{output_path}/{offset+i}_d_net')    
    
    end = time.time()

    comm.Barrier()
    if rank == 0:
      print('Done')
      print("Execution time {} seconds".format(end - start))


