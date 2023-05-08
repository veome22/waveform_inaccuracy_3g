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

import gwbench_network_funcs as gwnet

parser = argparse.ArgumentParser(description='Generate a list of binaries sampled from a uniform grid in Mc and eta.')

parser.add_argument('--N_q', default="10", type=int,  help='number of bins in the mass-ratio (q) space (default: 10)')
parser.add_argument('--N_mtot', default="10", type=int,  help='number of bins in the total mass space (default: 10)')
parser.add_argument('--N_chi1z', default="10", type=int,  help='number of bins in the chi1_z space (default: 10)')
parser.add_argument('--N_chi2z', default="10", type=int,  help='number of bins in the chi2_z space (default: 10)')

parser.add_argument('-o', '--outputdir',  default="../data/", type=str,  help='directory of output networks (default: ../data/)')

parser.add_argument('--mtot_min', default="10.0",  type=float, help='minimum total binary mass in Solar Mass (default: 10.0)')
parser.add_argument('--mtot_max', default="100.0",  type=float, help='maximum total binary mass in Solar Mass (default: 100.0)')

parser.add_argument('--q_min', default="0.1",  type=float, help='minimum mass ratio (q) (default: 0.1)')
parser.add_argument('--q_max', default="0.99",  type=float, help='maximum mass ratio (q) (default: 0.99)')

parser.add_argument('--chi1z_min', default="-1.0",  type=float, help='minimum chi1_z (default: -1.0)')
parser.add_argument('--chi1z_max', default="1.0",  type=float, help='maximum chi1_z (default: 1.0)')

parser.add_argument('--chi2z_min', default="-1.0",  type=float, help='minimum chi2_z (default: -1.0)')
parser.add_argument('--chi2z_max', default="1.0",  type=float, help='maximum chi2_z (default: 1.0)')


#parser.add_argument('--SNR', default=None, type=float, nargs='+',  help='SNRs of events (default: [10.0, 20.0, 50.0, 100.0, 200.0])')


parser.add_argument('--DL', default="400.0",  type=float, help='luminosity distance of event in Mpc (default: 400.0, modeled after GW150914)')
parser.add_argument('--SNR', default="None",  type=float, help='SNR of event in Mpc. Overrides the DL argument if specified. (default: None')

parser.add_argument('--offset', default="0",  type=int, help='starting index offset')

parser.add_argument('--approximant1', default="IMRPhenomXAS",  type=str, help='Approximant to use for reference waveform')
parser.add_argument('--approximant2', default="IMRPhenomD",  type=str, help='Approximant to use for other waveform')

parser.add_argument('--suffix1', default="xas",  type=str, help='filename suffix to use for reference waveform')
parser.add_argument('--suffix2', default="d",  type=str, help='filename suffix to use for other waveform')


args = vars(parser.parse_args())
# print(args)

n_q = args["N_q"]
n_mtot = args["N_mtot"]
n_chi1z = args["N_chi1z"]
n_chi2z = args["N_chi2z"]

output_dir = args["outputdir"]

mtot_min = args["mtot_min"]
mtot_max = args["mtot_max"]


q_min = args["q_min"]
q_max = args["q_max"]

chi1z_min = args["chi1z_min"]
chi1z_max = args["chi1z_max"]

chi2z_min = args["chi2z_min"]
chi2z_max = args["chi2z_max"]

DL = args["DL"]
target_snr = args["SNR"]

offset = args["offset"]

approximant1 = args["approximant1"]
approximant2 = args["approximant2"]

suffix1 = args["suffix1"]
suffix2 = args["suffix2"]

seed=42

output_path = output_dir + f'snr_{target_snr:.1f}_mtot_{mtot_min:.0f}_{mtot_max:.0f}_q_{q_min:.2f}_{q_max:.2f}_chi1z_{chi1z_min:.1f}_{chi2z_max:.1f}'

sys.stdout.write("\n Simulating grid into" + output_path + "\n\n")

n_total = n_q * n_mtot * n_chi1z * n_chi2z


q_range = np.geomspace(q_min, q_max, num=n_q)
mtot_range = np.linspace(mtot_min, mtot_max, num=n_mtot)

chi1z_range = np.linspace(chi1z_min, chi1z_max, num=n_chi1z)
chi2z_range = np.linspace(chi2z_min, chi2z_max, num=n_chi2z)

# get the grid versions of all varied parameters
qs, mtots, chi1zs, chi2zs = np.meshgrid(q_range, mtot_range, chi1z_range, chi2z_range)

# compute individual masses over the grid and flatten
mass1 = (mtots / (qs+1)).reshape(n_total)
mass2 = (mtots* (qs/(qs+1))).reshape(n_total)

# flatten the spins
chi1_z = chi1zs.reshape(n_total)
chi2_z = chi2zs.reshape(n_total)


Mcs = (mass1*mass2)**(3/5) / (mass1+mass2)**(1/5) 
etas = (mass1*mass2) / (mass1+mass2)**2

mtotals = (mass1+mass2)

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

    for i, task in enumerate(range(n_total)):
        
        if i%size!=rank: continue
        
        inj_params = {
            'Mc':    Mcs[i],
            'eta':   etas[i],
            'chi1x': 0.,
            'chi2x': 0.,
            'chi1y': 0.,
            'chi2y': 0.,
            'chi1z': chi1_z[i],
            'chi2z': chi2_z[i],
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

        sys.stdout.write(f"Mc: {Mcs[i]:.2f}, eta: {etas[i]:.2f}, chi1z: {chi1_z[i]:.2f}, chi2_z: {chi2_z[i]:.2f}\n")
        
        # Make sure the distance is set to achieve target SNR
        if target_snr is not None:
            # get the fiducial snr at DL
            net1_snr = gwnet.get_network_snr(inj_params=inj_params, f_max=f_highs[i], approximant=approximant1)
            
            # calculate DL required to hit target_snr
            new_DL = DL * (net1_snr.snr / target_snr)
            
            # adjust injected DL as needed
            inj_params['DL'] = new_DL


        net2 = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant=approximant2)

        if net2.cov is None:
            sys.stdout.write(f"Matrix not invertible for Mc={Mcs[i]:.2f}, eta={etas[i]:.2f}, writing empty file\n.")
            with open(f'{output_path}/{offset+i}_{suffix1}_net', "wb") as fi:
                dill.dump(None, fi)
            with open(f'{output_path}/{offset+i}_{suffix2}_net', "wb") as fi:
                dill.dump(None, fi)
        else:
            net1 = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i], approximant=approximant1)
            net1.save_network(f'{output_path}/{offset+i}_{suffix1}_net')
            net2.save_network(f'{output_path}/{offset+i}_{suffix2}_net')    
    
    end = time.time()

    comm.Barrier()
    if rank == 0:
      print('Done')
      print("Execution time {} seconds".format(end - start))


