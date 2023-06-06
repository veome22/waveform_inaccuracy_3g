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

from pycbc.types import FrequencySeries
from pycbc.filter import match

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

args = vars(parser.parse_args())

num_injs = args["N"]
input_file = args["input"]
output_path = args["outputdir"]

offset = args["offset"]

approx1 = args["approx1"]
approx2= args["approx2"]
hybr = args["hybr"]

net_key = args["net_key"]



if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

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
        
        sys.stdout.write("\n Binary number %d (%d) being simulated by processor %d of %d\n" % (i+offset, task, rank, size))
        sys.stdout.write(f"Mc: {Mcs[i+offset]:.2f}, eta: {etas[i+offset]:.2f}\n")
    
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

       
        net_hybr = gwnet.get_hybrid_network_response(inj_params=inj_params, network_key=net_key, f_max=f_highs[i+offset],
                            approximant1=approx1, approximant2=approx2, cond_num=1e25)



        # Compute mismatch
        net_true = gwnet.get_network_response(inj_params=inj_params, f_max=f_highs[i+offset], approximant=approx1, network_key=net_key, calc_detector_responses=False)
        
        delta_f = net_hybr.f[1] - net_hybr.f[0]
        psd = FrequencySeries(net_hybr.detectors[0].psd, delta_f=delta_f) # caluclate mismatch using any one detector PSD
        
        # make sure that the detector and waveform frequency ranges overlap
        freq_mask = np.in1d(net_hybr.f, net_hybr.detectors[0].f, assume_unique=True)
        
        hp1_pyc = FrequencySeries(net_true.hfp[freq_mask], delta_f=delta_f)
        hp2_pyc = FrequencySeries(net_hybr.hfp[freq_mask], delta_f=delta_f)
        faith, index = match(hp1_pyc, hp2_pyc, psd=psd, low_frequency_cutoff=net_hybr.f[0], high_frequency_cutoff=net_hybr.f[-1])
        
        # Compute the inner product (unoptimized faithfulness)
        hp1_norm = np.sum((hp1_pyc * np.conjugate(hp1_pyc) / psd).data)
        hp2_norm = np.sum((hp2_pyc * np.conjugate(hp2_pyc) / psd).data)
        inner_prod = np.abs(np.sum((hp1_pyc * np.conjugate(hp2_pyc)/psd).data)) / np.abs(np.sqrt(hp1_norm*hp2_norm))



        # Compute the z stat error and bias
        z_inj = z_at_value(Planck18.luminosity_distance, net_hybr.inj_params["DL"] * u.Mpc, zmax=1e10)
        param_list = net_hybr.deriv_symbs_string.split()
        DL_bias = np.array(net_hybr.cutler_vallisneri_bias)[0][param_list.index('DL')]
    
        # ensure that the biased redshift cannot be below 0
        if (net_hybr.inj_params["DL"] + DL_bias) < 0:
            z_bias = 1e-8 - z_inj
        else:    
            z_bias = z_at_value(Planck18.luminosity_distance, (net_hybr.inj_params["DL"]+DL_bias) * u.Mpc, zmax=1e10)
        
        z_err = z_at_value(Planck18.luminosity_distance, (net_hybr.inj_params["DL"]+net_hybr.errs["DL"]) * u.Mpc, zmax=1e10) - z_inj


        print(f"PyCBC Faithfulness: {faith}, Inner Product {inner_prod}\n")

        # Save binary parameters, statistical errors, waveform bias, mismatch, inner product
        np.savez(outfile, inj_params=net_hybr.inj_params, errs=net_hybr.errs, cv_bias=net_hybr.cutler_vallisneri_bias, snr=net_hybr.snr, faith=faith, inner_prod=inner_prod,\
                z_inj=z_inj, z_err=z_err, z_bias=z_bias, index=i+offset)


    end = time.time()

    comm.Barrier()
    
    if rank == 0:
      print('Done')
      print("Execution time {} seconds".format(end - start))


