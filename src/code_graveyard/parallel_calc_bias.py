from mpi4py import MPI
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate biases for a given parameter and population.')

#parser.add_argument('-N', default="48", type=int,  help='number of calculations to perform per call of function (default: 48)')
parser.add_argument('--nparallel', default="2", type=int,  help='number of parallel calls of the function (default: 28)')
parser.add_argument('-o', '--outputdir',  default="../output/mtot_q_grids", type=str,  help='directory of output file (default: ../output/mtot_q_grids)')

parser.add_argument('-m', '--mtot',  default="5", type=int,  help='mtot (default: 5)')

parser.add_argument('--SNR',  default="100", type=float,  help='SNR (default: 100)')
#parser.add_argument('--offset', default="0",  type=int, help='starting index offset')
#parser.add_argument('-p', '--parameters', default="0", type=int,  help='parameter to compute errors for (default: 0 : Mc)')

parser.add_argument('-i', '--inputdir', default="../data/mtot_5_grid_DL_1000",  type=str, help='input directory of network files (default: ../data/mtot_5_grid_DL_1000')

parser.add_argument('--suffix1', default="xas",  type=str, help='suffix name of network files for approximant 1 (default: xas')
parser.add_argument('--suffix2', default="d",  type=str, help='suffix name of network files for approximant 2 (default: d')


args = vars(parser.parse_args())

#N_events = args["N"]
n_parallel = args["nparallel"]
output_path = args["outputdir"]
mtot = args["mtot"]
snr = args["SNR"]
#offset = args["offset"]
inputdir = args["inputdir"]
params = [0,1] # [Mc, eta]
suffix1 = args["suffix1"]
suffix2 = args["suffix2"]


deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'
param_list = deriv_symbs_string.split()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


for i, task in enumerate(range(n_parallel)):
    
    if i%size!=rank: continue

    param = param_list[i]
    outfile_name = f"mtot_{mtot}_{param}_grid_SNR_{snr}.csv"
    
    print(f'python calc_parameter_bias.py -p {i} -i {inputdir} -o {output_path}/{outfile_name} --suffix1 {suffix1} --suffix2 {suffix2}')
    os.system(f'python calc_parameter_bias.py -p {i} -i {inputdir} -o {output_path}/{outfile_name} --suffix1 {suffix1} --suffix2 {suffix2}')



comm.Barrier()
if rank == 0:
    print('Done')

