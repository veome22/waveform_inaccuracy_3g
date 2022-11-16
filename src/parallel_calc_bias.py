from mpi4py import MPI
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate biases for a given parameter and population.')

parser.add_argument('-N', default="48", type=int,  help='number of calculations to perform per call of function (default: 48)')
parser.add_argument('--nparallel', default="2", type=int,  help='number of parallel calls of the function (default: 28)')
parser.add_argument('-o', '--outputdir',  default="../output", type=str,  help='directory of output file (default: ../output)')
parser.add_argument('--offset', default="0",  type=int, help='starting index offset')
parser.add_argument('-p', '--parameter', default="0", type=int,  help='parameter to compute errors for (default: 0 : Mc)')
parser.add_argument('-i', '--inputdir', default="../data/uniform_networks_f_max",  type=str, help='input directory of network files (default: ../data/uniform_networks_f_max')


args = vars(parser.parse_args())

N_events = args["N"]
n_parallel = args["nparallel"]
output_path = args["outputdir"]
offset = args["offset"]
inputdir = args["inputdir"]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


for i, task in enumerate(range(n_parallel)):
    
    if i%size!=rank: continue

    start_offset = offset + (i*N_events)
    end = offset + (i*N_events) + N_events
    outfile_name = f"uniform_f_max_{start_offset}to{end}.csv"
    
    print(f'python calc_parameter_bias.py -p 0 -i {inputdir} -o {output_path}/{outfile_name} -N {N_events} --offset {start_offset}')
    os.system(f'python calc_parameter_bias.py -p 0 -i {inputdir} -o {output_path}/{outfile_name} -N {N_events} --offset {start_offset}')



comm.Barrier()
if rank == 0:
    print('Done')

