from mpi4py import MPI
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate biases for a given parameter and population.')

parser.add_argument('-N', default="48", type=int,  help='number of calculations to perform per call of function (default: 48)')
parser.add_argument('--nparallel', default="2", type=int,  help='number of parallel calls of the function (default: 28)')
parser.add_argument('-o', '--outputdir',  default="../output/powerlaw_3.5_lams_fine", type=str,  help='directory of output file (default: ../output/powerlaw_3.5_lams_fine)')


parser.add_argument('--offset', default="0",  type=int, help='starting index offset')

parser.add_argument('-i', '--inputdir', default="../data/powerlaw_3.5_fine",  type=str, help='input directory of network files (default: ../data/powerlaw_3.5_fine')

parser.add_argument('--minlambda', default="0.0", type=float,  help='minimum lambda to compute bias (default: 0.0)')
parser.add_argument('--maxlambda', default="1.0", type=float,  help='minimum lambda to compute bias (default: 1.0)')
parser.add_argument('--nlambda', default="20", type=int,  help='number of lambdas to compute bias (default: 20)')




args = vars(parser.parse_args())

N_events = args["N"]
n_parallel = args["nparallel"]
output_path = args["outputdir"]
offset = args["offset"]
inputdir = args["inputdir"]
params = [0,1] # [Mc, eta]
minlambda = args["minlambda"]
maxlambda = args["maxlambda"]
nlambda = args["nlambda"]


deriv_symbs_string = 'Mc eta DL tc phic iota ra dec psi'
param_list = deriv_symbs_string.split()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


for i, task in enumerate(range(n_parallel)):
    
    if i%size!=rank: continue

    param = param_list[i]
    outfile_name = f"powerlaw_alpha_3.5_lam_0_1_{param}_{offset}_{offset+N_events}.csv"
    command = f'python calc_bias_lambda_grid.py -p {i} -N {N_events}  -i {inputdir} -o {output_path}/{outfile_name} --minlambda {minlambda} --maxlambda {maxlambda} --nlambda {nlambda} --offset {offset}'
    print(command)
    os.system(command)


comm.Barrier()
if rank == 0:
    print('Done')

