import os

# N = 1
# offset = 100000
# timeout = '1:00:00'
# nodes = 1
# pop_file = '../data/smooth_powerlaw_pop_extra.npz'

N = 100000 
offset = 0
timeout = '24:00:00'
nodes = 10
pop_file = '../data/smooth_powerlaw_pop.npz'

# hybrs = [0.0, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
hybrs = [0.0]

job_dir = '/home/vkapil1/scratch16-berti/3G_systematics_veome/sbatch_scripts/hybrid_powerlaw_3G_scripts/'
output_dir = '../data/powerlaw_smooth_hybrid_3G_production'

for hybr in hybrs:
    job_file = job_dir + f'hybr_{hybr}.script'
    script_text = f"""#!/bin/bash
#SBATCH --job-name=hybrid_powerlaw_alpha_3.5_smooth_3G_hybr_{hybr}
#SBATCH --time={timeout}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=48
#SBATCH --partition=defq
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vkapil1@jh.edu
    
module load anaconda
conda activate gwbench_dev
    
echo $SLURM_JOB_NODELIST
mpirun  -np {nodes*48} python ~/scratch16-berti/3G_systematics_veome/src/compute_hybrid_fisher.py -N {N} --offset {offset} -i \"{pop_file}\" -o \"{output_dir}\" --hybr {hybr} --approx1 \"IMRPhenomXAS\" --approx2 \"IMRPhenomD\"  --net_key \"3G\"

    """

    with open(job_file, "w") as jf:
        jf.write(script_text)

    os.system("sbatch %s" %job_file)
