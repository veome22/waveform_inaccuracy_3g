import os
import numpy as np

N = 100000
# N = 80000
n_per_job = 20000
# offset = 20000
offset = 0
timeout = '20:00:00'
output_dir = '../output/powerlaw_smooth_hybrid_3G_production'

# N = 213
# n_per_job = 213
# offset = 99787
# timeout = '4:00:00'
# output_dir = '../output/powerlaw_smooth_hybrid_3G_production_extra'

nodes = 1


job_dir = '/home/vkapil1/scratch16-berti/3G_systematics_veome/sbatch_scripts/hybrid_powerlaw_3G_scripts/'
input_dir = '../data/powerlaw_smooth_hybrid_3G_production'

binary_range = np.arange(offset, offset+N, n_per_job)

for start in binary_range:
    job_file = job_dir + f'postprocess_{start}_{start+n_per_job}.script'
    script_text = f"""#!/bin/bash

#SBATCH --job-name=post_process_hybrid_powerlaw_{start}_{start+n_per_job}
#SBATCH --time={timeout}
#SBATCH --nodes={nodes} 
#SBATCH --ntasks-per-node=48
#SBATCH --partition=parallel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vkapil1@jh.edu

module load anaconda
source activate gwbench_dev


python ~/scratch16-berti/3G_systematics_veome/src/combine_fisher_binaries.py -i \"{input_dir}\" -o \"{output_dir}\" -N {n_per_job} --offset {start}
    """

    with open(job_file, "w") as jf:
        jf.write(script_text)

    os.system("sbatch %s" %job_file)
