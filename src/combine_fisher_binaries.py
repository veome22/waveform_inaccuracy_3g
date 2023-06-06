import pandas as pd
import glob
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Combine the output from fisher calculations.')

parser.add_argument('-i', '--inputdir', default="../data/powerlaw_smooth_hybrid_3G/hybr_0.0/",  type=str, help='input directory of npz files (default: ../data/powerlaw_smooth_hybrid_3G/hybr_0.0/')

parser.add_argument('-o', '--outfile', default="../output/powerlaw_smooth_hybrid_3G/hybr_0.0.csv",  type=str, help='file to output the binary properties and biases (default: ../output/powerlaw_smooth_hybrid_3G/hybr_0.0.csv)')

args = vars(parser.parse_args())

inputdir = args["inputdir"]
outfile = args["outfile"]

files = glob.glob1(inputdir,f"hybr_*_bin_*")
df = pd.DataFrame()

print(f"{len(files)} files found. \n")

for file in tqdm(files):
    try:    
        data = np.load(inputdir+file, allow_pickle=True)
    except:
        continue

    df_inj = pd.DataFrame([data['inj_params'].item()])

    param_cols = data['errs'].item().keys()
    err_colnames = [str(param)+'_err' for param in list(param_cols)]
    df_inj[err_colnames] = list(data['errs'].item().values())

    bias_colnames = [str(param)+'_bias' for param in list(param_cols)]
    df_inj[bias_colnames] = list(data['cv_bias'][0])

    df_inj['snr'] = data['snr']
    df_inj["index"] = data['index']

    df_inj['faith'] = data['faith']
    df_inj['inner_prod'] = data['inner_prod']
    df_inj['z'] = data['z_inj']
    df_inj['z_err'] = data['z_err']
    df_inj['z_bias'] = data['z_bias']

    df_inj.set_index("index", inplace=True)
    

    df = pd.concat([df, df_inj], ignore_index=False)

print(f"Completed. Writing output to {outfile}")
df.to_csv(outfile)

        
