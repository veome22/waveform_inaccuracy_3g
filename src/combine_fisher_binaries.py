import pandas as pd
import glob
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Combine the output from fisher calculations.')

parser.add_argument('-i', '--inputdir', default="../data/powerlaw_smooth_hybrid_3G/hybr_0.0/",  type=str, help='input directory of npz files (default: ../data/powerlaw_smooth_hybrid_3G/hybr_0.0/')

parser.add_argument('-o', '--outputdir', default="../output/powerlaw_smooth_hybrid_3G",  type=str, help='folder to output the binary properties and biases (default: ../output/powerlaw_smooth_hybrid_3G)')

args = vars(parser.parse_args())

inputdir = args["inputdir"]
outdir = args["outputdir"]

list_of_folders = glob.glob(inputdir+"/*/")

for folder in list_of_folders:
    outfile = outdir + "/" + folder.split('/')[-2]
    print(f"Reading from {folder}")

    files = glob.glob1(folder,f"hybr_*_bin_*")
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
        
        
        # Get DL info for redshift calculations
        DL_inj = data['inj_params'].item()['DL']
        DL_err = data['errs'].item()['DL']
        DL_index = list(data['errs'].item().keys()).index("DL")
        DL_bias = list(data['cv_bias'])[0][DL_index]

        # Compute the z stat error and bias
        z_inj = z_at_value(Planck18.luminosity_distance, DL_inj * u.Mpc)

        # ensure that the biased redshift cannot be below 0
        if (DL_inj + DL_bias) < 0:
            z_bias = 1e-8 - z_inj
        else:
            z_bias = z_at_value(Planck18.luminosity_distance, (DL_inj + DL_bias) * u.Mpc) - z_inj

        z_err = z_at_value(Planck18.luminosity_distance, (DL_inj + DL_err) * u.Mpc) - z_inj



        df_inj['z'] = data['z_inj']
        df_inj['z_err'] = data['z_err']
        df_inj['z_bias'] = data['z_bias']

        df_inj.set_index("index", inplace=True)
        

        df = pd.concat([df, df_inj], ignore_index=False)

    print(f"Completed. Writing output to {outfile}")
    df.to_csv(outfile)

        
