import pandas as pd
import glob
import argparse
import numpy as np
from tqdm import tqdm
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
from scipy.stats import multivariate_normal
import pycbc.conversions as conv
import multiprocessing

parser = argparse.ArgumentParser(description='Combine the output from fisher calculations.')

parser.add_argument('-i', '--inputdir', default="../data/powerlaw_smooth_hybrid_3G",  type=str, help='input directory of npz files (default: ../data/powerlaw_smooth_hybrid_3G')

parser.add_argument('-o', '--outputdir', default="../output/powerlaw_smooth_hybrid_3G",  type=str, help='folder to output the binary properties and biases (default: ../output/powerlaw_smooth_hybrid_3G)')

args = vars(parser.parse_args())

inputdir = args["inputdir"]
outdir = args["outputdir"]

list_of_folders = glob.glob(inputdir+"/*/")


def post_process(folder):
    outfile = outdir + "/" + folder.split('/')[-2]+".csv"
    print(f"Reading from {folder}")

    files = glob.glob1(folder,f"hybr_*_bin_*")
    df = pd.DataFrame()

    print(f"{len(files)} files found. \n")

    for file in tqdm(files):
        try:    
            data = np.load(folder+file, allow_pickle=True)
            cov = data['cov']
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

        z_err = z_at_value(Planck18.luminosity_distance, (DL_inj + DL_err) * u.Mpc, zmax=10e5) - z_inj

        df_inj['z'] = z_inj
        df_inj['z_err'] = z_err
        df_inj['z_bias'] = z_bias

        
        # Compute source-frame masses 
        Mc_inj = data['inj_params'].item()['Mc']
        eta_inj =  data['inj_params'].item()['eta']
        DL_inj = data['inj_params'].item()['DL']
        
        df_inj['m1_src'] = conv.mass1_from_mchirp_eta(Mc_inj/(1+z_inj), eta_inj) # injected values
        df_inj['m2_src'] = conv.mass2_from_mchirp_eta(Mc_inj/(1+z_inj), eta_inj) # injected values
        

        # Sample source-frame masses to get statistical errors
        dist_m = multivariate_normal(mean=[Mc_inj, eta_inj, DL_inj], cov=data['cov'][:3, :3], allow_singular=True)        
        samples = dist_m.rvs(1000) # sample Mc, eta, DL using the covariance matrix
        samples = samples[(samples[:,0] > 0)] # Mc should be positive
        samples = samples[(samples[:,1] < 0.25) * (samples[:,1] > 0.0)] # eta should be between 0 and 0.25
        samples = samples[(samples[:,2] > 0.0)] # DL should be positive
        
        mchirp_samples = samples[:,0]
        eta_samples = samples[:,1]
        z_samples = z_at_value(Planck18.luminosity_distance, samples[:,2] * u.Mpc)

        m1_src_samples = conv.mass1_from_mchirp_eta(mchirp_samples/(1+z_samples), eta_samples)
        df_inj['m1_src_err'] = np.percentile(m1_src_samples, 84) - np.percentile(m1_src_samples,16) # ~1 sigma interval

        m2_src_samples = conv.mass2_from_mchirp_eta(mchirp_samples/(1+z_samples), eta_samples)
        df_inj['m2_src_err'] = np.percentile(m2_src_samples, 84) - np.percentile(m2_src_samples,16) # ~1 sigma interval


        # Compute source mass biases
        mchirp_biased = np.maximum(Mc_inj + df_inj["Mc_bias"], 0.0) # make sure that Mc can't be negative
        eta_biased = np.maximum(eta_inj + + df_inj["eta_bias"], 0.0) # make sure that eta isn't negative
        eta_biased = np.minimum(eta_biased, 0.25) # make sure that eta isn't larger than 0.25
        z_biased = np.maximum(z_inj+z_bias, 1e-8)# make sure that redshift isn't negative

        m1_biased = conv.mass1_from_mchirp_eta(mchirp_biased/(1+z_biased), eta_biased)
        m2_biased = conv.mass2_from_mchirp_eta(mchirp_biased/(1+z_biased), eta_biased)

        df_inj['m1_src_bias'] = m1_biased - df_inj['m1_src']
        df_inj['m2_src_bias'] = m2_biased - df_inj['m2_src']


        # compute detector-frame masses
        df_inj['m1_det'] = conv.mass1_from_mchirp_eta(Mc_inj, eta_inj) # injected values
        df_inj['m2_det'] = conv.mass2_from_mchirp_eta(Mc_inj, eta_inj) # injected values

        # sample detector-frame masses to get statistical errors
        m1_det_samples = conv.mass1_from_mchirp_eta(mchirp_samples, eta_samples)
        df_inj['m1_det_err'] = np.percentile(m1_det_samples, 84) - np.percentile(m1_det_samples,16) # ~1 sigma interval

        m2_det_samples = conv.mass2_from_mchirp_eta(mchirp_samples/(1+z_samples), eta_samples)
        df_inj['m2_det_err'] = np.percentile(m2_det_samples, 84) - np.percentile(m2_det_samples,16) # ~1 sigma interval
        
        # compute detector-frame mass biases
        m1_biased = conv.mass1_from_mchirp_eta(mchirp_biased, eta_biased)
        m2_biased = conv.mass2_from_mchirp_eta(mchirp_biased, eta_biased)

        df_inj['m1_det_bias'] = m1_biased - df_inj['m1_det']
        df_inj['m2_det_bias'] = m2_biased - df_inj['m2_det']



        # Mass Ratio
        df_inj['q'] = df_inj['m2_det']/df_inj['m1_det']
        
        q_samples = m2_det_samples / m1_det_samples
        df_inj['q_err'] = np.percentile(q_samples, 84) - np.percentile(q_samples,16) # ~1 sigma interval
        q_biased = np.minimum((m2_biased/m1_biased), 1.0) # set upper limit on q
        q_biased = np.maximum(q_biased, 0.0) # set lower limit on q
        df_inj['q_bias'] = q_biased - df_inj['q']



        df_inj.set_index("index", inplace=True)
        

        df = pd.concat([df, df_inj], ignore_index=False)

    print(f"Completed. Writing output to {outfile}")
    df.to_csv(outfile)


if __name__ == "__main__":
    
    pool = multiprocessing.Pool()
    results = pool.map(post_process, list_of_folders)

    print("Computation Complete!")

            
