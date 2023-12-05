import numpy as np
from gwbench import injections
from gwbench import network
import astropy.units as u
from gwbench import basic_relations as br
from gwbench import waveform as wfc

network_dict = {
        'aLIGO':    ['aLIGO_H','aLIGO_L','V+_V'],
        'A+':       ['A+_H', 'A+_L','A+_V'],
        'Voyager':  ['Voyager-CBO_H', 'Voyager-CBO_L', 'Voyager-CBO_I'],
        '3G':       ['CE-40_C', 'CE-20_S', 'ET_ET1', 'ET_ET2', 'ET_ET3']
        }

def get_network_spec(net_key):
    return network_dict[net_key]

def get_network_snr(inj_params, f_min=5., f_max=1024., network_key = None, network_spec = ['CE-40_C', 'CE-20_S', 'ET_ET1', 'ET_ET2', 'ET_ET3'], approximant='IMRPhenomXAS', deriv_symbs_string = 'Mc eta DL chi1z chi2z iota ra dec psi', cond_num=1e25):

    # if plain text network key is passed, override full network_spec
    if network_key is not None:
        network_spec = network_dict[network_key]

    # initialize the network with the desired detectors
    net = network.Network(network_spec)

    # choose the desired waveform 
    wf_model_name = 'lal_bbh'
    # pass the chosen waveform to the network for initialization
    net.set_wf_vars(wf_model_name=wf_model_name, wf_other_var_dic = {'approximant': approximant})

    # pick the desired frequency range
    #f_min = 5.
    #f_max = 1024.
    d_f = 2**-4
    f = np.arange(f_min, f_max, d_f)

    # choose whether to take Earth's rotation into account
    use_rot = 0

    # pass all these variables to the network
    net.set_net_vars(
        f=f, inj_params=inj_params,
        deriv_symbs_string=deriv_symbs_string,
        use_rot=use_rot
        )

    # setup antenna patterns, location phase factors, and PSDs
    net.setup_ant_pat_lpf_psds()

    # compute the detector responses
    net.calc_det_responses()

    # calculate the network and detector SNRs
    net.calc_snrs()

    return net


def get_network_response(inj_params, f_min=5., f_max=1024., 
        network_key = None, 
        network_spec = ['CE-40_C', 'CE-20_S', 'ET_ET1', 'ET_ET2', 'ET_ET3'], 
        approximant='IMRPhenomXAS', 
        deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota ra dec psi', 
        cond_num=1e25, calc_detector_responses=True):
    
    # if plain text network key is passed, override full network_spec
    if network_key is not None:
        network_spec = network_dict[network_key]


    # initialize the network with the desired detectors
    net = network.Network(network_spec)

    # choose the desired waveform 
    wf_model_name = 'lal_bbh'
    # pass the chosen waveform to the network for initialization
    net.set_wf_vars(wf_model_name=wf_model_name, wf_other_var_dic = {'approximant': approximant})

    # pick the desired frequency range
    #f_min = 5.
    #f_max = 1024.
    d_f = 2**-4
    f = np.arange(f_min, f_max, d_f)

    # choose whether to take Earth's rotation into account
    use_rot = 0

    # pass all these variables to the network
    net.set_net_vars(
        f=f, inj_params=inj_params,
        deriv_symbs_string=deriv_symbs_string,
        use_rot=use_rot
        )

    # compute the WF polarizations
    net.calc_wf_polarizations()

    if not calc_detector_responses:
        return net
    
    # compute the WF polarizations and their derivatives
    net.calc_wf_polarizations_derivs_num()

    # setup antenna patterns, location phase factors, and PSDs
    net.setup_ant_pat_lpf_psds()

    # compute the detector responses
    net.calc_det_responses()
    # compute the detector responses and their derivatives
    net.calc_det_responses_derivs_num()

    # calculate the network and detector SNRs
    net.calc_snrs()

    # calculate the network and detector Fisher matrices, condition numbers,
    # covariance matrices, error estimates, and inversion errors
    net.calc_errors(cond_sup=cond_num)

    # calculate the 90%-credible sky area (in deg)
    # net.calc_sky_area_90()

    return net



def get_hybrid_network_response(inj_params2, inj_params1=None, 
        f_min=5., f_max=1024.,
        network_key = None,
        network_spec = ['CE-40_C', 'CE-20_S', 'ET_ET1', 'ET_ET2', 'ET_ET3'],
        approximant1='IMRPhenomXAS', approximant2='IMRPhenomD',
        deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota ra dec psi',
        cond_num=1e25):

    # if plain text network key is passed, override full network_spec
    if network_key is not None:
        network_spec = network_dict[network_key]


    # initialize the network with the desired detectors
    net = network.Network(network_spec)

    # choose the desired waveform 
    wf_model_name =  'lal_hybrid_bbh'
    # pass the chosen waveform to the network for initialization
    net.set_wf_vars(wf_model_name, wf_other_var_dic = {'approximant1': approximant1,
                                                       'approximant2': approximant2
                                                       })
    # pick the desired frequency range
    d_f = 2**-4
    f = np.arange(f_min, f_max, d_f)

    # choose whether to take Earth's rotation into account
    use_rot = 0

    # pass all these variables to the network
    net.set_net_vars(
        f=f, inj_params=inj_params2,
        deriv_symbs_string=deriv_symbs_string,
        use_rot=use_rot
        )

    # compute the WF polarizations
    net.calc_wf_polarizations()
    # compute the WF polarizations and their derivatives
    net.calc_wf_polarizations_derivs_num()

    # setup antenna patterns, location phase factors, and PSDs
    net.setup_ant_pat_lpf_psds()

    # compute the detector responses
    net.calc_det_responses()
    # compute the detector responses and their derivatives
    net.calc_det_responses_derivs_num()

    # calculate the network and detector SNRs
    net.calc_snrs()

    # calculate the network and detector Fisher matrices, condition numbers,
    # covariance matrices, error estimates, and inversion errors
    net.calc_errors(cond_sup=cond_num)

    # compute the cutler-vallisneri bias with respect to the approximant1 waveform model
    wf_true = wfc.Waveform(wf_model_name = 'lal_bbh', wf_other_var_dic = {'approximant': approximant1})
    net.calc_cutler_vallisneri_bias(wf=wf_true, inj_params=inj_params1)

    return net

