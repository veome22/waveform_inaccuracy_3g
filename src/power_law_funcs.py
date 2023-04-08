import numpy as np
from numba import njit



def sample_m1_events(df_mc, df_eta, biased=False, bias_index=19, injected=False):
    z = df_mc["z"]
    mchirp = df_mc["Mc"]/ (1+z)
    eta = df_mc["eta"]

    sigMc_raw = df_mc["Mc_stat_err"]
    sigEta = df_eta["eta_stat_err"]
    sigZ = df_mc["z_stat_err"]

    # sigMc = np.sqrt((((mchirp**2)*((sigZ**2)*((1.+z)**-4.)))+((sigMc_raw**2)*((1.+z)**-2.))))


    m1_mu_detections = df_mc["m1"] / (1+z)

    aux0=0.25*((((1.+((1.+(-4.*eta))**0.5))**2))*((eta**2)*((mchirp**2)*(sigZ**2))))
    aux1=(((0.3*((1.+(-4.*eta))**0.5))+((0.3*((1.+(-4.*eta))**1.))+(1.*eta)))**2)
    aux2=(((1.+((1.+(-4.*eta))**0.5))**2))*((eta**2)*((sigMc_raw**2)*(((1.+z)**2))))
    aux3=(((1.+(-4.*eta))**-1.)*(aux1*((mchirp**2)*((sigEta**2)*(((1.+z)**2))))))+(0.25*aux2)
    m1_sigma_events=np.sqrt(((eta**-3.2)*(((1.+z)**-4.)*(aux0+aux3))))
    m1_sigma_events = m1_sigma_events.values

    if biased:
        bias_mc = df_mc[f"bias_{bias_index}"]
        bias_eta = df_eta[f"bias_{bias_index}"]

        mchirp_biased = (df_mc["Mc"] + bias_mc) / (1+z)
        eta_biased = np.minimum(df_mc["eta"]+bias_eta, 0.25) # make sure that eta doesn't exceed what is physically possible

        m1_mu_detections = conv.mass1_from_mchirp_eta(mchirp_biased, eta_biased)

    # sample mu from the detection gaussians to avoid Biases
    # m1_mu_sampled = stats.truncnorm.rvs(0, 1000, m1_mu_detections, m1_sigma_events)
    m1_mu_sampled = np.random.normal(m1_mu_detections, m1_sigma_events)

    if injected:
        m1_mu_sampled = m1_mu_detections.values

    return m1_mu_sampled, m1_sigma_events


def sample_m2_events(df_mc, df_eta, biased=False, bias_index=19, injected=False):
    z = df_mc["z"]
    mchirp = df_mc["Mc"]/ (1+z)
    eta = df_mc["eta"]

    sigMc_raw = df_mc["Mc_stat_err"]
    sigEta = df_eta["eta_stat_err"]
    sigZ = df_mc["z_stat_err"]

    # sigMc = np.sqrt((((mchirp**2)*((sigZ**2)*((1.+z)**-4.)))+((sigMc_raw**2)*((1.+z)**-2.))))


    m2_mu_detections = df_mc["m2"] / (1+z)

    aux0=0.25*((((-1.+((1.+(-4.*eta))**0.5))**2))*((eta**2)*((mchirp**2)*(sigZ**2))))
    aux1=(((0.3*((1.+(-4.*eta))**0.5))+((-0.3*((1.+(-4.*eta))**1.))+(-1.*eta)))**2)
    aux2=(((0.5+(-0.5*((1.+(-4.*eta))**0.5)))**2))*((eta**2)*((sigMc_raw**2)*(((1.+z)**2))))
    aux3=(((1.+(-4.*eta))**-1.)*(aux1*((mchirp**2)*((sigEta**2)*(((1.+z)**2))))))+aux2
    m2_sigma_events=np.sqrt(((eta**-3.2)*(((1.+z)**-4.)*(aux0+aux3))))
    m2_sigma_events = m2_sigma_events.values



    if biased:
        bias_mc = df_mc[f"bias_{bias_index}"]
        bias_eta = df_eta[f"bias_{bias_index}"]

        mchirp_biased = (df_mc["Mc"] + bias_mc) / (1+z)
        eta_biased = np.minimum(df_mc["eta"]+bias_eta, 0.25) # make sure that eta doesn't exceed what is physically possible

        m2_mu_detections = conv.mass2_from_mchirp_eta(mchirp_biased, eta_biased)

    # sample mu from the detection gaussians to avoid Biases
    # m2_mu_sampled = stats.truncnorm.rvs(0, 1000, m2_mu_detections, m2_sigma_events)
    m2_mu_sampled = np.random.normal(m2_mu_detections, m2_sigma_events)

    if injected:
        m2_mu_sampled = m2_mu_detections.values

    return m2_mu_sampled, m2_sigma_events #, m2_mu_detections





def sample_m1_m2_events(df_mc, df_eta, biased=False, bias_index=19, injected=True):
    z = df_mc["z"]
    mchirp = df_mc["Mc"]/ (1+z)
    eta = df_mc["eta"]

    sigMc = df_mc["Mc_stat_err"]
    sigEta = df_eta["eta_stat_err"]
    sigZ = df_mc["z_stat_err"]

    # sigMc = np.sqrt((((mchirp**2)*((sigZ**2)*((1.+z)**-4.)))+((sigMc_raw**2)*((1.+z)**-2.))))


    m1_mu_detections = df_mc["m1"] / (1+z)
    m2_mu_detections = df_mc["m2"] / (1+z)

    # compute variance of m1
    aux0=0.25*((((1.+((1.+(-4.*eta))**0.5))**2))*((eta**2)*((mchirp**2)*(sigZ**2))))
    aux1=(((0.3*((1.+(-4.*eta))**0.5))+((0.3*((1.+(-4.*eta))**1.))+(1.*eta)))**2)
    aux2=(((1.+((1.+(-4.*eta))**0.5))**2))*((eta**2)*((sigMc**2)*(((1.+z)**2))))
    aux3=(((1.+(-4.*eta))**-1.)*(aux1*((mchirp**2)*((sigEta**2)*(((1.+z)**2))))))+(0.25*aux2)
    m1_variance=(eta**-3.2)*(((1.+z)**-4.)*(aux0+aux3))
    m1_variance = m1_variance.values

    # compute variance of m2
    aux0=0.25*((((-1.+((1.+(-4.*eta))**0.5))**2))*((eta**2)*((mchirp**2)*(sigZ**2))))
    aux1=(((0.3*((1.+(-4.*eta))**0.5))+((-0.3*((1.+(-4.*eta))**1.))+(-1.*eta)))**2)
    aux2=(((0.5+(-0.5*((1.+(-4.*eta))**0.5)))**2))*((eta**2)*((sigMc**2)*(((1.+z)**2))))
    aux3=(((1.+(-4.*eta))**-1.)*(aux1*((mchirp**2)*((sigEta**2)*(((1.+z)**2))))))+aux2
    m2_variance=(eta**-3.2)*(((1.+z)**-4.)*(aux0+aux3))
    m2_variance = m2_variance.values

    # compute covariance of m1 and m2
    aux0=(0.25+(-0.25*((1.+(-4.*eta))**1.)))*((eta**2)*((sigMc**2)*(((1.+(1.*z))**2))))
    aux1=(-0.09*((1.+(-4.*eta))**2.))+((((1.+(-4.*eta))**1.)*(0.09+(-0.6*eta)))+(-1.*(eta**2)))
    aux2=((0.25+(-0.25*((1.+(-4.*eta))**1.)))*((eta**2)*(sigZ**2)))+(((1.+(-4.*eta))**-1.)*(aux1*((sigEta**2)*(((1.+(1.*z))**2)))))
    m1_m2_covariance=(eta**-3.2)*(((1.+z)**-4.)*(aux0+((mchirp**2)*aux2)))
    m1_m2_covariance = m1_m2_covariance.values

    if biased:
        bias_mc = df_mc[f"bias_{bias_index}"]
        bias_eta = df_eta[f"bias_{bias_index}"]

        mchirp_biased = (df_mc["Mc"] + bias_mc) / (1+z)
        eta_biased = np.minimum(df_mc["eta"]+bias_eta, 0.25) # make sure that eta doesn't exceed what is physically possible
        # eta_biased = df_mc["eta"]+bias_eta

        m1_mu_detections = conv.mass1_from_mchirp_eta(mchirp_biased, eta_biased)
        m2_mu_detections = conv.mass2_from_mchirp_eta(mchirp_biased, eta_biased)

    # sample mu from the detection gaussians to avoid Biases
    m1_mu_sampled = np.random.normal(m1_mu_detections, np.sqrt(m1_variance))
    m2_mu_sampled = np.random.normal(m2_mu_detections, np.sqrt(m2_variance))

    if injected:
        m1_mu_sampled = m1_mu_detections.values
        m2_mu_sampled = m2_mu_detections.values

    return m1_mu_sampled, m2_mu_sampled,  m1_variance, m2_variance, m1_m2_covariance




def power(m1, alpha, m_min, m_max):
    '''
    BBH merger primary mass PDF.
    '''
    if alpha != -1:
        N1 = 1 / ((m_max**(alpha+1) - m_min**(alpha+1))/(alpha+1))
    else:
        N1 = 1/(np.log(m_max/m_min))

    return np.piecewise(m1, [(m1 < m_min), (m1 >= m_min)*(m1 < m_max), (m1 >= m_max)],
                        [0, lambda m1: N1*m1**alpha, 0])


def normal_dist(m1, mu, sigma, amp=1.0):
    A = np.sqrt(2*np.pi) * sigma
    return amp * np.exp(-(m1 - mu) ** 2 / (2 * sigma**2)) / A
    # return stats.norm.pdf(m1, loc=mu, scale=sigma)

def trunc_normal_dist(m1, mu, sigma, m_min=None, m_max=None):
    if m_min is None:
        m_min = np.min(m1)
    if m_max is None:
        m_max = np.max(m1)

    a, b = (m_min - mu) / sigma, (m_max - mu) / sigma
    return stats.truncnorm.pdf(m1, a, b, loc = mu, scale = sigma)

def sigmoid(x, a):
    return 1/(1 + np.exp(a-x))

def bivariate_normal_dist(m1, m2, mu1, mu2, cov):
    sig1 = np.sqrt(cov[0,0])
    sig2 = np.sqrt(cov[1,1])
    sig12 = cov[0,1]

    rho = sig12 / (sig1 * sig2)

    Z = ((m1-mu1)**2 / (sig1)**2) + ((m2-mu2)**2 / (sig2)**2) - ((2*rho*(m1-mu1)*(m2-mu2)) / (sig1*sig2))

    A = 2*np.pi * sig1 * sig2 * np.sqrt(1-(rho**2))

    return np.exp(-(Z / (2 * (1 - rho**2)))) / A


'''
@njit
def bivariate_normal_dist_njit(m1, m2, mu1, mu2, cov00, cov01, cov11):
    sig1 = np.sqrt(cov00)
    sig2 = np.sqrt(cov11)
    sig12 = cov01
    rho = sig12 / (sig1 * sig2)
    Z = ((m1-mu1)**2 / (sig1)**2) + ((m2-mu2)**2 / (sig2)**2) - ((2*rho*(m1-mu1)*(m2-mu2)) / (sig1*sig2))
    A = 2*np.pi * sig1 * sig2 * np.sqrt(1-(rho**2))
    return np.exp(-(Z / (2 * (1 - rho**2)))) / A

@njit
def integrate_trap_njit(y,x):
    s = 0
    for i in range(1, x.shape[0]):
        s += (x[i]-x[i-1])*(y[i]+y[i-1])
    return s/2


@njit
def butterworth_njit(m1, m0, eta):
    y=(1+ (m0/m1)**eta)**(-1)
    norm = integrate_trap_njit(y, m1)
    #return (1+ (m0/m1)**eta)**(-1) / norm

'''
