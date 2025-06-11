import numpy as np
from scipy.stats import norm
from numba import njit 

# SSJ imports
from sequence_jacobian import grids, interpolate  

"""
Firms
"""


################### 
# Prelims
###################

# This is just for initialization (this has to be of proper dimensions)
def firm_init(s_grid, k_grid, alpha_k, theta):
    V = s_grid[:, np.newaxis] * (k_grid**alpha_k)**theta
    Va = s_grid[:, np.newaxis] * k_grid**(alpha_k*theta-1) * (alpha_k*theta)
    return V, Va

# grids for capital, incumbent productivity and entrant signal
def firm_make_grid(rho_s, sigma_s, ns, kmin, kmax, nk, ns_e, pareto):

    # capital
    k_grid = grids.agrid(kmax,nk,kmin)
    
    # incumbent productivity (use Tauchen to be consistent with the signal_to_prod transition)
    s_grid, pi_s, Pi_s = grids.markov_tauchen(rho=rho_s, sigma=sigma_s, N=ns, normalize=False)   

    # construct entrant signal grid (make it finer than the productivity grid by having ns_e > ns)
    s_grid_e = np.linspace(s_grid[0], s_grid[-1], ns_e)    # what matters is the transition matrix from s_e to s

    # transition matrix from signal to productivity (using tauchen method, s = \rho_s * s_e + \sigma_s * \eta)
    # following piece of code is borrowed from CP Fortran code
    Pi_s_e_s = np.empty((ns_e, ns))
    step = (s_grid[1] - s_grid[0])/2
    up_grid  = np.linspace(s_grid[0] - step, s_grid[-1] + step, ns+1)
    for q in range(ns_e):

        # Compute the conditional expectation of tomorrow's idiosyncratic shock
        s_bar = rho_s*s_grid_e[q]

        # fill tauchen probabilities from today's signal to all possible states tomorrow
        normalizza =  norm.cdf((up_grid[-1]-s_bar)/sigma_s) -  norm.cdf((up_grid[0]-s_bar)/sigma_s)
        for s in range(ns):
            Pi_s_e_s[q,s] =  norm.cdf((up_grid[s+1]-s_bar)/sigma_s) -  norm.cdf((up_grid[s]-s_bar)/sigma_s)
        Pi_s_e_s[q,:] /=   normalizza

    s_grid = np.exp(s_grid)
    s_grid_e = np.exp(s_grid_e)
    
    # stationary mass over the signal grid for pareto distribution
    step = (np.log(s_grid_e[-1]) - np.log(s_grid_e[0]))/(ns_e-1)
    ultimo = s_grid_e[-1]*np.exp(step);

    pi_s_e = np.empty((ns_e,))
    for q in range(ns_e-1):
        pi_s_e[q] = 1.0 /(s_grid_e[q]**pareto) - 1.0 /(s_grid_e[q+1]**pareto)

    pi_s_e[-1] = 1.0 /(s_grid_e[-1]**pareto) - 1.0 /(ultimo**pareto);
    pi_s_e = pi_s_e*s_grid_e[0]**pareto

    # Take the truncation into account
    pi_s_e = pi_s_e/( 1.0  - (s_grid_e[0]/ultimo)**pareto)

    return s_grid, pi_s, Pi_s, k_grid, s_grid_e, pi_s_e, Pi_s_e_s


#######################
# Incumbents Prelims
#######################

def firm_operation(w, X, s_grid, k_grid, alpha_k, alpha_l, theta): 
    
    """Computes the labor demand, output and maximized profits for the incumbent firms""" 
    
    # compute labor and output
    num = alpha_l*theta * X * s_grid[:, np.newaxis] * k_grid**(alpha_k*theta)/w
    l = num**(1/(1-alpha_l*theta))
    y = X * s_grid[:, np.newaxis] * (k_grid**alpha_k * l**alpha_l)**theta
    
    # profit and its derivative (derivative required in Envelope condition)
    t1 = (1 - alpha_l*theta) / (alpha_l*theta)
    t2 = t1 * w**(-1/t1)
    t3 = t2 * ( alpha_l*theta*X*s_grid[:, np.newaxis])**(1 / (1 - alpha_l*theta) )
    
    profit = t3 * k_grid ** (alpha_k*theta / (1 - alpha_l*theta) )
    profit_k = t3 * k_grid ** (alpha_k*theta / (1 - alpha_l*theta) - 1) * (alpha_k*theta / (1 - alpha_l*theta) )
    
    return profit, profit_k, l, y

#######################
# Incumbents Stage 3 - capital choice for exiting and continuing firms
#######################

# value of the continuing firm - (Stage 3 with optimal capital policy choice conditional on continuing)
def firm_continue(V, Va, w, r, X, profit, profit_k, k_grid, s_grid, alpha_k, alpha_l, theta, delta, c_1, c_f):

    # use all FOCs on endogenous grid only for the continue state
    W  = 1/(1+r) * V                                              # end-of-stage vfun
    Wa = (1/(1+r) * Va - 1)/(2*c_1) + 1 - delta                   # first-order condition
    k_endo = k_grid/Wa                                            # this is endogenous grid (capital today)
    
    # interpolate with upper envelope
    V, k, div = upperenv(W, k_endo, profit, k_grid, delta, c_1, c_f)
    
    # update Va on exogenous grid
    Va = profit_k + 1 - delta + c_1*( (k/k_grid)**2 - (1-delta)**2 )    # envelope condition
    
    return V, Va, k, div

# value of exiting (Exit firm value has a closed form solution); capital = 0 for exiting firm
def firm_exit(profit, profit_k, k_grid, delta, c_1):
    V = profit + (1-delta)*k_grid - c_1*(-(1-delta))**2*k_grid 
    Va = profit_k + (1-delta) - c_1*(-(1-delta))**2
    k = np.zeros_like(V)
    div = V.copy()    # dividends
    return V, Va, k, div

# Stage 3 (capital policy choice)
def firm_incumbent_capital(V, Va, w, r, X, profit, profit_k, k_grid, s_grid, alpha_k, alpha_l, theta, delta, c_1, c_f):

    ############### 1. Continuing firm ###############

    V_cont, Va_cont, k_cont, div_cont = firm_continue(V, Va, w, r, X, profit, profit_k, k_grid, s_grid, alpha_k, alpha_l, theta, delta, c_1, c_f)
    
    ############### 2. Exiting firm ###############
    
    V_exit, Va_exit, k_exit, div_exit = firm_exit(profit, profit_k, k_grid, delta, c_1)
    
    ############### 3. Concatenate for discrete-choice stage ###############
    
    # expand to include exit values
    V = np.zeros((2, V_cont.shape[0], V_cont.shape[1]))
    V[0,:,:] = V_exit.copy()
    V[1,:,:] = V_cont.copy()
    
    Va = np.zeros((2, Va_cont.shape[0], Va_cont.shape[1]))
    Va[0,:,:] = Va_exit.copy()
    Va[1,:,:] = Va_cont.copy()
    
    k = np.zeros((2, k_cont.shape[0], k_cont.shape[1]))
    k[0,:,:] = k_exit.copy()
    k[1,:,:] = k_cont.copy()
    
    div = np.zeros((2, div_cont.shape[0], div_cont.shape[1]))
    div[0,:,:] = div_exit.copy()
    div[1,:,:] = div_cont.copy()
    
    return V, Va, k, div   # includes continuing and exiting firms (2 x Ns x Nk); V is V^(3)

def upperenv(W, k_endo, profit, k_grid, delta, c_1, c_f):
    
    # collapse (n, z, a) into (b, a)
    shape = W.shape
    W = W.reshape((-1, shape[-1]))
    k_endo = k_endo.reshape((-1, shape[-1]))
    profit = profit.reshape((-1, shape[-1]))
    V, k, div = upperenv_vec(W, k_endo, profit, k_grid, delta, c_1, c_f)

    # report on (n, z, a)
    return V.reshape(shape), k.reshape(shape), div.reshape(shape)

@njit
def upperenv_vec(W, k_endo, profit, k_grid, delta, c_1, c_f):
    """Interpolate value function and investment to exogenous grid."""
    n_b, n_k = W.shape
    k = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)
    div = -np.inf * np.ones_like(W)

    # loop over other states, collapsed into single axis
    for ib in range(n_b):
        # loop over segments of endogenous asset grid from EGM (not necessarily increasing)
        for ja in range(n_k - 1):
            k_low, k_high = k_endo[ib, ja], k_endo[ib, ja + 1]
            W_low, W_high = W[ib, ja], W[ib, ja + 1]
            kp_low, kp_high = k_grid[ja], k_grid[ja + 1]
            
           # loop over exogenous asset grid (increasing) 
            for ia in range(n_k):  
                kcur = k_grid[ia]
                profit_cur = profit[ib, ia]
                
                interp = (k_low <= kcur <= k_high) 
                extrap = (ja == n_k - 2) and (kcur > k_endo[ib, n_k - 1])

                # exploit that k_grid is increasing
                if (k_high < kcur < k_endo[ib, n_k - 1]):
                    break

                if interp or extrap:
                    W0 = interpolate.interpolate_point(kcur, k_low, k_high, W_low, W_high)
                    k0 = interpolate.interpolate_point(kcur, k_low, k_high, kp_low, kp_high)
                    if k0 < k_grid[0]:
                        k0 = k_grid[0]
                        k0 = k[ib, ia-1]
                    div0 = profit_cur - c_f + (1-delta)*kcur - k0 - c_1*(k0 - (1-delta)*kcur)**2/kcur
                    V0 = div0 + W0
                    
                    # upper envelope, update if new is better
                    if V0 > V[ib, ia]:
                        k[ib, ia] = k0
                        V[ib, ia] = V0
                        div[ib, ia] = div0

    return V, k, div

#######################
# Incumbents Stage 2 - exit choice
#######################

# this is the logit choice stage (for both entrants and incumbents) - See function below taken directly from SSJ with little modification for computing dV


#######################
# Incumbents Stage 1 - Productivity choice
#######################

# use markov chain (directly in backward iteration function)

###################
# Entrants
###################

## capital choice conditional on signal (Here, the input V is V(k,s') where s' is the productivity entrants will use to **PRODUCE**)
def firm_entrant_capital(V_in, r, k_grid, c_e):
    
    # conditional on entry entrant
    V_e = -k_grid - c_e + 1/(1+r) * V_in    # expected value after next productivity draw conditional on today's signal
    k_e_idx = np.argmax(V_e, axis=1)  
    V_e_max = np.max(V_e, axis=1)
    k_e = k_grid[k_e_idx]
    
    # expand to include 'stay-out' values
    V = np.zeros((2, V_e_max.shape[0]))
    V[1,:] = V_e_max.copy()
    
    k = np.zeros((2, k_e.shape[0]))
    k[1,:] = k_e.copy()
    
    div = np.zeros((2, k_e.shape[0]))
    div[1,:] = - k[1,:] - c_e
    
    return V, k, div


#### Other functions that are in-built in SSJ

# Stage 2 (exit choice)
def logit_choice(V, Va, scale, dV = 1):
    
    """Logit choice probabilities and logsum along 0th axis"""
    const = V.max(axis=0)    # max along discrete choices
    Vnorm = V - const
    Vexp = np.exp(Vnorm / scale)
    
    Vexpsum = Vexp.sum(axis=0)

    P = Vexp / Vexpsum
    EV = const + scale * np.log(Vexpsum)

    # compute derivative
    if dV == 1:
        Va = np.sum(Vexp * Va, axis=0) / Vexpsum

    return EV, Va, P


# discrete choice forward iteration
def batch_multiply_ith_dimension(P, i, X):
    """If P is (D, X.shape) array, multiply P and X along ith dimension of X."""
    # standardize arrays
    P = P.swapaxes(1, 1 + i)
    X = X.swapaxes(0, i)
    Pshape = P.shape
    P = P.reshape((*Pshape[:2], -1))
    X = X.reshape((X.shape[0], -1))

    # P[i, j, ...] @ X[j, ...]
    X = np.einsum('ijb,jb->ib', P, X)

    # original shape and order
    X = X.reshape(Pshape[0], *Pshape[2:])
    return X.swapaxes(0, i)


# lotteries for continuous choice
def get_lottery(a, a_grid):
    # step 1: find the i such that a' lies between gridpoints a_i and a_(i+1)
    a_i = np.searchsorted(a_grid, a) - 1
    
    # step 2: obtain lottery probabilities pi
    a_pi = (a_grid[a_i+1] - a)/(a_grid[a_i+1] - a_grid[a_i])
    
    return a_i, a_pi

@njit
def forward_policy(D, a_i, a_pi):

    Dend = np.zeros_like(D)

    # for each state (discrete and continuous) - could compress non-policy states to single dimension
    for n in range(a_i.shape[0]):
        for s in range(a_i.shape[1]):
            for a in range(a_i.shape[2]):

                # send pi(s,a) of the mass to gridpoint i(s,a)
                Dend[n, s, a_i[n,s,a]] += a_pi[n, s,a]*D[n,s,a]

                # send 1-pi(s,a) of the mass to gridpoint i(s,a)+1
                Dend[n, s, a_i[n,s,a]+1] += (1-a_pi[n, s,a])*D[n,s,a]

    return Dend

 
@njit
def forward_policy_noexit(D, a_i, a_pi):

    Dend = np.zeros_like(D)

    # for each state (discrete and continuous) - could compress non-policy states to single dimension
    for s in range(a_i.shape[0]):
        for a in range(a_i.shape[1]):

            # send pi(s,a) of the mass to gridpoint i(s,a)
            Dend[s, a_i[s,a]] += a_pi[s,a]*D[s,a]

            # send 1-pi(s,a) of the mass to gridpoint i(s,a)+1
            Dend[s, a_i[s,a]+1] += (1-a_pi[s,a])*D[s,a]

    return Dend
