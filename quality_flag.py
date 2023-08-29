"""
AUTHOR : Khaled Al Moulla
DATE   : 2023-08-29

Quality flag of spectroscopic solar observations used to identify points taken during poor weather conditions.
Derived by fitting a line to the daily airmass-magnitude relation with a Markov Chain Monte Carlo (MCMC) approach.

Based on a mixture model by Foreman-Mackey (2014), and adapted from the methodology of Collier Cameron et al. (2019)
who already applied the model on HARPS-N data.

Copyright (c) 2023 Khaled Al Moulla
"""

### ----------------------------------------------------------
### MODULES

import emcee                # version 2.2.1
from   matplotlib.colorbar  import Colorbar
import matplotlib.gridspec  as     gridspec
import matplotlib.pylab     as     pylab
import matplotlib.pyplot    as     plt
import numpy                as     np
import pandas               as     pd
pd.options.mode.chained_assignment = None
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### ----------------------------------------------------------
### PREAMBLE

# Settings
targ      = 'Sun'                         # target
inst      = 'HARPS'                       # instrument
TZ        = -4                            # time zone w.r.t. UT
path_in   = ''                            #  input path
path_out  = ''                            # output path
path_fig  = ''                            # figure path
file_in   = f'{targ}_{inst}.csv'          #  input file name
file_out1 = f'{targ}_{inst}_qualflag.csv' # output file name: quality flags
file_out2 = f'{targ}_{inst}_mcmcpara.csv' # output file name: MCMC parameters

# Input DataFrame
df_all = pd.read_csv(path_in+file_in, sep=',')

# Column names of expected variables
col_jdb      = 'jdb'     # Julian Date Barycentric (JDB)
col_vrad     = 'vrad'    # radial velocity
col_vrad_err = 'svrad'   # radial velocity error
col_airm     = 'airmass' # airmass
col_snr      = 'sn60'    # signal-to-noise ratio (SNR)

# Column names of computed variables
col_jdn      = 'jdn'     # Julian Day Number (JDN)
col_hang     = 'hang'    # hour angle
col_magn     = 'mag60'   # magnitude
col_magn_err = 'smag60'  # magnitude error

# Units
unit_vrad = 'km/s'

# Toggle True/False
clip_outlier = True # outlier clipping
save_outputs = True # save output files
plot_figures = True # plot and save figures

### ----------------------------------------------------------
### INPUT

# One modified Julian Day Number (JDN), defined as INT[JD - 2,400,000 + 0.5], can be given as a system argument.

# If the script is called w/out any system arguments, all days found in the input file will be analyzed
try:
    jdni = int(sys.argv[1:][0])

# If the script is called w/ a JDN, only points on that day will be analyzed
except:
    jdni = None

### ----------------------------------------------------------
### FIGURES

# Colormap
cmap = 'RdYlGn'

# Name
fig_name = lambda jdn, n: f'Fig_{targ}_{inst}' + '_' + str(jdn) + '_' + str(n) + '.pdf'

# Figure size and fontsizes
plot_params = {'figure.figsize'        : (20, 10),
               'figure.titlesize'      :  25,
               'axes.titlesize'        :  25,
               'axes.labelsize'        :  20,
               'xtick.labelsize'       :  20,
               'ytick.labelsize'       :  20,
               'legend.title_fontsize' :  15,
               'legend.fontsize'       :  15
              }
pylab.rcParams.update(plot_params)

### ----------------------------------------------------------
### CONSTANTS

# Discription and bounds of MCMC parameters
pname   = ['$\hat{c}_{0}$', '$c_{1}$', '$\sigma_{\mathrm{jit}}$', '$\hat{\mu}$', '$\sigma$', '$Q$']
bounds  = [(-0.4  , 0.4 ),    # c_0 : intercept of foreground line w.r.t. weighted mean airmass
           ( 0.075, 0.4 ),    # c_1 : slope of foreground line
           ( 0.001, 0.08),    # jit : white noise of foreground
           ( 0.0  , 0.8 ),    # mu  : mean of background w.r.t. weighted mean magnitude
           ( 0.1  , 0.7 ),    # sig : std of background
           ( 0.0  , 1.0 )]    # Q   : fraction of foreground population
ndim    = np.shape(bounds)[0] # nr. of parameters in MCMC
max_c0  = 0.1                 # allowed max. for c0 w.r.t. weighted mean airmass
                              # (larger values indicate a swap between fore- & background)

# MCMC walkers and chains
min_pnt = 10                  # allowed min. nr. of points per day
nwalker = 32                  # nr. of walkers in MCMC
nthin_b = 50                  # thinning step for sampling burn-in chains
nthin_p = 100                 # thinning step for sampling production chain
npt_end = int(2e3)            # nr. of end points from which to sample burn-ins
nstep_b = int(1e4)            # nr. of steps in burn-in
nstep_p = nstep_b             # nr. of steps in production

### ----------------------------------------------------------
### FUNCTIONS

# Interquartile clipping.
# data    : data to be filtered
# returns : boolean for whether points in <data> are w/in the accepted range
def iqr_clip(data, low=False):
    Q1, Q3 = np.nanpercentile(data, [25,75])
    IQR    = Q3 - Q1
    if low:
        return (data > Q1-IQR)
    else:
        return (data > Q1-IQR) & (data < Q3+IQR)

# Sample Gaussian w/ median and MAD from an emcee chain, limited by bounds.
# chain   : emcee chain
# returns : list w/ IC for Walkers
def gauss_bound(chain):
    sample = chain[:,-npt_end::nthin_b,:].reshape(-1,ndim)
    pmed   = np.median(sample, axis=0)
    pmad   = np.median(np.abs(sample-pmed), axis=0)
    p0     = [np.random.normal(pmed,pmad) for _ in range(nwalker)]
    for i in range(nwalker):
        for j in range(ndim):
            if   p0[i][j] < bounds[j][0]:
                    p0[i][j] = bounds[j][0]
            elif p0[i][j] > bounds[j][1]:
                    p0[i][j] = bounds[j][1]
    return p0

# Likelihood of foreground population.
# p       : sample parameters
# x       : data x-coordinates
# y       : data y-coordinates
# yerr    : data y-errors
# returns : Gaussian (ln) probability of belonging to foreground
def lnlike_fg(p, x, y, yerr):
    *c, jit, _, _, _ = p
    ym  = c[0] + c[1]*x
    var = yerr**2 + jit**2
    return -0.5*((y-ym)**2/var + np.log(var))

# Likelihood of background population.
# p       : sample parameters
# x       : data x-coordinates
# y       : data y-coordinates
# yerr    : data y-errors
# returns : Gaussian (ln) probability of belonging to background
def lnlike_bg(p, x, y, yerr):
    *_, mu, sig, _ = p
    var = yerr**2 + sig**2
    return -0.5*((y-mu)**2/var + np.log(var))

# Prior probability.
# p       : sample parameters
# bounds  : uniform bounds on <p>
# returns : probability 1 if w/in bounds else 0 (0 & -INF in natural logarithms)
def lnprior(p, bounds):
    if not all(b[0] <= v <= b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0

# Posterior probability.
# p       : sample parameters
# bounds  : uniform bounds on <p>
# x       : data x-coordinates
# y       : data y-coordinates
# yerr    : data y-errors
# returns : combined probaility (prior+likelihoods in log scale);
#           2D-blob for fore- & and background
def lnprob(p, bounds, x, y, yerr):
    
    # Quality
    *_, Q = p
    
    # Check the prior
    lp = lnprior(p, bounds)
    if not np.isfinite(lp):
        return -np.inf, None
    
    # Compute the vector of foreground likelihoods & include the q prior
    ll_fg = lnlike_fg(p, x, y, yerr)
    if Q == 0.0:
        arg1 = -np.inf
    else:
        arg1 = ll_fg + np.log(Q)
    
    # Compute the vector of background likelihoods & include the q prior
    ll_bg = lnlike_bg(p, x, y, yerr)
    if Q == 1.0:
        arg2 = -np.inf
    else:
        arg2 = ll_bg + np.log(1.0 - Q)
    
    # Combine these using log-add-exp for numerical stability
    ll = np.sum(np.logaddexp(arg1, arg2))
    
    # Use the emcee 'blobs' feature to track fore- & background
    return lp + ll, (arg1, arg2)

# Makes plot of MCMC sampling & its Auto-Correlation Function (ACF).
# chain   : tuple w/ chains
# cname   : tuple w/ chain names
# returns : figure
def plot_mcmc(chain, cname):
    
    # Nr. of chains
    nchain = len(chain)
    
    # Outer figure
    fig = plt.figure()
    outer = gridspec.GridSpec(1, nchain, hspace=0)
    
    # Reference for inner axes
    axes = np.empty(shape=(ndim, 2), dtype=object)
    
    # Loop through chains
    for i in range(nchain):
        
        # Inner figure
        inner = gridspec.GridSpecFromSubplotSpec(ndim, 2, subplot_spec=outer[i], hspace=0, wspace=0)
        
        # Chain title
        ax = fig.add_subplot(outer[i])
        ax.set_title('\n'+cname[i]+'\n\n')
        ax.axis('off')
        
        # Median w.r.t. Walkers
        cmed = np.median(chain[i], axis=0)
        
        # Loop through parameters
        for j in range(ndim):
            
            # COLUMN 1: Parameter vs. Iteration
            
            # Elements
            ax0 = fig.add_subplot(inner[j,0], sharex=axes[0,0])
            axes[j,0] = ax0
            ax0.semilogx(np.arange(1,chain[i].shape[1]+1), chain[i][:,:,j].T, alpha=0.5)
            ax0.semilogx(np.arange(1,chain[i].shape[1]+1), cmed[:,j], 'k')
            
            # Axes
            ax0.set_xlim(1,nstep_p)
            ax0.set_ylim(bounds[j])

            ylim   = ax0.get_ylim()
            ydel   = ylim[1] - ylim[0]
            yticks = ylim[0] + ydel/4*np.array([1,2,3])
            yticks = np.round(yticks,2)
            ax0.set_yticks(yticks)
            
            # COLUMN 2: ACF vs. Lag
            maxlags = 400
            
            # Elements
            ax1 = fig.add_subplot(inner[j,1], sharex=axes[0,1])
            axes[j,1] = ax1
            ax1.acorr(cmed[:,j]-np.median(cmed[:,j]), maxlags=maxlags, color='k')
            
            # Axes
            ax1.set_xlim(1,maxlags)
            ax1.set_ylim(-0.5,1)

            ax1.set_yticks([0.0,0.5])
            
            if i == 0:
                
                ax0.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                                labelleft=False)
                
                if j != ndim-1:
                    ax0.tick_params(labelbottom=False)
                    ax1.tick_params(labelbottom=False)
            
            if i == 1:
                
                ax0.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                                labelbottom=False, labelleft=False)
                ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                                labelbottom=False, labelleft=False)
            
            if i == 2:

                ax0.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                                labelbottom=False, labelleft=False)
                ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                                labelbottom=False, labelleft=False, labelright=True)
            
            if i == 0:
                if j == 0:
                    ax0.set_title('Parameter')
                    ax1.set_title('ACF')
                if j == ndim-1:
                    ax0.set_xlabel('Step')
                    ax1.set_xlabel('Lag')
                ax0.set_ylabel(pname[j])
        
        # Align y-labels
        fig.align_ylabels(axes)
        fig.tight_layout()
    
    return fig

### ----------------------------------------------------------
### DATA

# Variables
jdb      = df_all[col_jdb     ]
vrad     = df_all[col_vrad    ]
vrad_err = df_all[col_vrad_err]
airm     = df_all[col_airm    ]
snr      = df_all[col_snr     ]

# Compute new variables
jdn      = (jdb+0.5).astype('int')
hang     = (jdb - jdn + TZ/24)*360
magn     = -5*np.log10(snr)
magn_err = 2.5/(np.log(10)*snr)

# Add needed variables to DataFrame
df_all[col_jdn     ] = jdn
df_all[col_hang    ] = hang
df_all[col_magn    ] = magn
df_all[col_magn_err] = magn_err

# Add boolean columns for interquartile clipping on RV and SNR
df_all.insert(df_all.columns.get_loc(col_vrad_err)+1, 'in_vrad', np.ones(df_all.shape[0], dtype=bool))
df_all.insert(df_all.columns.get_loc(col_snr)     +1, 'in_snr' , np.ones(df_all.shape[0], dtype=bool))

# Add NaN column for Quality Flag
df_all['qualflag'] = np.empty(df_all.shape[0])*np.nan

# Array w/ each unique day
jdn_arr = np.unique(df_all[col_jdn])

# New DataFrame w/ parameters for each day
df_day = pd.DataFrame()
df_day[col_jdn ] = jdn_arr
df_day['in_day'] = np.zeros(df_day.shape[0], dtype=bool)
df_day['c0'    ] = np.empty(df_day.shape[0])*np.nan
df_day['c1'    ] = np.empty(df_day.shape[0])*np.nan
df_day['jit'   ] = np.empty(df_day.shape[0])*np.nan
df_day['mu'    ] = np.empty(df_day.shape[0])*np.nan
df_day['sig'   ] = np.empty(df_day.shape[0])*np.nan
df_day['Q'     ] = np.empty(df_day.shape[0])*np.nan

### ----------------------------------------------------------
### MCMC

# If only 1 day selected, modify looped array to only include that day
if jdni is not None:
    jdns = np.array([jdni])
else:
    jdns = jdn_arr

# Loop through days
for jdni in jdns:

    # All index
    i_all = df_all[col_jdb][df_all[col_jdn] == jdni].index

    # Day index
    i_day  = np.where(jdn_arr == jdni)[0]
    if len(i_day) > 0: i_day = i_day[0]

    # Outlier clipping
    if clip_outlier:
        df_all.in_vrad[i_all] = iqr_clip(df_all[col_vrad][i_all]          )
        df_all.in_snr [i_all] = iqr_clip(df_all[col_snr ][i_all], low=True)

    # Skip MCMC if day has too few points
    xser = df_all[col_airm][(df_all[col_jdn] == jdni) & df_all.in_vrad & df_all.in_snr]
    if xser.size < min_pnt: continue
    
    # Data points
    x    = np.array(df_all[col_airm    ][xser.index])
    y    = np.array(df_all[col_magn    ][xser.index])
    yerr = np.array(df_all[col_magn_err][xser.index])
    xhat = np.average(x, weights=1/yerr**2)
    yhat = np.average(y, weights=1/yerr**2)
    
    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalker, ndim, lnprob, args=(bounds,x-xhat,y-yhat,yerr))
    
    # 1st burn-in chain
    p0 = [[(b[1]+b[0])/2 for b in bounds] + 1e-5*np.random.randn(ndim) for _ in range(nwalker)]
    sampler.run_mcmc(np.array(p0), nstep_b)
    if plot_figures:
        chain_b1 = sampler.chain
        cname_b1 = '1$^{\mathrm{st}}$ burn-in chain'
    
    # 2nd burn-in chain
    p0 = gauss_bound(sampler.chain)
    sampler.reset()
    sampler.run_mcmc(p0, nstep_b)
    if plot_figures:
        chain_b2 = sampler.chain
        cname_b2 = '2$^{\mathrm{nd}}$ burn-in chain'
    
    # Production chain
    p0 = gauss_bound(sampler.chain)
    sampler.reset()
    sampler.run_mcmc(p0, nstep_p)
    if plot_figures:
        chain_p = sampler.chain
        cname_p = 'Production chain'
    
    # Extract MCMC parameters
    para = np.array(sampler.flatchain[::nthin_p,:])
    c0, c1, jit, mu, sig, Q = np.median(para, axis=0).T
    
    # Save results if fore- & background correctly identified
    if c0 < max_c0:
        
        # Flag day as valid
        df_day.in_day[i_day] = True
        
        # Save parameters
        c0 += yhat - xhat*c1
        mu += yhat
        df_day.c0 [i_day] = c0
        df_day.c1 [i_day] = c1
        df_day.jit[i_day] = jit
        df_day.mu [i_day] = mu
        df_day.sig[i_day] = sig
        df_day.Q  [i_day] = Q
        
        # Quality (i.e. probability of belonging to foreground)
        norm = 0.0
        qual = np.zeros(x.size)
        for ii in range(sampler.chain.shape[1]):
            for jj in range(sampler.chain.shape[0]):
                ll_fg, ll_bg = sampler.blobs[ii][jj]
                qual += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        qual /= norm
        
        # Save quality
        df_all.qualflag[xser.index] = qual

### ----------------------------------------------------------
### PLOT
    
    if plot_figures:
        
        # FIGURE 1: MCMC sampler
        fig = plot_mcmc([chain_b1, chain_b2, chain_p],
                        [cname_b1, cname_b2, cname_p])
        plt.tight_layout()
        plt.savefig(path_fig+fig_name(jdni,1))
        plt.close(fig)
    
    if plot_figures & df_day.in_day[i_day]:
        
        # FIGURE 2: 2x2 mosaic of Quality distribution
        fig     = plt.figure()
        gsg_ext = gridspec.GridSpec(1, 2, hspace=0, wspace=0, width_ratios=[1,0.025])
        gsg_int = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gsg_ext[0], hspace=0, wspace=0)
        
        # Data
        qual     = df_all.qualflag[(df_all[col_jdn] == jdni) & df_all.in_vrad & df_all.in_snr]
        hang     = df_all[col_hang    ][qual.index]
        airm     = df_all[col_airm    ][qual.index]
        magn     = df_all[col_magn    ][qual.index]
        magn_err = df_all[col_magn_err][qual.index]
        vrad     = df_all[col_vrad    ][qual.index]
        vrad_err = df_all[col_vrad_err][qual.index]
        
        # Accepted & Rejected sub-sets
        cut = 0.9
        qual_in, qual_out = qual[qual>cut]     , qual[qual<cut]
        hang_in, hang_out = hang[qual_in.index], hang[qual_out.index]
        airm_in, airm_out = airm[qual_in.index], airm[qual_out.index]
        magn_in, magn_out = magn[qual_in.index], magn[qual_out.index]
        vrad_in, vrad_out = vrad[qual_in.index], vrad[qual_out.index]
        
        # SUBPLOT 1: Magnitude vs. Airmass
        
        ax00 = plt.subplot(gsg_int[0,0])
        
        # Elements
        axc0 = ax00.scatter(airm_in , magn_in , c=qual_in , marker='o', s=25, edgecolors='k', cmap=cmap,
                            vmin=0, vmax=1, zorder=100)
        axc0 = ax00.scatter(airm_out, magn_out, c=qual_out, marker='s', s=25, edgecolors='k', cmap=cmap,
                            vmin=0, vmax=1, zorder=100)
        ax00.errorbar(airm, magn, magn_err, fmt=',k', capsize=2.5)
        xlim     = ax00.get_xlim()
        airm_lin = np.array(xlim)
        ax00.plot(airm_lin, c0+c1*airm_lin, '-k')
        ax00.fill_between(airm_lin, c0+c1*airm_lin-jit, c0+c1*airm_lin+jit, color='gray', alpha=0.5)
        ax00.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        
        # x-axis
        ax00.set_xlabel('Airmass')
        ax00.set_xlim(xlim)
        
        # y-axis
        ax00.set_ylabel('$m$ [mag]')
        ax00.invert_yaxis()
        
        # Legend
        col1, = ax00.plot([], 'ko'  , mfc='w', label='Accepted')
        col2, = ax00.plot([], 'ks'  , mfc='w', label='Rejected')
        col3, = ax00.plot([], 'k'   ,          label='Best fit')
        col4, = ax00.plot([], 'gray', lw=4   , label=r'$\pm\sigma_{\mathrm{jit}}$')
        leg   = ax00.legend(handles=[col1,col2,col3,col4], loc='upper right', edgecolor='k', framealpha=1)
        leg.set_zorder(101)
        
        # SUBPLOT 2: Magnitude vs. Hour Angle
        
        ax01 = plt.subplot(gsg_int[0,1], sharey=ax00)
        
        # Elements
        ax01.scatter(hang_in , magn_in , c=qual_in , marker='o', s=25, edgecolors='k', cmap=cmap,
                     vmin=0, vmax=1, zorder=100)
        ax01.scatter(hang_out, magn_out, c=qual_out, marker='s', s=25, edgecolors='k', cmap=cmap,
                     vmin=0, vmax=1, zorder=100)
        ax01.errorbar(hang, magn, magn_err, fmt=',k', capsize=2.5)
        ax01.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                         labelbottom=False, labelleft=False)
        
        # SUBPLOT 3: -
        
        ax10 = plt.subplot(gsg_int[1,0])
        ax10.set_axis_off()
        
        # SUBPLOT 4: Radial Velocity vs. Hour Angle
        
        ax11 = plt.subplot(gsg_int[1,1], sharex=ax01)
        
        # Elements
        ax11.scatter(hang_in , vrad_in -np.median(vrad), c=qual_in , marker='o', s=25, edgecolors='k', cmap=cmap,
                     vmin=0, vmax=1, zorder=100)
        ax11.scatter(hang_out, vrad_out-np.median(vrad), c=qual_out, marker='s', s=25, edgecolors='k', cmap=cmap,
                     vmin=0, vmax=1, zorder=100)
        ax11.errorbar(hang   , vrad    -np.median(vrad), vrad_err, fmt=',k', capsize=2.5)
        ax11.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        
        # x-axis
        ax11.set_xlabel('Hour angle [°]')
        
        # y-axis
        ax11.set_ylabel(f'RV [{unit_vrad}]')
        
        # COLORBAR
        cbax = plt.subplot(gsg_ext[1])
        cb   = Colorbar(ax=cbax, mappable=axc0)
        cb.set_label('P(good point)')
        cb.ax.axhline(cut, linestyle='--', color='k')
        
        # SAVE FIGURE
        fig.tight_layout()
        plt.savefig(path_fig+fig_name(jdni,2))
        plt.close(fig)

### ----------------------------------------------------------
### SAVE

    # If only 1 day selected, save file w/ only that day
    if save_outputs & (len(jdns) == 1):
        df_all[df_all[col_jdn] == jdni].to_csv(path_out+file_out1.split('.')[0]+f'_{jdni}.'+file_out1.split('.')[1], sep=',', index=False)
        df_day[df_day[col_jdn] == jdni].to_csv(path_out+file_out2.split('.')[0]+f'_{jdni}.'+file_out2.split('.')[1], sep=',', index=False)

# If all days selected, save file w/ all days
if save_outputs & (len(jdns) > 1):
    df_all.to_csv(path_out+file_out1, sep=',', index=False)
    df_day.to_csv(path_out+file_out2, sep=',', index=False)