# Quality Flag

Quality flag of spectroscopic solar observations used to identify points taken during poor weather conditions. Derived by fitting a line to the daily airmass-magnitude relation with a Markov Chain Monte Carlo (MCMC) approach.

Based on a mixture model by Foreman-Mackey (2014), and adapted from the methodology of Collier Cameron et al. (2019) who already applied the model on HARPS-N data.

## Requirements

emcee version 2.2.1
```
pip install emcee==2.2.1
```

## How to use the code

1. Edit the preamble

```
### PREAMBLE

# Settings
targ      = 'Sun'                     # target
inst      = 'HARPS'                   # instrument
TZ        = -4                        # time zone w.r.t. UT
path_in   = ''                        #  input path
path_out  = ''                        # output path
path_fig  = ''                        # figure path
file_in   = f'{targ}_{inst}.rdb'      #  input file name
file_out1 = f'{targ}_{inst}_qualflag' # output file name: quality flags
file_out2 = f'{targ}_{inst}_mcmcpara' # output file name: MCMC parameters

...
```

2. Call the script

- If the script is called **without** any system arguments, all days found in the input file will be analyzed:

```
python qualflag.py
```

- If the script is called **with** a Julian Date Number (JDN), defined as INT[JD - 2,400,000 + 0.5], only points on that day will be analyzed:
```
python qualflag.py 58000
```
Here, JDN = 58000 is used as an example.

## References

Collier Cameron, A., Mortier, A., Phillips, D., et al. (2019), MNRAS, 487, 1082, DOI:[10.1093/mnras/stz1215](https://doi.org/10.1093/mnras/stz1215)

Foreman-Mackey, D. 2014, Blog Post: Mixture Models, Zenodo, DOI:[10.5281/zenodo.15856](https://doi.org/10.5281/zenodo.15856)
