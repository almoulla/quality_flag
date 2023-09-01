# Quality Flag

Quality flag of spectroscopic solar observations used to identify points taken during poor weather conditions. Derived by fitting a line to the daily airmass-magnitude relation with a Markov Chain Monte Carlo (MCMC) approach.

Based on a mixture model by Foreman-Mackey (2014), and adapted from the methodology of Collier Cameron et al. (2019) who already applied the model on HARPS-N data.

## Citation

If you use this code, please cite [Al Moulla et al. (2023)](https://doi.org/10.1051/0004-6361/202244663).

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
targ      = 'Sun'                         # target
inst      = 'HARPS'                       # instrument
TZ        = -4                            # time zone w.r.t. UT
path_in   = ''                            #  input path
path_out  = ''                            # output path
path_fig  = ''                            # figure path
file_in   = f'{targ}_{inst}.csv'          #  input file name
file_out1 = f'{targ}_{inst}_qualflag.csv' # output file name: quality flags
file_out2 = f'{targ}_{inst}_mcmcpara.csv' # output file name: MCMC parameters

...
```

2. Call the script

- If the script is called **without** any system arguments, all days found in the input file will be analyzed:

```
python quality_flag.py
```

- If the script is called **with** a modified Julian Day Number (JDN), defined as INT[JD - 2,400,000 + 0.5], only points on that day will be analyzed:
```
python quality_flag.py 58000
```
Here, JDN = 58000 is used as an example.

## References

Al Moulla, K., Dumusque, X., Figueira, P., et al. 2023, A&A, 669, A39, DOI: [10.1051/0004-6361/202244663](https://doi.org/10.1051/0004-6361/202244663)

Collier Cameron, A., Mortier, A., Phillips, D., et al. 2019, MNRAS, 487, 1082, DOI: [10.1093/mnras/stz1215](https://doi.org/10.1093/mnras/stz1215)

Foreman-Mackey, D. 2014, Blog Post: Mixture Models, Zenodo, DOI: [10.5281/zenodo.15856](https://doi.org/10.5281/zenodo.15856)
