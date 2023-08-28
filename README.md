# waveform_inaccuracy_3g

This is the code behind 'Waveform errors for binary black hole populations in next-generation gravitational wave detectors'. 

We use a Fisher matrix approach along with thhe Cutler & Vallisneri [(2007)](https://arxiv.org/abs/0707.2982) formalism to understand the distribution of waveform calibration biases in 3G gravitational wave detectors over an astrophysical population of binary black holes (BBHs).
The BBH populaiton is inspired by the ``TRUNCATED`` model from [R. Abbott (LIGO Lab., Caltech) et al.](https://arxiv.org/abs/2111.03634) model, with a smoothing factor at the low-mass end. 

## Procedure
1. Run [src/sample_smooth_powerlaw_pop.py](../main/src/sample_smooth_powerlaw_pop.py) to generate a population of BBHs. 
e.g. 
```
python ~/scratch16-berti/3G_systematics_veome/src/sample_smooth_powerlaw_pop.py -N 100000 -o "../data/smooth_powerlaw_pop.npz" --alpha -3.5 --mmin 5.0 --mmin_lim 3.0 --mmax 60.0 --chi_alpha 2.0 --chi_beta 7.0
```

2. Then run [src/compute_hybrid_fisher.py](../main/src/compute_hybrid_fisher.py) to compute the Fisher matrix errors, waveform derivatives, Cutler-Vallisneri biases, etc. using [``GWBENCH``](https://arxiv.org/abs/2010.15202).
e.g. to compute errors using ``IMRPhenomXAS`` and ``IMRPhenomD`` for a 3G detector network in parallel using ``MPI``, 
```
mpirun -launcher fork -np 48 python ~/scratch16-berti/3G_systematics_veome/src/compute_hybrid_fisher.py -N 100000 -i "../data/smooth_powerlaw_pop.npz" -o "../data/powerlaw_smooth_hybrid_3G/hybr_0.0/" --hybr 0.0 --offset 0 --approx1 "IMRPhenomXAS" --approx2 "IMRPhenomD"  --net_key "3G"
```

The ``--offset`` flag is used to numerically offset the index of the simulated binaries, which is helpful to split the population into batches. To compute errors using hybrid waveforms instead, use the ``hybr`` parameter:

```
mpirun -launcher fork -np 48 python ~/scratch16-berti/3G_systematics_veome/src/compute_hybrid_fisher.py -N 100000 -i "../data/smooth_powerlaw_pop.npz" -o "../data/powerlaw_smooth_hybrid_3G/hybr_0.0/" --hybr 0.9 --offset 0 --approx1 "IMRPhenomXAS" --approx2 "IMRPhenomD"  --net_key "3G"
```

3. Finally, run [src/combine_fisher_binaries.py](../main/src/combine_fisher_binaries.py) to combine the output files produced by ``GWBENCH`` into a csv file. This script also propagates biases and errors to a few astrophysically relevant parameters such as source-frame mass, redshift, mass-ratio, etc.
e.g.

```
python ~/scratch16-berti/3G_systematics_veome/src/combine_fisher_binaries.py -i "../data/powerlaw_smooth_hybrid_3G" -o "../output/powerlaw_smooth_hybrid_3G" -N 100000 --offset 0
```

## Results
The output files used for publication can be found in [output/powerlaw_smooth_hybrid_3G/](../main/output/powerlaw_smooth_hybrid_3G).

Overall distribution of biases across the population between ``IMRPhenomXAS`` and ``IMRPhenomD:
![image](https://github.com/veome22/waveform_inaccuracy_3g/assets/66737615/65c4f5e5-82a4-4319-bb62-2507867d44e4)

Behavior of bias distribution for various hybrid waveform models:
![image](https://github.com/veome22/waveform_inaccuracy_3g/assets/66737615/d424de91-dcd0-4199-a660-89da3e2d70a3)


Fraction of events with $\geq 2\sigma$ bias vs average waveform model mismatch:
![image](https://github.com/veome22/waveform_inaccuracy_3g/assets/66737615/c49e4b55-42e6-462a-970e-4ecac4d8d3b6)

