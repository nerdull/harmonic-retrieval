# Frequency-Selective Harmonic Retrieval
The harmonic retrieval problem is to identify sinusoids in a segment of sampled data and to estimate the frequency of each sinusoid.
This repo hosts a solution to the problem, which is derived from the state space representation in frequency domain.
A major advantage of this method is manifested by its ability to focus only on a narrow frequency band and completely ignore the rest, which can greatly save the computational cost.
Moreover, the accuracy of the retrieval result is remarkably good.
Since it exploits the sinusoidal model of the data, as well as the modulus and the argument of the Fourier transform, the method is able to achieve a superresolution, surpassing the Rayleigh limit of the conventional periodogram estimate.

This code is implemented in `Python`, with a particular aim of easy-to-use in mind.

## Prerequisites
 - `Python 3`
 - `SciPy` & `NumPy`

## Inventory
**class** Harm_Retri()

### attributes
 - y_k: Fourier transform of the input data
 - freq: frequency grid in native units, i.e., from `-1/2` to `1/2`
 - idx_l: lower index (inclusive) of the focused frequency band after round-off on the frequency grid
 - idx_u: upper index (inclusive) of the focused frequency band after round-off on the frequency grid

### methods
load(data)
 - argument
   * data: complex input data in 2D array, where the last axis is for Fourier transform and the first axis is for average

aim(band)
 - argument
   * band: 2-tuple comprising lower and upper limits of the frequency band in native units

shoot(J_c, rho)
 - arguments
   * J_c: number of harmonics in the band
   * rho: power spectral density of the noise in the band
 - returns
   * fhat: retrieved harmonics in ascending order
   * val: eigenvalues in descending order from one intermediate step as a hint on the actual J_c 

## Example
This code is also shipped with a simple example.
Simply run the code in the standalone mode.

``` python
python3 harmonic_retrieval.py
```

For details, see the bottom lines of the code.

## Reference
Xiangcheng Chen, [_Phys. Rev. E_ **101**, 053310](https://doi.org/10.1103/PhysRevE.101.053310) (2020).

An [Open Access](https://pure.rug.nl/ws/portalfiles/portal/171907531/PhysRevE.101.053310.pdf) is kindly provided by the University of Groningen.

## License
This repository is licensed under the **GNU GPLv3**.
