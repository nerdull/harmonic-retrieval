#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
from numpy.polynomial.polynomial import polyvander
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.linalg import svd, null_space, eigh, eigvals, solve


class Harm_Retri(object):
    '''
    This class implements a harmonic retrieval method based on the state space model in frequency domain.
    It estimates the frequencies of sinusoidal waves, a.k.a. harmonics, from a piece of equi-distanced samples.
    -----
    Attributes:
        y_k: 2D complex array
            DFT of the input data.
        freq: 1D real array
            Frequency grid in the native units, i.e., from -1/2 to 1/2.
        idx_l: int
            Lower index (inclusive) of the focused frequency band after round-off on the frequency grid.
        idx_u: int
            Upper index (inclusive) of the focused frequency band after round-off on the frequency grid.
    '''

    def __init__(self):
        '''
        Class initialization.
        '''
        pass

    def load(self, data):
        '''
        Load the data for the subsequent processing.
        -----
        Arguments:
            data: 1D/2D complex array
                Input data comprising harmonics to be retrieved.
                If data is 1D of size N, it is automatically promoted to a 2D array of shape (1, N).
                For a 2D array of shape (avg, N), DFT is applied along the last axis whereas the average is applied along the first axis.
        '''
        if len(data.shape) == 1:
            y_n = data.reshape(1,-1)
        else:
            y_n = data
        self._N = y_n.shape[1]
        self.y_k = fftshift(fft(y_n))

    def aim(self, band):
        '''
        Focus on a narrow frequency band where the sought harmonics reside.
        -----
        Arguments:
            band: float 2-tuple
                Lower and upper limits of the designed frequency band, normalized into the range of (-1/2, 1/2).
                They will be rounded to the nearest points on the frequency grid.
        '''
        self.freq = fftshift(fftfreq(self._N))
        self.idx_l, self.idx_u = np.argmin(np.abs(self.freq - np.array(band).reshape(2,1)), axis=1)

    def shoot(self, J_c, rho):
        '''
        Estimate the harmonic frequencies with the passed on a priori.
        -----
        Arguments:
            J_c: int
                Number of harmonics in the designated band.
            rho: float number or 1D float array
                Power spectral density of the noise in the designated band.
        -----
        Returns:
            fhat: float 1D array
                Estimate of the harmonic frequencies in the designated band in ascending order.
            val: float 1D array
                Eigenvalues of the estimate of the harmonic term in descending order.
        '''
        Z = polyvander(np.exp(2j*np.pi*self.freq[self.idx_l:self.idx_u+1]), J_c+1).T[1:]
        Q = null_space(Z)
        D = np.diag(self._N*rho * np.sum(np.abs(Q)**2, axis=1))
        YQ = self.y_k[:,self.idx_l:self.idx_u+1,np.newaxis] * Q[np.newaxis,:,:]
        YPYh = np.mean(YQ@YQ.conj().transpose(0,2,1), axis=0)
        val, Vec = eigh(YPYh-D) # eigenvalues in ascending order and the corresponding eigenvectors
        C_c = Z @ Vec[:,:-J_c-1:-1]
        A_c = self._tls(C_c[:-1], C_c[1:])
        fhat = np.angle(eigvals(A_c)) / (2*np.pi)
        return np.sort(fhat), val[::-1]

    def _tls(self, A, B):
        '''
        Solve the matrix equation AX=B for the unknown X in the sense of total least squares.
        -----
        Arguments:
            A: 2D complex array of shape (m, n)
            B: 2D complex array of shape (m, n)
        -----
        Returns:
            X: 2D complex array of shape (n, n)
        '''
        n = A.shape[1]
        Vh = svd(np.hstack((A,B)))[-1]
        V_2h, V_4h = Vh[n:,:n], Vh[n:,n:]
        X = solve(V_4h, -V_2h).T.conj()
        return X


if __name__ == "__main__":
    from scipy.stats import norm

    f_0, rho = np.array((.19, .21)), 100
    print("The set frequencies are {:g} and {:g}.".format(*f_0))
    signal = np.exp(2j*np.pi*f_0[0]*np.arange(10000)) + np.exp(2j*np.pi*f_0[1]*np.arange(10000))
    noise = norm.rvs(scale=np.sqrt(rho/2), size=20000).view(complex) # 10000 complex normally distributed random numbers with the power spectral density of rho
    data = (signal + noise).reshape(10, 1000)

    haret = Harm_Retri()
    haret.load(data)
    haret.aim((.17, .23))
    fhat = haret.shoot(2, rho)[0]
    print("The retrieved harmonics are at {:.5f} and {:.5f}.".format(*fhat))
