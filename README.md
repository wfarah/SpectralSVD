# SpectralSVD

A method for RFI mitigation based on the eigenvector decomposition of the (total intensity) covariance matrix of multibeam-systems. The code reads/writes data from/to PSRDADA ring buffers.

## Requires:
* [Intel(R) MKL](https://software.intel.com/en-us/mkl) or [LAPACKE](https://www.netlib.org/lapack/lapacke.html) + [BLAS](http://www.netlib.org/blas/)
* [PSRDADA](http://psrdada.sourceforge.net/download.shtml)
