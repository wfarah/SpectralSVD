import numpy as np
import matplotlib.pyplot as plt
import glob
import os,sys

from sigpyproc.Readers import FilReader

def get_neig(eigs, total_eigs, min_eigs=5, thresh=1.25):
    rmean = 0
    neig = 0
    eigs_r = eigs[::-1].copy()
    
    for i in range(min_eigs):
        rmean += eigs_r[i]
    rmean /= min_eigs

    for i in range(min_eigs, total_eigs):
        if (eigs_r[i]/rmean > thresh):
            neig += 1
        else:
            rmean = (rmean*(i-1) + eigs_r[i])/i

    return neig

def AIC(eigs, nsamps=256):
    aic = np.zeros((len(eigs)))
    neigs = len(eigs)
    for i in range(len(aic)):
        a = reduce(np.multiply,eigs[i:])
        b = 1./(neigs - i) * np.sum(eigs[i:])
        b = b**(neigs - i)
        aic[i] = -2*nsamps*np.log(a/b) + 2*i*(2*neigs - i)

    return aic

def MDL(eigs, nsamps=256):
    mdl = np.zeros((len(eigs)))
    neigs = len(eigs)
    for i in range(len(mdl)):
        a = reduce(np.multiply,eigs[i:])
        b = 1./(neigs - i) * np.sum(eigs[i:])
        b = b**(neigs - i)
        mdl[i] = -nsamps*np.log(a/b) + 0.5*i*(2*neigs - i)*np.log(nsamps)

    return mdl


BASEDIR = "/fred/oz002/users/wfarah/askap/RFI_mitigation/SB08772/20190515082551"

fil_files = []

for ffile in sorted(glob.glob(os.path.join(BASEDIR,"*.fil"))):
    fil_files.append(FilReader(ffile))

nsamps_per_block = 256
nchans = fil_files[0].header.nchans
ntotal = fil_files[0].header.nsamples
nbeams = len(fil_files)

block_read = np.zeros((nbeams, nchans, nsamps_per_block))
work_arr = np.zeros((nbeams, nsamps_per_block))


for ibeam,beam in enumerate(range(nbeams)):
    block_read[ibeam] = fil_files[ibeam].readBlock(0, nsamps_per_block)

for ichan in range(nchans):
    work_arr[:] = block_read[:, ichan, :]
    U, W, Vt = np.linalg.svd(work_arr, full_matrices=False)
    print W
