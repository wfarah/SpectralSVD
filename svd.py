#!/home/wfarah/miniconda2/bin/python

import numpy as np
from sigpyproc.Readers import FilReader
import sys,os
import glob
import argparse
import atexit

try:
    from progress.bar import Bar
except ImportError as e:
    sys.stderr.write("pip install 'progress' module to display a progress bar\n")
    USEPBAR = False
else:
    USEPBAR = True

def get_mad(arr):
    med = np.median(arr, axis=1)
    mad = np.median(np.abs(arr.T - med), axis=0)
    return mad
    
def neigs_to_zap(eigvals, thresh=1.3):
    neigs_zap = 0
    for i in range(len(eigvals) - 1):
        if eigvals[i]/eigvals[i+1] > thresh:
            neigs_zap += 1
    return neigs_zap

def main(args):
    inputlist = args.inlist
    outdir = args.outdir
    neig = args.neig
    nbeams = len(inputlist)

    outnames = [os.path.join(outdir,"cleaned_"+os.path.basename(i)) 
            for i in inputlist]

    if len(set(outnames)) < nbeams:
        sys.stderr.write("ERR: please do not use same basename for input filterbank files\n")
        sys.exit()

    inputfils = []

    for beam in inputlist:
        inputfils.append(FilReader(beam))


    nsamples = inputfils[0].header.nsamples
    nchans = inputfils[0].header.nchans
    samp_per_block = args.samps

    work = np.zeros((nbeams, nchans, samp_per_block), dtype=np.float32)

    if inputfils[0].header.nbits == 8:
        rescale = True
        for inputfil in inputfils:
            inputfil.header.nbits = 32
    else:
        rescale = False

    outfiles = [inputfils[ifile].header.prepOutfile(i) for ifile,i in enumerate(outnames)]

    total_n_blocks = nsamples/samp_per_block

    statfile = args.statfile
    if not statfile.endswith(".eig"):
        statfile += ".eig"

    eig_file = open(args.statfile,"w")
    atexit.register(eig_file.close)

    if USEPBAR:
        pbar = Bar('Processing filterbanks', 
                max=total_n_blocks, suffix='%(percent)d%%')
        atexit.register(pbar.finish)

    for iblock in range(total_n_blocks):
        nzaps = []

        # Extract data
        for ibeam in range(nbeams):
            work[ibeam] = inputfils[ibeam].readBlock(iblock*samp_per_block, samp_per_block)

            if rescale:
                work[ibeam] = (work[ibeam].T - np.median(work[ibeam], axis=1)).T
                std = 1.4826*get_mad(work[ibeam])
                mask = np.abs(std) > 1e-12
                work[ibeam][mask] = (work[ibeam][mask].T / std[mask]).T

        # Do SVD for each data block
        for ichan in range(nchans):
            try:
                U, W, Vt = np.linalg.svd(work[:,ichan,:], full_matrices=False)
            except Exception as e:
                np.save("debug.npy",work[:,ichan,:])
                sys.stderr.write("Exception caught: saving ./debug.npy\n")
                raise e
            neig = neigs_to_zap(W)
            nzaps.append(neig)
            if neig > 0:
                Y = np.dot(work[:,ichan,:], Vt.T)
                Vt.T[:,:neig] = 0.0
                work[:,ichan,:] = np.dot(Y, Vt)
        
        # Write output nzap file
        nzaps = np.array(nzaps, dtype=np.int32)
        nzaps.tofile(eig_file)

        # Write output
        for ibeam,outfile in enumerate(outfiles):
            if args.resc_out:
                work[ibeam] = (work[ibeam].T - np.median(work[ibeam], axis=1)).T
                std = 1.4826*get_mad(work[ibeam])
                mask = ((np.abs(std) > 1e-12) & (nzaps > 0))
                work[ibeam][mask] = (work[ibeam][mask].T / std[mask]).T
            work[ibeam].T.astype("float32").tofile(outfile)

        
        if USEPBAR:
            pbar.next()

    if USEPBAR:
        pbar.finish()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Performs eigenflagging on\
              filterbank data')

    parser.add_argument('inlist', nargs='+',
            help='Input files')
    parser.add_argument('-o', dest='outdir', type=str,
            help='Output directory (default=./)', 
            default="./")
    parser.add_argument('-k', dest='neig', type=int,
            help='Number of eigenvectors to remove (default=1)', 
            default=1)
    parser.add_argument('-n', dest='samps', type=int,
            help='Number of samples per block (default=256)',
            default=256)
    parser.add_argument('-s', dest='statfile', type=str,
            help='Eigen-statistics file (default=out.eig)',
            default="./out.eig")
    parser.add_argument('-r', dest='resc_out',
            help='Rescale output', action='store_true')

    args = parser.parse_args()
    main(args)
