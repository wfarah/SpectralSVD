rc_params = {"figure.figsize": (14,12),
             "xtick.major.size": 9,
             "xtick.minor.size": 4,
             "ytick.major.size": 9,
             "ytick.minor.size": 4,
             "xtick.major.width": 2,
             "xtick.minor.width": 1,
             "ytick.major.width": 2,
             "ytick.minor.width": 1,
             "xtick.major.pad": 8,
             "xtick.minor.pad": 8,
             "ytick.major.pad": 8,
             "ytick.minor.pad": 8,
             "lines.linewidth": 2,
             "lines.markersize": 10,
             "axes.linewidth": 2,
             "legend.loc": "best",
             "xtick.labelsize" : 22,
             "ytick.labelsize" : 22,
             "font.size": 24,
             "font.family" : "Times New Roman",
             "font.weight": "normal",
             "xtick.direction": "in",
             "ytick.direction": "in",
             }
import matplotlib
matplotlib.rcParams.update(rc_params)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams.update(rc_params)
from matplotlib import cm

import numpy as np
import sys
import os
import argparse

def main(args):
    NCHANS = args.nchans

    dm0_exists = False
    if os.path.exists(args.eigfile+".0dm"):
        data0dm = np.fromfile(args.eigfile+".0dm", dtype=np.uint8)
        dm0_exists = True
    data = np.fromfile(args.eigfile,dtype=np.int32)
    data = data[:data.shape[0]/NCHANS*NCHANS]
    data = data.reshape((-1,NCHANS)).T


    plt.figure()
    
    if dm0_exists:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 5], height_ratios=[7,1])
    else:
        gs = gridspec.GridSpec(2, 1, width_ratios=[1, 5])

    gs.update(wspace = 0)
    ax1 = plt.subplot(gs[0])
    d = data.mean(axis=1)
    ax1.plot(d,range(NCHANS))
    ax1.set_xlabel("Mean eigvec zapped")
    ax1.set_ylabel("Frequency channel index")
    ax1.set_xticks(range(0,int(np.max(d))+1,2))
    ax1.tick_params()
    ax1.set_ylim(0,NCHANS-1)

    ax2 = plt.subplot(gs[1])
    a = ax2.imshow(np.flipud(data),interpolation='nearest',aspect='auto')
    ax2.set_xlabel("Block index")
    ax2.tick_params()
    ax2.set_yticklabels([])
    cbar = plt.colorbar(a,ax=ax2)
    cbar.ax.tick_params()
    cbar.set_label("Number of eigenvectors removed")
    _xlim =  plt.xlim()

    plt.title(os.path.basename(args.eigfile))

    if dm0_exists:
        ax3 = plt.subplot(gs[3])
        plt.plot(data0dm)
        plt.xlim(_xlim)
        plt.xlabel("Block index")
        plt.ylabel("0DM")

    if not args.savepath:
        plt.show()
    else:
        plt.savefig(args.savepath, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Plots the mean eigenvectors zero-ed')

    parser.add_argument(type=str, dest='eigfile',
            help = 'input file')
    parser.add_argument('-n', type=int, dest='nchans',
            help = 'nchans (default=336)',
            default=336)
    parser.add_argument('-s', type=str, dest='savepath',
            help = 'path to save plot (plt.show() otherwise)')

    args = parser.parse_args()

    main(args)

