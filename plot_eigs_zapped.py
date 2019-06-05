import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np
import sys
import os

NCHANS = 336

data = np.fromfile(sys.argv[1],dtype=np.int32)
data = data[:data.shape[0]/NCHANS*NCHANS]
data = data.reshape((-1,NCHANS)).T


plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])
gs.update(wspace = 0)
ax1 = plt.subplot(gs[0])
d = data.mean(axis=1)
ax1.plot(d,range(NCHANS))
ax1.set_xlabel("Mean eigvec zapped", fontsize=18)
ax1.set_ylabel("Frequency channel index", fontsize=18)
ax1.set_xticks(range(0,int(np.max(d))+1,2))
ax1.tick_params(labelsize=16)
ax1.set_ylim(0,NCHANS-1)

ax2 = plt.subplot(gs[1])
a = ax2.imshow(np.flipud(data),interpolation='nearest',aspect='auto')
ax2.set_xlabel("Block index", fontsize=18)
ax2.tick_params(labelsize=16)
ax2.set_yticklabels([])
cbar = plt.colorbar(a,ax=ax2)
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Number of eigenvectors removed", fontsize=18)

plt.title(os.path.basename(sys.argv[1]))

plt.show()
