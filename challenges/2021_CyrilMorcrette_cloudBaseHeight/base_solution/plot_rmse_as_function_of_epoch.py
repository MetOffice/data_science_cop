#!/usr/bin/env python	
#
# Cyril Morcrette (2020), Met Office, UK
#
# Use "module load scitools/experimental-current" at the command line before running this.

# Import some modules
import numpy as np
import matplotlib.pyplot as plt

filein='cbh_ml_timeseries_rmse_train.txt'
train=np.loadtxt(filein)

filein='cbh_ml_timeseries_rmse_val.txt'
val=np.loadtxt(filein)

tmp=train.shape
nt=tmp[0]

train=train[1:nt]
val=val[1:nt]

# Since there are 124 files to read in before an epoch is completed
# and since the cbh_ml_timeseries_rmse_*.txt files are updated after training on each file
# one needs to divide by 124 to get a correct epoch counter.
time=np.arange(1,nt)/124.0

plt.plot(time,train,'k+-')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.plot(time,val,'b+-')
fig.savefig('cbh_ml_rmse_plot.png')

