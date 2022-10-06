# Cyril Morcrette (2020), Met Office, UK
#
# Use "module load scitools/experimental-current" at the command line before running this.

# Import some modules
import numpy as np
import matplotlib.pyplot as plt

filein='cbh_ml_timeseries_acc_train.txt'
train=np.loadtxt(filein)

with open('cbh_ml_timeseries_acc_val.txt') as f:
        val=np.loadtxt(f)
tmp=train.shape
nt=tmp[0]

train=train[1:nt]
val=val[1:nt]

time=np.arange(1,nt)/124.0
plt.plot(time,train,'k+-')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.plot(time,val,'b+-')
plt.show()
plt.figure().savefig('cbh_ml_acc_plot.png')
