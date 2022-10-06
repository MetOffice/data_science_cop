#!/usr/bin/env python	
#
# Cyril Morcrette (2020), Met Office, UK
#
# Use "module load scitools/experimental-current" at the command line before running this.
#
# Import some modules (possibly more than are strictly needed, but I did not bother stripping out what was not needed).

import numpy as np

import matplotlib.pyplot as plt
import iris
from netCDF4 import Dataset

from cjm_functions import make_stash_string
from cjm_functions import load_in_one_file_for_cbh_ml

from keras.models import model_from_json
import iris.quickplot as qplt

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide tensorflow log message about performance limitations
import warnings
warnings.filterwarnings("ignore")
def main():

    #Point to the file that we are using for independent test (i.e. data NOT used in training!)
    pp_file='/data/nwp1/frme/ML_minichallenge/dev/20170701T0000Z_glm_pa010.pp'
    result = make_stash_string(0,266)
    bcf = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))

    nz,ny,nx=bcf.shape
    true_cbh_3d=np.copy(bcf)*0.0    
    true_cbh_2d=np.copy(bcf)*0.0  

    # Set a threshold for determining that cloud base has been found (e.g. 2 oktas)
    thresh=2.0/8.0

    # Simple search algorithm to find cloud base working from the bottom up.
    # If no cloud base is found, then cloud-base is set to be in top most layer 
    # (NB real cloud base is never that high).
    for j in np.arange(0,ny,1):
        for i in np.arange(0,nx,1):
            found=0
            for k in np.arange(0,nz,1):
                if found==0 and bcf.data[k,j,i]>thresh:
                    true_cbh_3d.data[k,j,i]=1
                    true_cbh_2d.data[0,j,i]= np.float64(k)
                    found=1
            if found==0:
                true_cbh_2d.data[0,j,i]=69.0

    # Read in the temp, qv and pressure data for use as inputs for predicting CBH.
    result=load_in_one_file_for_cbh_ml(pp_file)

    # Read in the details of the machine learning model (i.e. number of layer and nodes etc).
    # This was written out by cbh_ml.py
    json_file=open('./cbh_ml_model.json', 'r')
    loaded_model_json=json_file.read()
    json_file.close()
    model=model_from_json(loaded_model_json)

    # Now read in the most recent weights saved when training that model.
    model.load_weights('./cbh_ml_latest_saved_weights.h5')

    big_data=np.transpose(result['data'])
    X=big_data[:,  0:210]
    # X contains temp, qv, pres. Note that we are not making use of the last 70 entries (which are the answer)
    # This array has 210 columns and as many rows as the product of lat and lon points in the model domain.

    # Make a prediction
    y= model.predict(X)

    # Manipulate array into a nicer shape
    # i.e. reshape back into 3d with height, lat, lon (here hard-wired for N320 model data).
    predicted_cbh_3d=np.reshape(np.transpose(y),(70,480,640))
    # and put it into a cube so it can be written out using iris save.
    predicted_cbh_3d_CUBE=np.copy(bcf)*0.0
    predicted_cbh_3d_CUBE.data=predicted_cbh_3d

    # Create new array (but only use k=0 level for storing the level number at which the cloud base is found).
    predicted_cbh_2d=np.copy(bcf)*0.0

    # At each location, the ML has output a column of data which is the probability of the cloud base being at that level.
    # Set cloud base to be at whichever level has the highest probability.
    for j in np.arange(0,ny,1):
        for i in np.arange(0,nx,1):
            tmp=predicted_cbh_3d[:,j,i]
            ind=np.where(tmp==np.amax(tmp))
            ind=np.float64(ind[0])
            predicted_cbh_2d.data[0,j,i]=ind

    # Maps
    fig = plt.figure()
    ax=fig.add_subplot(3,1,1)
    qplt.pcolormesh(true_cbh_2d[0,:,:],vmin=0.0, vmax=50.0)
    ax=fig.add_subplot(3,1,2)
    qplt.pcolormesh(predicted_cbh_2d[0,:,:],vmin=0.0, vmax=50.0)
    ax=fig.add_subplot(3,1,3)
    diff=predicted_cbh_2d[0,:,:]-true_cbh_2d[0,:,:]
    qplt.pcolormesh(diff,vmin=-20, vmax=20)
    plt.show()
    fileout='cbh_ml_maps.png'
    plt.savefig(fileout)
    plt.close()
    iris.save(true_cbh_2d[0,:,:], 'map_true_cbh.nc')
    iris.save(predicted_cbh_2d[0,:,:], 'map_predicted_cbh.nc')

    # Print some error metrics (NB these will be affected by cases where no cloud base is found and hence it is said to be in the top-most level.
    print('Mean Error=',np.mean(diff.data))
    print('Mean Absolute Error=',np.mean(np.abs(diff.data)))
    print('Root Mean Squared Error=',np.sqrt(np.mean((diff.data)**2.0)))

    # Cross section
    fig = plt.figure()
    ax=fig.add_subplot(3,1,1)
    plt.pcolormesh(true_cbh_3d.data[:,:,0],vmin=0.0, vmax=1.0)
    ax=fig.add_subplot(3,1,2)
    plt.pcolormesh(predicted_cbh_3d[:,:,0],vmin=0.0, vmax=1.0)
    ax=fig.add_subplot(3,1,3)
    diff=predicted_cbh_3d[:,:,0]-true_cbh_3d.data[:,:,0]
    plt.pcolormesh(diff,vmin=-1, vmax=1)
    plt.show()
    fileout='cbh_ml_crosssection.png'
    plt.savefig(fileout)
    plt.close()
    iris.save(true_cbh_3d, 'true_cbh.nc')
    iris.save(predicted_cbh_3d_CUBE, 'predicted_cbh.nc')

    # Now lets plot some profiles to see what the ML has predicted
    temp=X[:,0:70]
    qv=X[:,70:140]
    pres=X[:,140:210]

    temp=np.reshape(np.transpose(temp),(70,480,640))
    qv=np.reshape(np.transpose(qv),(70,480,640))
    pres=np.reshape(np.transpose(pres),(70,480,640))

    max_temp=320.0
    min_temp=140.0
    max_qv=0.025
    max_pres=105000.0
    #Un-Normalise/standardise (put back into physical space).
    temp=(temp*(max_temp-min_temp))+min_temp
    qv=qv*max_qv

    lev=np.arange(0,70,1)

    fig, axs = plt.subplots(4, 4)
    nj=240 # We are sort of randomly selecting 4 places to plot profiles for just for illustrating what going on.
    ni=0
    im=axs[0,0].plot(temp[:,nj,ni],lev,'k-')
    im=axs[0,0].set_xlabel('Temp [K]')
    im=axs[0,0].set_ylabel('Level')
    im=axs[0,1].plot(qv[:,nj,ni],lev,'k-')
    im=axs[0,1].set_xlabel('qv [kg/kg]')
    im=axs[0,1].set_ylabel('Level')
    im=axs[0,2].plot(bcf.data[:,nj,ni],lev,'k-')
    im=axs[0,2].plot([-0.1,0],[true_cbh_2d.data[0,nj,ni],true_cbh_2d.data[0,nj,ni]],'r-')
    im=axs[0,2].set_xlabel('BCF [-]')
    im=axs[0,2].set_ylabel('Level')
    #
    im=axs[0,3].plot(predicted_cbh_3d[:,nj,ni],lev,'b-')
    im=axs[0,3].set_xlabel('Prob CBH [-]')
    im=axs[0,3].set_ylabel('Level')
    tmp=predicted_cbh_3d[:,nj,ni]
    ind=np.where(tmp==np.amax(tmp))
    ind=np.float64(ind[0])
    im=axs[0,3].plot([-0.1,0],[ind,ind],'g-')
    nj=400
    ni=200
    im=axs[1,0].plot(temp[:,nj,ni],lev,'k-')
    im=axs[1,0].set_xlabel('Temp [K]')
    im=axs[1,0].set_ylabel('Level')
    im=axs[1,1].plot(qv[:,nj,ni],lev,'k-')
    im=axs[1,1].set_xlabel('qv [kg/kg]')
    im=axs[1,1].set_ylabel('Level')
    im=axs[1,2].plot(bcf.data[:,nj,ni],lev,'k-')
    im=axs[1,2].plot([-0.1,0],[true_cbh_2d.data[0,nj,ni],true_cbh_2d.data[0,nj,ni]],'r-')
    im=axs[1,2].set_xlabel('BCF [-]')
    im=axs[1,2].set_ylabel('Level')
    #
    im=axs[1,3].plot(predicted_cbh_3d[:,nj,ni],lev,'b-')
    im=axs[1,3].set_xlabel('Prob CBH [-]')
    im=axs[1,3].set_ylabel('Level')
    tmp=predicted_cbh_3d[:,nj,ni]
    ind=np.where(tmp==np.amax(tmp))
    ind=np.float64(ind[0])
    im=axs[1,3].plot([-0.1,0],[ind,ind],'g-')
    nj=60
    ni=400
    im=axs[2,0].plot(temp[:,nj,ni],lev,'k-')
    im=axs[2,0].set_xlabel('Temp [K]')
    im=axs[2,0].set_ylabel('Level')
    im=axs[2,1].plot(qv[:,nj,ni],lev,'k-')
    im=axs[2,1].set_xlabel('qv [kg/kg]')
    im=axs[2,1].set_ylabel('Level')
    im=axs[2,2].plot(bcf.data[:,nj,ni],lev,'k-')
    im=axs[2,2].plot([-0.1,0],[true_cbh_2d.data[0,nj,ni],true_cbh_2d.data[0,nj,ni]],'r-')
    im=axs[2,2].set_xlabel('BCF [-]')
    im=axs[2,2].set_ylabel('Level')
    #
    im=axs[2,3].plot(predicted_cbh_3d[:,nj,ni],lev,'b-')
    im=axs[2,3].set_xlabel('Prob CBH [-]')
    im=axs[2,3].set_ylabel('Level')
    tmp=predicted_cbh_3d[:,nj,ni]
    ind=np.where(tmp==np.amax(tmp))
    ind=np.float64(ind[0])
    im=axs[2,3].plot([-0.1,0],[ind,ind],'g-')
    nj=360
    ni=550
    im=axs[3,0].plot(temp[:,nj,ni],lev,'k-')
    im=axs[3,0].set_xlabel('Temp [K]')
    im=axs[3,0].set_ylabel('Level')
    im=axs[3,1].plot(qv[:,nj,ni],lev,'k-')
    im=axs[3,1].set_xlabel('qv [kg/kg]')
    im=axs[3,1].set_ylabel('Level')
    im=axs[3,2].plot(bcf.data[:,nj,ni],lev,'k-')
    im=axs[3,2].plot([-0.1,0],[true_cbh_2d.data[0,nj,ni],true_cbh_2d.data[0,nj,ni]],'r-')
    im=axs[3,2].set_xlabel('BCF [-]')
    im=axs[3,2].set_ylabel('Level')
    #
    im=axs[3,3].plot(predicted_cbh_3d[:,nj,ni],lev,'b-')
    im=axs[3,3].set_xlabel('Prob CBH [-]')
    im=axs[3,3].set_ylabel('Level')
    tmp=predicted_cbh_3d[:,nj,ni]
    ind=np.where(tmp==np.amax(tmp))
    ind=np.float64(ind[0])
    im=axs[3,3].plot([-0.1,0],[ind,ind],'g-')
    plt.show()
    fileout='cbh_ml_profiles.png'
    plt.savefig(fileout)
    plt.close()

    # Prepare to calculate some skill scores.
    true=np.reshape(true_cbh_2d.data[0,:,:],(1,480*640))
    pred=np.reshape(predicted_cbh_2d.data[0,:,:],(1,480*640))

    store_seds=0
    store_sedi=0
    [tmp, nn]=true.shape

    # Only go up to (but not including) level 53, around 20km up
    for k in np.arange(0,53,1):
        a=0.0
        b=0.0
        c=0.0
        d=0.0
        for i in np.arange(0,nn):
            # Is the cloud-base at this level or below.
            if   true[0,i]<=np.float64(k) and pred[0,i] <=np.float64(k):
                # Hit
                a=a+1.0
            elif true[0,i]> np.float64(k) and pred[0,i] <=np.float64(k):
                # False alarm
                b=b+1.0
            elif true[0,i]<=np.float64(k) and pred[0,i] > np.float64(k):
                # Miss
                c=c+1.0
            else:
                # Correct negative
                d=d+1.0
        if a == 0.0 or b == 0.0:
            a += 1 
            b += 1
        n=a+b+c+d
        ar=((a+b)*(a+c))/n;
        # Symmetric Extreme Dependency Score (SEDS) has advantage that it is 1.0 for perfect forecast
        # and 0.0 for no better than climatology 
        # (unlike Equitable Threat Score [ETS] for which it is not clear at what point a low score means it is poor.
        seds=np.log(ar/a)/np.log(a/n)
        store_seds=np.append(store_seds,seds)
        # Also calculate SEDI from Ferro and Stephenson (equation 2)
        # https://journals.ametsoc.org/view/journals/wefo/26/5/waf-d-10-05030_1.xml
        h      = a / (a+c)
        f      = b / (b+d)
        top    = np.log(f) - np.log(h) - np.log(1.0-f) + np.log(1.0-h)
        bottom = np.log(f) + np.log(h) + np.log(1.0-f) + np.log(1.0-h)
        sedi   = top / bottom
        store_sedi=np.append(store_sedi,sedi)

    # Calculate mean values, write it out and print it out.
    mean_seds=np.mean(store_seds[1:])
    fileout='cbh_ml_mean_seds.a.txt'
    np.savetxt(fileout, np.ones((1,1))*mean_seds, fmt='%10.7f')
    mean_sedi=np.mean(store_sedi[1:])
    fileout='cbh_ml_mean_sedi.a.txt'
    np.savetxt(fileout, np.ones((1,1))*mean_sedi, fmt='%10.7f')
    #
    print('Mean SEDS and SEDI',mean_seds,mean_sedi)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

