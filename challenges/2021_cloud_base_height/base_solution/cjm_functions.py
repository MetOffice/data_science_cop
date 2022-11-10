#!/usr/bin/env python	
#
# Cyril J Morcrette (2018), Met Office, UK
#
# Import some modules

from netCDF4 import Dataset
from datetime import timedelta
import numpy as np
import iris
import iris.analysis
import subprocess
import matplotlib.pyplot as plt
import warnings

def daterange(start_date, end_date):
    # This does include the end_date
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def make_region_str(region_number):
    if region_number < 10:
        region_str='r0'+str(region_number)
    else:
        region_str='r'+str(region_number)
    return region_str;

def retrieve_all_files_for_a_region_and_day(roseid,date,region_number,flag,inpath):
    #,single_time_stamp,name_str,analysis_time):
    # If flag == 1, it deletes the files and then retrieves a clean copy
    # If flag == 0, it just deletes the files.
    #
    # Ensure the directory exists for putting the spatially averaged data into.
    tmppath=inpath+'netcdf/'+date.strftime("%Y%m%d")+'/'
    subprocess.call(["mkdir", tmppath])
    region_str=make_region_str(region_number)
    tmppath=inpath+'netcdf/'+date.strftime("%Y%m%d")+'/'+region_str+'/'
    subprocess.call(["mkdir", tmppath])
    #
    # Ensure the directory exists for storing the raw pp files retrieved from mass.
    tmppath=inpath+'pp/'+date.strftime("%Y%m%d")+'/'
    subprocess.call(["mkdir", tmppath])
    #
    # moo retrieval will not overwrite an existing file.
    # So remove any existing occurrence of the files we are about to retrieve.
    # Does not like having a * when deleting (so manually specify a,b,c,d)
    filenames=date.strftime("%Y%m%d")+'T0000Z_'+region_str+'_km3p3_RA2M_pa000.pp'
    stalefiles=tmppath+filenames
    subprocess.call(["rm", stalefiles])
    filenames=date.strftime("%Y%m%d")+'T0000Z_'+region_str+'_km3p3_RA2M_pb000.pp'
    stalefiles=tmppath+filenames
    subprocess.call(["rm", stalefiles])
    filenames=date.strftime("%Y%m%d")+'T0000Z_'+region_str+'_km3p3_RA2M_pc000.pp'
    stalefiles=tmppath+filenames
    subprocess.call(["rm", stalefiles])
    filenames=date.strftime("%Y%m%d")+'T0000Z_'+region_str+'_km3p3_RA2M_pd000.pp'
    stalefiles=tmppath+filenames
    subprocess.call(["rm", stalefiles])
    #
    # Seems happy to retrieve files with a *, though, so retrieve several in one go.
    filenames=date.strftime("%Y%m%d")+'T0000Z_'+region_str+'*.pp'
    if flag==1:
        moopath='moose:/devfc/'+roseid+'/field.pp/'
        fullname=moopath+filenames
        subprocess.call(["moo","get", fullname, tmppath])
    # endif
    outcome=1
    return outcome;

def loop_over_fields(single_date,roseid,name_str,analysis_time,region_str,list_stream,list_stash_sec,list_stash_code,list_ndims,tmppath):
    print('Processing region '+region_str)
    print('list_stash_code=',list_stash_code)
    for stream_number in np.arange(0,len(list_stream),1):
        ndims=list_ndims[stream_number]
        stream=list_stream[stream_number]
        stash_sec=list_stash_sec[stream_number]
        stash_code=list_stash_code[stream_number]
        print('About to read in',ndims,'d field STASH',stash_sec,stash_code)
        outcome=read_in_and_cut_out_multi_timesteps(single_date,0,roseid,name_str,analysis_time,stream,region_str,stash_sec,stash_code,ndims,tmppath)
    # end (stream_number)
    outcome=1
    return outcome;

def read_in_and_cut_out_multi_timesteps(date,vt,roseid,name_str,analysis_time,stream,region_str,stash_sec,stash_code,ndims,tmppath,single_time_stamp):
    # This reads in the data and only keeps the inner portion (away from the spin-up outer rim).
    filename=generate_ml_filename_in(date,vt,'.pp',stream,name_str,analysis_time,region_str)
    filein=tmppath+'pp/'+filename
    # Read in data
    result = make_stash_string(stash_sec,stash_code)
    fieldin = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    # Input array is 512 x 512. Assume we chop this up into a 16 x 16 array (ready for 32 x 32 averaging)
    # ignore the outer rim and take the central 14 x 14 (14 x 32 =448) so take the central 448 x 448.
    #
    # Check on whether the field being processes is 2d or 3d (actually 3d or 4d with time dimension)
    if ndims==2:
        data=fieldin[32:480,32:480]    # There is no time dimension as field was only written once (as it is constant in time).
    if ndims==3:
        data=fieldin[single_time_stamp,32:480,32:480]  # First dimension is time, next two are y and x.
    if ndims==4:
       data=fieldin[single_time_stamp,:,32:480,32:480] # First dimension is time, second is height, last two are y and x.
    # Now call the function that extracts the individual GCM-sized regions and processes them.
    outcome=process_subdomains(data,date,single_time_stamp,name_str,analysis_time,region_str,result['stashstr_fout'],ndims,tmppath)
    return outcome;

def process_subdomains(data,date,vt,name_str,analysis_time,region_str,output_label,ndims,tmppath):
    # Function that extracts the LAM data over an area corresponding to a GCM grid-boxes
    # and then processes it and writes it out.
    # The processing can be calculating the mean, standard deviation or the skewness.
    #
    # The data is of size (time,lat,lon),        i.e. (time,y,x)   so indexing is (:,j,i)
    #                or   (time,levels,lat,lon), i.e. (time,z,y,x) so indexing is (:,:,j,i)
    #
    # This next array could be populated further to aggregate over a different number of gridpoints (i.e. different "GCM" grid-sizes).
    size_of_subdomains=[32]
    # e.g. size_of_subdomains=[30,60,120]
    for scale in np.arange(0,len(size_of_subdomains),1):
        dx=size_of_subdomains[scale]
        nx=np.int(448/dx)
        # Hacky solution! Make use of cube info from input data
        # (with some of the right metadata properties to fill with the spatially averaged data) 
        # hence allowing it to be written out using iris save.
        # Note that the lat/lon in this cube will be wrong as it just uses the SW corner of the bigger original array, 
        # rather than redefining the lat and lon, but I do not care. I can always work it out correctly if needed.
        if ndims==2:
            # 2d, i.e. 2d fields fixed in time
            coarse_grained_mean      = np.copy(data[0:nx,0:nx])*0.0
            coarse_grained_std       = np.copy(data[0:nx,0:nx])*0.0
            coarse_grained_skew      = np.copy(data[0:nx,0:nx])*0.0
            moments                  = cut_up_2d_array_into_regions_and_find_moments(data,nx,dx)
            coarse_grained_mean.data = moments['mean_2d']
            coarse_grained_std.data  = moments['std_2d']
            coarse_grained_skew.data = moments['skew_2d']
        if ndims==3:
            # 3d, i.e. 2d fields varying in time
            coarse_grained_mean      = np.copy(data[0:nx,0:nx])*0.0
            coarse_grained_std       = np.copy(data[0:nx,0:nx])*0.0
            coarse_grained_skew      = np.copy(data[0:nx,0:nx])*0.0
            moments                  = cut_up_2d_array_into_regions_and_find_moments(data,nx,dx)
            coarse_grained_mean.data = moments['mean_2d']
            coarse_grained_std.data  = moments['std_2d']
            coarse_grained_skew.data = moments['skew_2d']
        if ndims==4:
            # 4d, i.e. 3d fields varying in time
            coarse_grained_mean      = np.copy(data[:,0:nx,0:nx])*0.0
            coarse_grained_std       = np.copy(data[:,0:nx,0:nx])*0.0
            coarse_grained_skew      = np.copy(data[:,0:nx,0:nx])*0.0
            moments                  = loop_over_3rd_dim(data,nx,dx)
            coarse_grained_mean.data = moments['mean_3d']
            coarse_grained_std.data  = moments['std_3d']
            coarse_grained_skew.data = moments['skew_3d']
        #
        filenameout=generate_mlarray_filename_out(date,vt,'.nc',name_str,analysis_time,region_str,dx,nx,output_label,'AVG')
        fileout=tmppath+'netcdf/'+filenameout
        iris.save(coarse_grained_mean, fileout)
        #
        filenameout=generate_mlarray_filename_out(date,vt,'.nc',name_str,analysis_time,region_str,dx,nx,output_label,'STD')
        fileout=tmppath+'netcdf/'+filenameout
        iris.save(coarse_grained_std, fileout)
        #
        filenameout=generate_mlarray_filename_out(date,vt,'.nc',name_str,analysis_time,region_str,dx,nx,output_label,'SKW')
        fileout=tmppath+'netcdf/'+filenameout
        iris.save(coarse_grained_skew, fileout)
        # Write out a file (containing number 1.0) to say that this field has been processed yet).
        filenameout=generate_flag_name(date,vt,'.dt',name_str,analysis_time,region_str,dx,nx,output_label,'FLG')
        fileout=tmppath+'flags/'+filenameout
        np.savetxt(fileout, [1.0], fmt='%4.2f')
    # end s
    outcome=1
    return outcome;

def loop_over_4th_dim(data4d,nx,dx):
    nt,nz,tmp1,tmp2 = data4d.shape
    mean_4d         = np.arange(nt*nz*nx*nx).reshape(nt,nz,nx,nx)*np.NaN
    std_4d          = np.arange(nt*nz*nx*nx).reshape(nt,nz,nx,nx)*np.NaN
    skew_4d         = np.arange(nt*nz*nx*nx).reshape(nt,nz,nx,nx)*np.NaN
    for time in np.arange(0,nt,1):
        data3d              = data4d[time,:,:,:]
        moments_3d          = loop_over_3rd_dim(data3d,nx,dx)
        mean_4d[time,:,:,:] = moments_3d['mean_3d']
        std_4d[time,:,:,:]  = moments_3d['std_3d']
        skew_4d[time,:,:,:] = moments_3d['skew_3d']
    return {'mean_4d':mean_4d,'std_4d':std_4d,'skew_4d':skew_4d};

def loop_over_3rd_dim(data3d,nx,dx):
    na,tmp1,tmp2  = data3d.shape
    # na is the length of the arbitrary 3rd dimension counting [3,2,1] this could be:
    #    height as in [k,j,i]
    # or time   as in [t,j,i]
    mean_3d = np.arange(na*nx*nx).reshape(na,nx,nx)*np.NaN
    std_3d  = np.arange(na*nx*nx).reshape(na,nx,nx)*np.NaN
    skew_3d = np.arange(na*nx*nx).reshape(na,nx,nx)*np.NaN
    for a in np.arange(0,na,1):
        data2d         = data3d[a,:,:]
        moments_2d     = cut_up_2d_array_into_regions_and_find_moments(data2d,nx,dx)
        mean_3d[a,:,:] = moments_2d['mean_2d']
        std_3d[a,:,:]  = moments_2d['std_2d']
        skew_3d[a,:,:] = moments_2d['skew_2d']
    return {'mean_3d':mean_3d,'std_3d':std_3d,'skew_3d':skew_3d};

def cut_up_2d_array_into_regions_and_find_moments(data2d,nx,dx):
    # Takes a 2d array of data and chops it up in GCM gridbox sized regions and calculated moments for each GCM region.
    mean_2d = np.arange(nx*nx).reshape(nx,nx)*np.NaN
    std_2d  = np.arange(nx*nx).reshape(nx,nx)*np.NaN
    skew_2d = np.arange(nx*nx).reshape(nx,nx)*np.NaN
    for j in np.arange(0,nx,1):
        for i in np.arange(0,nx,1):
            #    for j in np.arange(0,1,1):
            #        for i in np.arange(0,1,1):
            startx       = i*dx
            endx         = (i+1)*dx
            starty       = j*dx
            endy         = (j+1)*dx
            subdata2d    = data2d[starty:endy,startx:endx]
            moments_1d   = calc_mean_std_skew(subdata2d)
            mean_2d[j,i] = moments_1d['mean_1d']
            std_2d[j,i]  = moments_1d['std_1d']
            skew_2d[j,i] = moments_1d['skew_1d']
    return {'mean_2d':mean_2d,'std_2d':std_2d,'skew_2d':skew_2d};

def calc_mean_std_skew(data):
    # Assumes data coming in is 2d (x and y only, z and time dimension have been stripped out earlier!)
    mean_1d = np.mean(data.data)
    std_1d  = np.std(data.data)
    if std_1d > 0.0:
        tmp     = ((data.data-mean_1d)/std_1d)**3.0
        skew_1d = np.mean(tmp)
    else:
        skew_1d = 0.0
    return {'mean_1d':mean_1d,'std_1d':std_1d,'skew_1d':skew_1d};

def calc_bcu(date,roseid,name_str,analysis_time,region_str,tmppath,time_index,b_thresh,c_thresh,w_thresh,dummy_stash):
    # Calculated the mean value of a surface 2d field, only sampling over the columns that have a
    # buoyant cloudy updraught at some point in their profile.
    vt=0
    #  0,  4 THETA
    #  0, 10 QV
    # 16,  4 T ON THETA
    #  0,150 w
    #  0,266 BCF
    #
    dx=32
    nx=np.int(448/dx)
    #
    nz=70
    #
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region_str)
    result = make_stash_string(4,203)
    dummy  = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    print('dummy',dummy)
    # Hacky solution! Make use of cube info from already-loaded field to define a smaller cube.
    bcu_ppn=dummy[time_index,0:nx,0:nx]
    print('bcu_ppn',bcu_ppn)
    bcu_ppn.data[:]=np.NaN
    #
    # Read in THETA
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region_str)
    result = make_stash_string(0,4)
    theta  = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    theta=theta[time_index,:,32:480,32:480]
    # Read in QV
    filename=generate_ml_filename_in(date,vt,'.pp','b',name_str,analysis_time,region_str)
    result = make_stash_string(0,10)
    qv     = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qv=qv[time_index,:,32:480,32:480]
    # Read in T
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region_str)
    result = make_stash_string(16,4)
    temp   = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    temp=temp[time_index,:,32:480,32:480]
    # Read in Bulk Cloud Fraction
    filename=generate_ml_filename_in(date,vt,'.pp','d',name_str,analysis_time,region_str)
    result = make_stash_string(0,266)
    bcf    = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    bcf=bcf[time_index,:,32:480,32:480]
    # Read in w wind
    filename=generate_ml_filename_in(date,vt,'.pp','a',name_str,analysis_time,region_str)
    result = make_stash_string(0,150)
    w_wind = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    w_wind=w_wind[time_index,:,32:480,32:480]
    # Read in LSRR
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region_str)
    result = make_stash_string(4,203)
    lsrr = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    lsrr=lsrr[time_index,32:480,32:480]
    # Read in LSSR
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region_str)
    result = make_stash_string(4,204)
    lssr = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    lssr=lssr[time_index,32:480,32:480]
    #
    # Loop over the 14x14 array that this LAM's domain is being chopped up into.
    for j in np.arange(0,nx,1):
        for i in np.arange(0,nx,1):
            startx = i*dx
            endx   = (i+1)*dx
            starty = j*dx
            endy   = (j+1)*dx
            mask   = temp.data[0,starty:endy,startx:endx]*0.0
            for k in np.arange(0,nz,1):
                # Calculate theta v prime
                tempv        = temp.data[k,starty:endy,startx:endx]+(0.6*qv.data[k,starty:endy,startx:endx])
                thetav       = tempv * theta.data[k,starty:endy,startx:endx] / temp.data[k,starty:endy,startx:endx]
                mean_thetav  = np.mean(np.reshape(thetav,(1,dx*dx)))
                thetav_prime = thetav - mean_thetav
                w_here       = w_wind.data[k,starty:endy,startx:endx]
                bcf_here     = bcf.data[k,starty:endy,startx:endx]
                mask         = np.where((thetav_prime > b_thresh) & (bcf_here > c_thresh) & (w_here > w_thresh),mask+1,mask)
            #end k
            #
            # Only need to have one BCU event in the column for whole column to be considered a BCU column.
            mask                   = np.where((mask>1.0),1.0,mask)
            # Combine the large-scale rain and snowfall rates
            ppn                    = lsrr[starty:endy,startx:endx]+lssr[starty:endy,startx:endx]
            # Apply the mask, so only BCU precipitation is retained.
            masked_ppn             = mask*ppn.data
            # Mean over the GCM gridbox
            bcu_ppn.data[j,i] = np.mean(masked_ppn)
        # end i
    # end j
    ##end time
    #
    filenameout=generate_mlarray_filename_out(date,time_index,'.nc',name_str,analysis_time,region_str,dx,nx,'PPN','BCU_'+str(b_thresh)+'_'+str(c_thresh)+'_'+str(w_thresh))
    fileout=tmppath+'netcdf/'+filenameout
    print(fileout)
    iris.save(bcu_ppn, fileout)
    #
    # Write out a file (containing number 1.0) to say that this field has been processed yet).
    filenameout=generate_flag_name(date,time_index,'.dt',name_str,analysis_time,region_str,dx,nx,dummy_stash,'FLG')
    fileout=tmppath+'flags/'+filenameout
    np.savetxt(fileout, [1.0], fmt='%4.2f')
    #
    outcome=1
    return outcome;

#def loop_over_corrcoef(date,roseid,name_str,analysis_time,region_str,tmppath,single_time_stamp,dummy_stash):
#    outcome = calc_subdomain_corrcoef(date,roseid,name_str,analysis_time,region_str,tmppath,single_time_stamp,'theta_li','qt',dummy_stash)
#    outcome = calc_subdomain_corrcoef(date,roseid,name_str,analysis_time,region_str,tmppath,single_time_stamp,'theta_li','w',dummy_stash)
#    outcome = calc_subdomain_corrcoef(date,roseid,name_str,analysis_time,region_str,tmppath,single_time_stamp,'w','qv',dummy_stash)
#    return outcome;

def calc_subdomain_corrcoef(date,roseid,name_str,analysis_time,region_str,tmppath,time_index,name1,name2,dummy_stash):
    vt = 0
    dx = 32
    nx = np.int(448/dx)
    nz = 70 
    #
    # Hacky solution! Load a cube to get the right info to define a smaller cube so that I can then use iris save at the end.
    filename     = generate_ml_filename_in(date,vt,'.pp','a',name_str,analysis_time,region_str)
    result       = make_stash_string(0,150)
    corr         = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    corr         = corr[time_index,:,0:nx,0:nx]
    corr.data[:] = np.NaN
    #
    lv       = 2.501e6
    lf       = 2.834e6
    cp       = 1005.0
    lvovercp = lv / cp
    lfovercp = lf / cp
    #
    if name1=='theta_li':
        filename = generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region_str)
        result   = make_stash_string(0,4)
        theta    = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        theta    = theta[time_index,:,:,:]
        result   = make_stash_string(16,4)
        temp     = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        temp     = temp[time_index,:,:,:]
        #
        filename = generate_ml_filename_in(date,vt,'.pp','b',name_str,analysis_time,region_str)
        result   = make_stash_string(0,254)
        qcl      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcl      = qcl[time_index,:,:,:]
        result   = make_stash_string(0,12)
        qcf      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcf      = qcf[time_index,:,:,:]
        result   = make_stash_string(0,272)
        qrain    = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qrain    = qrain[time_index,:,:,:]
        result   = make_stash_string(0,273)
        qgraup   = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qgraup   = qgraup[time_index,:,:,:]
        #
        exner    = theta / temp
        # Do some tedious unit resetting steps because Iris thinks it is being clever stopping you adding things with different units.
        temp.units = '1'
        # Find liquid-ice potential temperature
        temp_li      = temp - (lvovercp*(qcl+qrain)) - (lfovercp*(qcf+qgraup))
        field1       = temp_li * exner
        field1.units = 'K'
    elif name1=='qt':
        filename = generate_ml_filename_in(date,vt,'.pp','b',name_str,analysis_time,region_str)
        result   = make_stash_string(0,10)
        qv       = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qv       = qv[time_index,:,:,:]
        result   = make_stash_string(0,254)
        qcl      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcl      = qcl[time_index,:,:,:]
        result   = make_stash_string(0,12)
        qcf      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcf      = qcf[time_index,:,:,:]
        result   = make_stash_string(0,272)
        qrain    = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qrain    = qrain[time_index,:,:,:]
        result   = make_stash_string(0,273)
        qgraup   = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qgraup   = qgraup[time_index,:,:,:]
        field1   = qv+qcl+qcf+qrain+qgraup
    elif name1=='w_wind':
        filename = generate_ml_filename_in(date,vt,'.pp','a',name_str,analysis_time,region_str)
        result   = make_stash_string(0,150)
        w_wind   = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        field1   = w_wind[time_index,:,:,:]
    # end if
    #
    if name2=='theta_li':
        filename = generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region_str)
        result   = make_stash_string(0,4)
        theta    = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        theta    = theta[time_index,:,:,:]
        result   = make_stash_string(16,4)
        temp     = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        temp     = temp[time_index,:,:,:]
        #
        filename = generate_ml_filename_in(date,vt,'.pp','b',name_str,analysis_time,region_str)
        result   = make_stash_string(0,254)
        qcl      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcl      = qcl[time_index,:,:,:]
        result   = make_stash_string(0,12)
        qcf      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcf      = qcf[time_index,:,:,:]
        result   = make_stash_string(0,272)
        qrain    = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qrain    = qrain[time_index,:,:,:]
        result   = make_stash_string(0,273)
        qgraup   = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qgraup   = qgraup[time_index,:,:,:]
        #
        exner    = theta / temp
        # Do some tedious unit resetting steps because Iris thinks it is being clever stopping you adding things with different units.
        temp.units = '1'
        # Find liquid-ice potential temperature
        temp_li      = temp - (lvovercp*(qcl+qrain)) - (lfovercp*(qcf+qgraup))
        field2       = temp_li * exner
        field2.units = 'K'
    elif name2=='qt':
        filename = generate_ml_filename_in(date,vt,'.pp','b',name_str,analysis_time,region_str)
        result   = make_stash_string(0,10)
        qv       = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qv       = qv[time_index,:,:,:]
        result   = make_stash_string(0,254)
        qcl      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcl      = qcl[time_index,:,:,:]
        result   = make_stash_string(0,12)
        qcf      = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qcf      = qcf[time_index,:,:,:]
        result   = make_stash_string(0,272)
        qrain    = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qrain    = qrain[time_index,:,:,:]
        result   = make_stash_string(0,273)
        qgraup   = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        qgraup   = qgraup[time_index,:,:,:]
        field2   = qv+qcl+qcf+qrain+qgraup
    elif name2=='w_wind':
        filename = generate_ml_filename_in(date,vt,'.pp','a',name_str,analysis_time,region_str)
        result   = make_stash_string(0,150)
        w_wind   = iris.load_cube(tmppath+'pp/'+filename,iris.AttributeConstraint(STASH=result['stashstr_iris']))
        field2   = w_wind[time_index,:,:,:]
    # end if
    #
    field1=field1[:,32:480,32:480]
    field2=field2[:,32:480,32:480]
    #
    # Loop over the 14x14 array that this LAM's domain is being chopped up into.
    for j in np.arange(0,nx,1):
        for i in np.arange(0,nx,1):
            startx = i*dx
            endx   = (i+1)*dx
            starty = j*dx
            endy   = (j+1)*dx
            for k in np.arange(0,nz,1):
                field1_here = np.reshape(field1.data[k,starty:endy,startx:endx],(1,dx*dx))
                field2_here = np.reshape(field2.data[k,starty:endy,startx:endx],(1,dx*dx))
                #
                # Calculate R, but we are pirates, so it's arh, me hearties!
                arh=np.corrcoef(field1_here,field2_here)
                if (np.isnan(arh[0,1])) and ( (np.std(field1_here)==0.0) or (np.std(field2_here)==0.0) ):
                    # corr=cov(X,Y)/(sigma_X*sigma_Y) so if either of the fields are constant, 
                    # then its standard deviation will be zero and hence corr will be NaN. 
                    # But in this case I'd really like to store a value of zero rather than nan.
                    arh[0,1]=0.0
                # end if
                corr.data[k,j,i] = arh[0,1]
            # end k
        # end i
    # end j
    #
    filenameout=generate_mlarray_filename_out(date,time_index,'.nc',name_str,analysis_time,region_str,dx,nx,name1+'-'+name2,'COR')
    fileout=tmppath+'netcdf/'+filenameout
    iris.save(corr, fileout)
    #
    # Write out a file (containing number 1.0) to say that this field has now been processed).
    filenameout=generate_flag_name(date,time_index,'.dt',name_str,analysis_time,region_str,dx,nx,dummy_stash,'FLG')
    fileout=tmppath+'flags/'+filenameout
    print(fileout)
    np.savetxt(fileout, [1.0], fmt='%4.2f')
    #
    outcome=1
    return outcome;

def make_stash_string(stashsec,stashcode):
    #
    stashsecstr=str(stashsec)
    if stashsec<10:
        stashsecstr='0'+stashsecstr
    # endif
    #
    stashcodestr=str(stashcode)
    if stashcode<100:
        stashcodestr='0'+stashcodestr
    # endif
    if stashcode<10:
        stashcodestr='0'+stashcodestr
    # endif
    stashstr_iris='m01s'+stashsecstr+'i'+stashcodestr
    stashstr_fout=stashsecstr+stashcodestr
    return {'stashstr_iris':stashstr_iris, 'stashstr_fout':stashstr_fout};

def generate_mlarray_filename_out(date,vt,ext,name_str,analysis_time,region_str,size,nx,stashnumber,operation):
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+'/'+region_str+'/'+date.strftime("%Y%m%d")+analysis_time+'_'+region_str+'_km3p3_'+name_str+'_'+str(size)+'x'+str(size)+'sampling_hence_'+str(nx)+'x'+str(nx)+'_time'+vtstr+'_'+operation+'_'+stashnumber+ext
    #
    return filename;

def generate_flag_name(date,vt,ext,name_str,analysis_time,region_str,size,nx,stashnumber,operation):
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+'/'+date.strftime("%Y%m%d")+analysis_time+'_'+region_str+'_km3p3_'+name_str+'_'+str(size)+'x'+str(size)+'sampling_hence_'+str(nx)+'x'+str(nx)+'_time'+vtstr+'_'+operation+'_stash'+stashnumber+ext
    #
    return filename;

def generate_ml_filename_in(date,vt,ext,stream,name_str,analysis_time,region_str):
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+'/'+date.strftime("%Y%m%d")+analysis_time+'_'+region_str+'_km3p3_'+name_str+'_p'+stream+vtstr+ext
    #
    return filename;

######################################################################################

def step(start,inc,end):
    # A nice function to get around the stupid way that np.linspace works
    num=round((end-start)/inc)+1.0
    number_array=np.linspace(start,end,num)
    return number_array;

def generate_ml_filename_in_OLD(date,vt,ext,stream,name_str,analysis_time,region):
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+'/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_pver'+stream+vtstr+ext
    #print(filename)
    #
    return filename;

def generate_ml_filename_out(date,vt,ext,name_str,analysis_time,region,size,subregion,stashnumber):
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+str(size)+'x'+str(size)+'_subdomain'+str(subregion)+'_'+vtstr+'_'+stashnumber+ext
    #print(filename)
    #
    return filename;

def get_lat_lon_hecto_nest(filename):
    # Read in the coords written out by maps_nesting_domains_aqua_only.m
    reg_lat=np.empty(0, int)
    reg_lon=np.empty(0, int)
    with open('/home/h01/frme/ml_lams_latlon_aqua_only.dat', 'r') as filestream:
        print(filestream)
        for line in filestream:
            currentline = line.split(",")
            reg_lat=np.append(reg_lat,int(currentline[1]))
            reg_lon=np.append(reg_lon,int(currentline[2]))
    #
    return {'reg_lat':reg_lat, 'reg_lon':reg_lon};

def name_of_hecto_lam(reg_lat,reg_lon):
    # Given lat and lon, generate a string that describes the location e.g. 50N150W or 20S140E
    if reg_lat>=0:
        lat_letter='N'
    else:
        lat_letter='S'
    if reg_lon<0:
        lon_letter='W'
    else:
        lon_letter='E'
    region=str(np.abs(reg_lat))+lat_letter+str(np.abs(reg_lon))+lon_letter
    return region;

def process_ml_lam_file(date,vt,roseid,name_str,analysis_time,stream,region,stash_sec,stash_code,ndims):
    # Version containing one timestamp per file (apart from at 02Z)
    # Currently hard-coded for user "frme"
    tmppath='/scratch/frme/ML/'+roseid+'/'
    filename=generate_ml_filename_in(date,vt,'.pp',stream,name_str,analysis_time,region)
    filein=tmppath+'pp/'+filename
    # Read in data
    result = make_stash_string(stash_sec,stash_code)
    fieldin = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #print(fieldin)
    # Input array is 360 x 360
    # Take the central 240 x 240
    # N.B. python counts from 0
    # Check on whether the fields being processes is 2d or 3d (actually 3d or 4d with time dimension)
    if ndims==2:
        #data=fieldin[:,60:300,60:300] #This was used when there were 6 10-minute data per hourly file
        data=fieldin[60:300,60:300] #this can be used when there is only one time point hence one less dimension
    if ndims==3:
        if vt==0:
            # For prognostic fields (theta, qv etc) they will get output at T+0 at the start of the run.
            # Hence the 000 file will contain data for 00 as well as 02 but only process the 02.
            # Note that file 02 contains data for 04 (!)
            fieldin=fieldin[1,:,:,:]
            #            print(data)
        #data=fieldin[:,:,60:300,60:300] #This was used when there were 6 10-minute data per hourly file
        data=fieldin[:,60:300,60:300] #this can be used when there is only one time point hence one less dimension
    # Now call the function that extracts and averages the LAM data onto GCM grid-boxes
    outcome=mean_subdomains(data,date,vt,name_str,analysis_time,region,result['stashstr_fout'],ndims,tmppath+'netcdf/')
    return outcome;

def mean_subdomains_OLD(data,date,vt,name_str,analysis_time,region,output_label,ndims,tmppath):
    # Function that extracts the LAM data over an area corresponding to a GCM grid-boxes
    # and then averages it and writes it out.
    #
    # The data is of size (time,lat,lon), i.e. (time,y,x) so indexing is (:,j,i)
    #
    # This next array could be populated further to aggregate over a different number of gridpoints.
    size_of_subdomains=[32]
    # e.g. size_of_subdomains=[30,60,120]
    for s in np.arange(0,len(size_of_subdomains),1):
        dx=size_of_subdomains[s]
#        subregion=0
        nx=np.int(448/dx)
        # Hacky solution! Make use of cube info for data to define a smaller cube 
        # (with some of the right metadata properties to fill with the spatially averaged data) 
        # hence allowing it to be written out using iris save.
        # Note that the lat/lon in this cube will be wrong as it just uses the SW corner of the bigger original array, 
        # rather than redefining the lat and lon, but I do not care. But I can always work it out correctly if needed.
        if ndims==2:
            coarse_grained_grid=data[0:nx,0:nx]
        if ndims==3:
            coarse_grained_grid=data[:,0:nx,0:nx]
        coarse_grained_grid.data[:]=np.NaN
        for j in np.arange(0,nx,1):
            for i in np.arange(0,nx,1):
                startx=i*dx
                endx=(i+1)*dx
                starty=j*dx
                endy=(j+1)*dx
                # print('Subregion=',subregion,' extracting from:',startx,endx,starty,endy)
                if ndims==2:
                    subdata=data[starty:endy,startx:endx]
                    if region=='0N0E':
                        horiz_meaned_data = subdata.collapsed(['longitude','latitude'], iris.analysis.MEAN)
                    else:
                        horiz_meaned_data = subdata.collapsed(['grid_longitude','grid_latitude'], iris.analysis.MEAN)
                    coarse_grained_grid.data[j,i]=horiz_meaned_data.data
                if ndims==3:
                    subdata=data[:,starty:endy,startx:endx]
                    if region=='0N0E':
                        horiz_meaned_data = subdata.collapsed(['longitude','latitude'], iris.analysis.MEAN)
                    else:
                        horiz_meaned_data = subdata.collapsed(['grid_longitude','grid_latitude'], iris.analysis.MEAN)

                    #print(horiz_meaned_data)
                    #print(horiz_meaned_data.data)
                    coarse_grained_grid.data[:,j,i]=horiz_meaned_data.data
            #    subregion=subregion+1
            # end i
        # end j
        filenameout=generate_mlarray_filename_out(date,vt,'.nc',name_str,analysis_time,region,size_of_subdomains[s],nx,output_label)
        fileout=tmppath+filenameout
        iris.save(coarse_grained_grid, fileout)

    outcome=1
    return outcome;

def extract_fields_for_advective_tendencies(date,vt,roseid,name_str,analysis_time,region):
    # Currently hard-coded for user "frme"
    tmppath='/scratch/frme/ML/'+roseid+'/'
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region)
    filein=tmppath+'pp/'+filename
    #
    # Read in moisture data
    result = make_stash_string(0,10)
    qv = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,254)
    qcl = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,12)
    qcf = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,272)
    qrain = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,273)
    qgraup = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    # Add all the water variables together
    qtotal = qv+qcl+qcf+qrain+qgraup
    #
    # Read in dry potential temperature data
    result = make_stash_string(0,4)
    theta_dry = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    # Read in pressure on theta levels
    filename=generate_ml_filename_in(date,vt,'.pp','f',name_str,analysis_time,region)
    filein_p=tmppath+'pp/'+filename
    result = make_stash_string(0,408)
    p_theta_levels = iris.load_cube(filein_p,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    # These are the calculations we need to do.
    # Tl    = T - (L/cp)*qcl
    # T     = theta * exner
    # exner = (p/pref) ** kay
    kay=0.286
    # In its wisdom iris is stopping me raising a variable to a non integer power because it doesn't know what to do with the units!
    exner=p_theta_levels
    exner.data=(p_theta_levels.data/1.0e5)**kay
    lv=2.501e6
    lf=2.834e6
    cp=1005.0
    lvovercp=lv/cp
    lfovercp=lf/cp
    liq=qcl+qrain
    ice=qcf+qgraup
    #
    # File for time 00Z actually contains data for 00Z and 02Z, but we only need 02Z, so ignore 0th element.
    # Would be more efficient to do this for each moist variable and exner before operating on them, but this is neater.
    if vt==0:
        qtotal         = qtotal[1,:,:,:]
        liq            = liq[1,:,:,:]
        ice            = ice[1,:,:,:]
        theta_dry      = theta_dry[1,:,:,:]
        exner          = exner[1,:,:,:]
    # Do some tedious unit resetting steps because Iris thinks it is being clever stopping you adding things with different units.
    theta_dry.units='1'
    liq.units='1'
    ice.units='1'
    exner.units='1'
    # Populate all meta data from theta_dry into theta (lat/lon coords etc).
    theta=theta_dry
    # Calculate a liq/ice static temperature
    theta=theta_dry-(lvovercp*liq/exner)-(lfovercp*ice/exner)
    theta.units='K'
    # Ensure sensible names
    theta.long_name='Liquid ice static potential temperature'
    theta.var_name='thetali'
    qtotal.long_name='Total humidity'
    qtotal.var_name='qtotal'
    #
    # Read in wind data
    # NB u wind is staggered half a grid-box to the west  and half a layer down.
    # NB v wind is staggered half a grid-box to the south and half a layer down.
    # NB w wind is on same grid and theta and q
    filename=generate_ml_filename_in(date,vt,'.pp','e',name_str,analysis_time,region)
    filein_wind=tmppath+'pp/'+filename
    result = make_stash_string(0,2)
    u_wind = iris.load_cube(filein_wind,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,3)
    v_wind = iris.load_cube(filein_wind,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,150)
    w_wind = iris.load_cube(filein_wind,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    if vt==0:
        # As before ignore 0th data for 00 in file containing both 00Z and 02Z data.
        u_wind=u_wind[1,:,:,:]
        v_wind=v_wind[1,:,:,:]
        w_wind=w_wind[1,:,:,:]
    #
    flux_qtotal=u_dot_grad_field(u_wind,v_wind,w_wind,qtotal,'qtotal')
    flux_theta=u_dot_grad_field(u_wind,v_wind,w_wind,theta,'theta')
    #
    flux_qtotal.long_name='Flux of qtotal from (u,v,w).grad(qttoal)'
    flux_qtotal.var_name='flux_qtotal'
    flux_theta.long_name='Flux of thetali from (u,v,w).grad(thetali)'
    flux_theta.var_name='flux_thetali'
    #
    write_out_intermediate_data=0
    if write_out_intermediate_data==1:
        # Write everything out so can potentially check things offline.
        #
        # vt=validity time
        if vt < 10:
            vtstr='00'+str(vt) 
        else:
            vtstr='0'+str(vt)
        # endif
        fileout=tmppath+'/netcdf/intermediate_advect_data/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_qtotal.nc'
        iris.save(qtotal, fileout)
        fileout=tmppath+'/netcdf/intermediate_advect_data/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_theta.nc'
        iris.save(theta, fileout)
        fileout=tmppath+'/netcdf/intermediate_advect_data/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_uwind.nc'
        iris.save(u_wind, fileout)
        fileout=tmppath+'/netcdf/intermediate_advect_data/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_vwind.nc'
        iris.save(v_wind, fileout)
        fileout=tmppath+'/netcdf/intermediate_advect_data/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_wwind.nc'
        iris.save(w_wind, fileout)
        fileout=tmppath+'/netcdf/intermediate_advect_data/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_python_flux_qtotal.nc'
        iris.save(flux_qtotal, fileout)
        fileout=tmppath+'/netcdf/intermediate_advect_data/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_python_flux_theta.nc'
        iris.save(flux_theta, fileout)
    # endif
    #
    # Make up some stash numbers so file has 5 digit ref but these are not actual UM stash codes
    #
    # 88181 thetali
    # 88182 total q
    outcome=mean_subdomains(theta[:,60:300,60:300],date,vt,name_str,analysis_time,region,'88181',3,tmppath+'netcdf/')
    outcome=mean_subdomains(qtotal[:,60:300,60:300],date,vt,name_str,analysis_time,region,'88182',3,tmppath+'netcdf/')
    #
    # 99181 thetali increment from advection
    # 99182 total q increment from advection
    outcome=mean_subdomains(flux_theta[:,60:300,60:300],date,vt,name_str,analysis_time,region,'99181',3,tmppath+'netcdf/')
    outcome=mean_subdomains(flux_qtotal[:,60:300,60:300],date,vt,name_str,analysis_time,region,'99182',3,tmppath+'netcdf/')
    #
    # Also read in, average, and write out: bulk cloud fraction
#    result   = make_stash_string(0,266)
#    bcf      = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
#    if vt==0:
#        bcf            = bcf[1,:,:,:]
#    outcome=mean_subdomains(bcf[:,60:300,60:300],date,vt,name_str,analysis_time,region,'00266',3,tmppath+'netcdf/')
    #
    return outcome;

def generate_filename_in(date,vt,ext,stream,name_str,analysis_time,region,res_name):
    if vt < 10:
        vtstr='000'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    #print(date.strftime("%Y%m%d"))
    #print(analysis_time)
    #print(region)
    #print(res_name)
    #print(name_str)
    #print(stream)
    #print(vtstr)
    #print(ext)
    filename=date.strftime("%Y%m%d")+analysis_time+'_'+region+'_'+res_name+'_'+name_str+'_p'+stream+vtstr+ext
    print(filename)
    #
    return filename;

def retrieve_a_file(date,vt,roseid,name_str,analysis_time,stream,region,res_name,flag):
    # If flag == 1, it deletes the file and then retrieves a clean copy
    # If flag == 0, it just deletes the file.
    #
    # Currently hard-coded for user "frme"
    tmppath='/scratch/frme/ML/'+roseid+'/pp/'+date.strftime("%Y%m%d")+'/'
    # Ensure the directory exists
    subprocess.call(["mkdir", tmppath])
    filename=generate_filename_in(date,vt,'.pp',stream,name_str,analysis_time,region,res_name)
    # moo retrieval will not overwrite an existing file.
    # So remove any existing occurrence of the file we are trying to retrieve
    stalefile=tmppath+filename
    subprocess.call(["rm", stalefile])
    if flag==1:
        moopath='moose:/devfc/'+roseid+'/field.pp/'
        fullname=moopath+filename
        print('fullname',fullname)
        subprocess.call(["moo","get", fullname, tmppath])
    # endif
    outcome=1
    return outcome;

def retrieve_all_files_for_a_day(roseid,date,region_str,flag):
    # If flag == 1, it deletes the files and then retrieves a clean copy
    # If flag == 0, it just deletes the files.
    #
    # Currently hard-coded for user "frme"
    tmppath='/scratch/frme/ML/'+roseid+'/pp/'+date.strftime("%Y%m%d")+'/'
    # Ensure the directory exists
    subprocess.call(["mkdir", tmppath])
    # moo retrieval will not overwrite an existing file.
    # So remove any existing occurrence of the files we are about to retrieve.
    stalefile=tmppath+'*'+region_str
    subprocess.call(["rm", stalefile])
    if flag==1:
        filename=date.strftime("%Y%m%d")+'*'+region_str
        moopath='moose:/devfc/'+roseid+'/field.pp/'
        fullname=moopath+filename
        subprocess.call(["moo","get", fullname, tmppath])
    # endif
    outcome=1
    return outcome;

def u_dot_grad_field(u_wind,v_wind,w_wind,field,field_str):
    # Use L70 with an 80 km top (needed for calculating ddz).
    z_top_of_model =  80000.0
    eta_theta=np.array([ .0002500,  .0006667,  .0012500,  .0020000,  .0029167,  .0040000,  
                         .0052500,  .0066667,  .0082500,  .0100000,  .0119167,  .0140000,  
                         .0162500,  .0186667,  .0212500,  .0240000,  .0269167,  .0300000,  
                         .0332500,  .0366667,  .0402500,  .0440000,  .0479167,  .0520000,  
                         .0562500,  .0606667,  .0652500,  .0700000,  .0749167,  .0800000,  
                         .0852500,  .0906668,  .0962505,  .1020017,  .1079213,  .1140113,  
                         .1202745,  .1267154,  .1333406,  .1401592,  .1471838,  .1544313,  
                         .1619238,  .1696895,  .1777643,  .1861929,  .1950307,  .2043451,  
                         .2142178,  .2247466,  .2360480,  .2482597,  .2615432,  .2760868,  
                         .2921094,  .3098631,  .3296378,  .3517651,  .3766222,  .4046373,  
                         .4362943,  .4721379,  .5127798,  .5589045,  .6112759,  .6707432,  
                         .7382500,  .8148403,  .9016668, 1.0000000])
    height_theta_levels=eta_theta*z_top_of_model
    # Everything written assuming UM's staggered grid.
    # ------------------------------------------------
    # NB (x,y) space is:
    #                     theta(i,j+1)
    #                      
    #                     v(i,j+1)
    #                      
    # theta(i-1,j) u(i,j) theta(i,j)   u(i+1,j) theta(i+1,j)
    #                      
    #                     v(i,j)
    #                       
    #                     theta(i,j-1)
    #
    # ------------------------------------------------
    # NB (x,z) space is:
    #
    # 80000 m
    #                     76066m
    # ...                 ...           
    # 53m theta,w (i,k+1)
    #                     36m u,v (i,k+1)
    # 20m theta,w (i,k)
    #                     10m u,v (i,k)
    # ----------Ground level--------------------------
    #
    # NB dimensions are (time, height, latitude, longitude)
    # i.e.              (t,    z,      y,        x        )
    #
    # i component
    dqdx_lhs=(field.data[:,60:300,60:300]-field.data[:,60:300,59:299])/1500.0
    dqdx_rhs=(field.data[:,60:300,61:301]-field.data[:,60:300,60:300])/1500.0
    #
    nz=field.shape[0]
    nx=field.shape[1]
    ny=field.shape[2]
    #
    u_lhs_half_lev_below=u_wind.data[:,60:300,60:300]
    u_rhs_half_lev_below=u_wind.data[:,60:300,61:301]
    #
    # Do this to get array of correct size
    u_lhs_half_lev_above=u_lhs_half_lev_below*0.0
    u_rhs_half_lev_above=u_rhs_half_lev_below*0.0
    #
    for k in np.arange(0,nz-1,1):
        # Copy information from one layer higher up
        u_lhs_half_lev_above[k,:,:]=u_lhs_half_lev_below[k+1,:,:]
        u_rhs_half_lev_above[k,:,:]=u_rhs_half_lev_below[k+1,:,:]
    # For top-most level (i.e. 70th level (69th in python-speak) set it to same as layer below it
    u_lhs_half_lev_above[nz-1,:,:]=u_lhs_half_lev_above[nz-2,:,:]
    u_rhs_half_lev_above[nz-1,:,:]=u_rhs_half_lev_above[nz-2,:,:]
    #
    # Linear-average (no pressure or density weighting) of values on half level above 
    # and below to get horizontal wind on this theta level
    u_theta_lev_lhs=(u_lhs_half_lev_above+u_lhs_half_lev_below)*0.5
    u_theta_lev_rhs=(u_rhs_half_lev_above+u_rhs_half_lev_below)*0.5
    # Calculate flux coming in and going out
    u_dqdx_flux_in =u_theta_lev_lhs*dqdx_lhs
    u_dqdx_flux_out=u_theta_lev_rhs*dqdx_rhs
    #
    # j component
    dqdy_south=(field.data[:,60:300,60:300]-field.data[:,59:299,60:300])/1500.0
    dqdy_north=(field.data[:,61:301,60:300]-field.data[:,60:300,60:300])/1500.0
    #
    v_south_half_lev_below=v_wind.data[:,60:300,60:300]
    v_north_half_lev_below=v_wind.data[:,61:301,60:300]
    #
    # Do this to get array of correct size
    v_south_half_lev_above=v_south_half_lev_below*0.0
    v_north_half_lev_above=v_north_half_lev_below*0.0
    #
    for k in np.arange(0,nz-1,1):
        # Copy information from one layer higher up
        v_south_half_lev_above[k,:,:]=v_south_half_lev_below[k+1,:,:]
        v_north_half_lev_above[k,:,:]=v_north_half_lev_below[k+1,:,:]
    # For top-most level (i.e. 70th level (69th in python-speak) set it to same as layer below it
    v_south_half_lev_above[nz-1,:,:]=v_south_half_lev_above[nz-2,:,:]
    v_north_half_lev_above[nz-1,:,:]=v_north_half_lev_above[nz-2,:,:]
    #
    # Linear-average (no pressure or density wieghting) of values on half level above 
    # and below to get horizontal wind on this theta level
    v_theta_lev_south=(v_south_half_lev_above+v_south_half_lev_below)*0.5
    v_theta_lev_north=(v_north_half_lev_above+v_north_half_lev_below)*0.5
    # Calculate flux coming in and going out
    v_dqdy_flux_in =v_theta_lev_south*dqdy_south
    v_dqdy_flux_out=v_theta_lev_north*dqdy_north
    #
    # k component
    # NB w is held on same levels as theta and q
    w_half_lev_below=w_wind.data[:,60:300,60:300]*0.0
    w_half_lev_above=w_wind.data[:,60:300,60:300]*0.0
    for k in np.arange(1,nz-1,1):
        w_half_lev_below[k,:,:]=(w_wind.data[k,60:300,60:300]+w_wind.data[k-1,60:300,60:300])*0.5
    w_half_lev_below[0,:,:]=0.0
    #
    for k in np.arange(0,nz-2,1):
        w_half_lev_above[k,:,:]=(w_wind.data[k,60:300,60:300]+w_wind.data[k+1,60:300,60:300])*0.5
    w_half_lev_above[nz-1,:,:]=0.0
    #
    dqdz_half_lev_below=field.data[:,60:300,60:300]*0.0
    for k in np.arange(1,nz-1,1):
        dqdz_half_lev_below[k,:,:]=(field.data[k,60:300,60:300]-field.data[k-1,60:300,60:300])/(height_theta_levels[k]-height_theta_levels[k-1])
    dqdz_half_lev_below[0,:,:]=0.0
    #
    dqdz_half_lev_above=field.data[:,60:300,60:300]*0.0
    for k in np.arange(0,nz-2,1):
        dqdz_half_lev_above[k,:,:]=(field.data[k+1,60:300,60:300]-field.data[k,60:300,60:300])/(height_theta_levels[k+1]-height_theta_levels[k])
    dqdz_half_lev_above[nz-1,:,:]=0.0
    #
    # Calculate flux coming in and going out
    w_dqdz_flux_in =w_half_lev_below*dqdz_half_lev_below
    w_dqdz_flux_out=w_half_lev_above*dqdz_half_lev_above
    #
    # Combine i, j, k components
    #
    # Seem to get memory issues is try to do this in one line.
    # net_flux=(u_dqdx_flux_in-u_dqdx_flux_out)+(v_dqdy_flux_in-v_dqdy_flux_out)+(w_dqdz_flux_in-w_dqdz_flux_out)
    x_bit=(u_dqdx_flux_in-u_dqdx_flux_out)
    y_bit=(v_dqdy_flux_in-v_dqdy_flux_out)
    z_bit=(w_dqdz_flux_in-w_dqdz_flux_out)
    net_flux=x_bit+y_bit+z_bit
    #
    net_flux_cube=field*0.0
    net_flux_cube.data[:,60:300,60:300]=net_flux
    return net_flux_cube;

def all_ml_processing_for_a_region_OLD(single_date,roseid,name_str,analysis_time,region_number,list_stream_3d,list_stash_sec_3d,list_stash_code_3d,list_stream,list_stash_sec,list_stash_code,reg_lat,reg_lon):
    #    region=name_of_hecto_lam(reg_lat[region_number],reg_lon[region_number])
    region=str(region_number)
    print('Processing region '+region+' ('+str(region_number+1)+' out of '+str(len(reg_lat))+')')
    #
    for vt in np.arange(0,22+1,2):
        # Here we are extracting theta, qv, qcl, qcf, qrain, qgraup, u, v, w, pressure and exner.
        # Use all that to calculate liquid water static temperature, hence liquid water potential temperature 
        # and its advective increments. Also calculate qtotal and its advective increments.
        #print('about to call: extract_fields_for_advective_tendencies')
        outcome=extract_fields_for_advective_tendencies(single_date,vt,roseid,name_str,analysis_time,region)
        #print('have called: extract_fields_for_advective_tendencies')
        #
        # Here we are extracting any other 3d fields not done as part of calculating theta-li, qt and their increments
        ndims=3
        for stream_number in np.arange(0,len(list_stream_3d),1):
            stream=list_stream_3d[stream_number]
            stash_sec=list_stash_sec_3d[stream_number]
            stash_code=list_stash_code_3d[stream_number]
            outcome=process_ml_lam_file(single_date,vt,roseid,name_str,analysis_time,stream,region,stash_sec,stash_code,ndims)
        # end (stream_number)
        #
        # Here we are extracting 2d data (e.g. surface fluxes, TOA SW etc).
        ndims=2
        for stream_number in np.arange(0,len(list_stream),1):
            stream=list_stream[stream_number]
            stash_sec=list_stash_sec[stream_number]
            stash_code=list_stash_code[stream_number]
            outcome=process_ml_lam_file(single_date,vt,roseid,name_str,analysis_time,stream,region,stash_sec,stash_code,ndims)
        # end (stream_number)
    # end (vt)
    outcome=1
    return outcome;

def load_in_one_file_for_cbh_ml(pp_file):
    warnings.filterwarnings("ignore","HybridHeightFactory")
    warnings.filterwarnings("ignore","orography")
    # Load in model diagnostics from pp file (for machine learning of cloud base height).
    # Each file contains 3d data (no time index)
    #
    # Load in specific humidity
    result = make_stash_string(0,10)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qv     = data.data
    # Load in bulk cloud fraction
    result = make_stash_string(0,266)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    bcf    = data.data
    # Load in pressure on theta levels
    result = make_stash_string(0,408)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    pres   = data.data
    # Load in temperature on theta levels
    result = make_stash_string(16,4)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    temp   = data.data
    # Data is 3d, but it will be easier to work with if the lat/lon are unfolded 
    # and we have a 2d distance-height curtain.
    nz,ny,nx = temp.shape
    temp     = np.reshape(temp,(nz,ny*nx))
    qv       = np.reshape(qv,  (nz,ny*nx))
    bcf      = np.reshape(bcf, (nz,ny*nx))
    pres     = np.reshape(pres,(nz,ny*nx))
    # Create new array to hold cloud base height    
    nz,ntotal=bcf.shape
    cbh=np.copy(bcf)*0.0    
    # Set a threshold for determining that cloud base has been found (e.g. 2 oktas)
    thresh=2.0/8.0
    # Simple search algorithm (done in a noddy way to be clear what is going on).
    for i in np.arange(0,ntotal,1):
        found=0
        for k in np.arange(0,nz,1):
            if found==0 and bcf[k,i]>thresh:
                cbh[k,i]=1.0
                found=1
    # Prepare to Normalise/standardise
    # Comment these back in to print out max and min values to help inform choice of hardwired values.
    # NB. You do NOT want to normalise each file using its own max and mean, because each file will then be normalised slightly differently!
    # print('maxtemp',np.amax(temp))
    # print('mintemp',np.amin(temp))
    # print('maxqv',np.amax(qv))
    # print('maxp',np.amax(pres))
    # Hardwired values found from inspecting one file
    max_temp = 320.0
    min_temp = 140.0
    max_qv   = 0.025
    max_pres = 106000.0
    # Normalise/standardise
    temp = (temp-min_temp) / (max_temp-min_temp)
    qv   = qv / max_qv
    pres = pres / max_pres
    # If no cloud base has been found, then set cloud base to be in top most layer of model (real cloud base is NEVER that high).
    for i in np.arange(0,ntotal,1):
        if np.amax(cbh[:,i])<0.5:
                cbh[69,i]=1.0
    # Combine all the variables together into a big array
    data=np.append(np.append(np.append(temp,qv,axis=0),pres,axis=0),cbh,axis=0)
    return {'data':data};

def load_in_one_file_for_massflux_ml(pp_file):
    # Load in model diagnostics from pp file (for machine learning of maximum massflux within a column).
    # Each file contains 3d data (no time index)
    #
    # Load in specific humidity
    result = make_stash_string(0,10)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qv     = data.data
    # Load in pressure on theta levels
    result = make_stash_string(0,408)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    pres   = data.data
    # Load in temperature on theta levels
    result = make_stash_string(16,4)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    temp   = data.data
    # Load in mass flux on theta levels
    result = make_stash_string(5,250)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    mf     = data.data
    # Load in w component of wind on theta levels
    result = make_stash_string(0,150)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    wwind  = data.data
    # Data is 3d, but it will be easier to work with if the lat/lon are unfolded 
    # and we have a 2d distance-height curtain.
    nz,ny,nx = temp.shape
    temp     = np.reshape(temp, (nz,ny*nx))
    qv       = np.reshape(qv,   (nz,ny*nx))
    mf       = np.reshape(mf,   (nz,ny*nx))
    pres     = np.reshape(pres, (nz,ny*nx))
    wwind    = np.reshape(wwind,(nz,ny*nx))
    # Prepare to Normalise/standardise
    # Comment these back in to print out max and min values to help inform choice of hardwired values.
    # NB. You do NOT want to normalise each file using its own max and mean, because each file will then be normalised slightly differently!
    print('maxtemp',np.amax(temp))
    print('mintemp',np.amin(temp))
    print('maxqv',np.amax(qv))
    print('maxp',np.amax(pres))
    print('max pmf',np.amax(mf))
    # Hardwired values found from inspecting one file
    max_temp  = 320.0
    min_temp  = 140.0
    max_qv    = 0.025
    max_pres  = 106000.0
    max_pmf   = 3.0
    typical_wwind = 0.1
    # Normalise/standardise
    temp  = (temp-min_temp) / (max_temp-min_temp)
    qv    = qv / max_qv
    pres  = pres / max_pres
    # Set stratospheric wind to zero
    wwind[51:70,:]=0.0
    print('max wwind',np.amax(wwind[0:50,:]))
    print('min wwind',np.amin(wwind[0:50,:]))
    print('mean abs wwind',np.mean(np.abs(wwind[0:50,:])))
    # Rescale wind and ensure within range of -1 to 1
    wwind = wwind / typical_wwind
    wwind = np.clip(wwind,-1.0,1.0)
    # Rescale massflux and ensure within range of 0 to 1
    mf    = mf / max_pmf
    mf    = np.clip(mf,0.0,1.0)
    # Combine all the variables together into a big array
    data=np.append(np.append(np.append(np.append(temp,qv,axis=0),pres,axis=0),wwind,axis=0),mf,axis=0)
    return {'data':data};

def load_in_one_file_for_surfprecip_ml(pp_file):
    # Load in model diagnostics from pp file (for machine learning of surface precipitation).
    # Each file contains 3d data (no time index)
    #
    # Load in specific humidity
    result = make_stash_string(0,10)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qv     = data.data
    # Load in pressure on theta levels
    result = make_stash_string(0,408)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    pres   = data.data
    # Load in temperature on theta levels
    result = make_stash_string(16,4)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    temp   = data.data
    # Load in large-scale rain rate
    result = make_stash_string(4,203)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    lsrr   = data.data
    # Load in large-scale snow rate
    result = make_stash_string(4,204)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    lssr   = data.data
    # Load in convective rain rate
    result = make_stash_string(5,205)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    cvrr   = data.data
    # Load in convective snow rate
    result = make_stash_string(5,206)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    cvsr  = data.data
    #
    # Data is 3d, but it will be easier to work with if the lat/lon are unfolded 
    # and we have a 2d distance-height curtain.
    nz,ny,nx = temp.shape
    temp     = np.reshape(temp, (nz,ny*nx))
    qv       = np.reshape(qv,   (nz,ny*nx))
    pres     = np.reshape(pres, (nz,ny*nx))
    lsrr     = np.reshape(lsrr, (1,ny*nx))
    lssr     = np.reshape(lssr, (1,ny*nx))
    cvrr     = np.reshape(cvrr, (1,ny*nx))
    cvsr     = np.reshape(cvsr, (1,ny*nx))
    # Prepare to Normalise/standardise
    # Comment these back in to print out max and min values to help inform choice of hardwired values.
    # NB. You do NOT want to normalise each file using its own max and mean, because each file will then be normalised slightly differently!
    #print('maxtemp',np.amax(temp))
    #print('mintemp',np.amin(temp))
    #print('maxqv',np.amax(qv))
    #print('maxp',np.amax(pres))
    # Note that a lot of the precip fields is zero. So feel free to calculate centiles having removed zero data.
    print('95th% and max lsrr',np.percentile(lsrr,95.0),np.amax(lsrr))
    print('95th% and max lssr',np.percentile(lssr,95.0),np.amax(lssr))
    print('95th% and max cvrr',np.percentile(cvrr,95.0),np.amax(cvrr))
    print('95th% and max cvsr',np.percentile(cvsr,95.0),np.amax(cvsr))
    # Hardwired values found from inspecting one file
    max_temp  = 320.0
    min_temp  = 140.0
    max_qv    = 0.025
    max_pres  = 106000.0
    # Normalise/standardise
    temp  = (temp-min_temp) / (max_temp-min_temp)
    qv    = qv / max_qv
    pres  = pres / max_pres
    ppn   = np.empty([4,nx*ny])+np.nan
    # Rescale each precip rate separately
    ppn[0,:] = lsrr / 3.0e-4
    ppn[1,:] = lssr / 2.0e-4
    ppn[2,:] = cvrr / 1.0e-3
    ppn[3,:] = cvsr / 3.0e-4
    # Combine all the variables together into a big array
    data=np.append(np.append(np.append(temp,qv,axis=0),pres,axis=0),ppn,axis=0)
    return {'data':data};

def load_in_one_file_for_cloud_ml(pp_file):
    # Load in model diagnostics from pp file (for machine learning of surface precipitation).
    # Each file contains 3d data (no time index)
    #
    # Load in specific humidity
    result = make_stash_string(0,10)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qv     = data.data
    # Load in pressure on theta levels
    result = make_stash_string(0,408)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    pres   = data.data
    # Load in temperature on theta levels
    result = make_stash_string(16,4)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    temp   = data.data
    # Load in QCL on theta levels
    result = make_stash_string(0,254)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qcl    = data.data
    # Load in QCF on theta levels
    result = make_stash_string(0,12)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qcf    = data.data
    # Load in QRAIN on theta levels
    result = make_stash_string(0,272)
    data   = iris.load_cube(pp_file,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qrain  = data.data
    #
    # Data is 3d, but it will be easier to work with if the lat/lon are unfolded 
    # and we have a 2d distance-height curtain.
    nz,ny,nx = temp.shape
    temp     = np.reshape(temp, (nz,ny*nx))
    qv       = np.reshape(qv,   (nz,ny*nx))
    qcl      = np.reshape(qcl,  (nz,ny*nx))
    qcf      = np.reshape(qcf,  (nz,ny*nx))
    qrain    = np.reshape(qrain,(nz,ny*nx))
    pres     = np.reshape(pres, (nz,ny*nx))
    #
    # Combine all moisture into a single variable
    qt=qv+qcl+qcf+qrain
    qliq=qcl+qrain
    #
    # Some constants
    cp=1005.0;
    lc=2.501e6;
    lf=0.334e6;
    #
    liq_correction=(  lc     / cp)  * qliq
    ice_correction=( (lc+lf) / cp ) * qcf
    temp_li=temp - (liq_correction+ice_correction)
    # Prepare to Normalise/standardise
    # Comment these back in to print out max and min values to help inform choice of hardwired values.
    # NB. You do NOT want to normalise each file using its own max and mean, because each file will then be normalised slightly differently!
    #    print('maxtemp',np.amax(temp))
    #    print('mintemp',np.amin(temp))
    #   print('maxqv',np.amax(qv))
    #   print('maxqt',np.amax(qt))
    #   print('maxp',np.amax(pres))
    #   print('maxqcl',np.amax(qcl))
    #   print('maxqcf',np.amax(qcl))
    #   print('maxqliq',np.amax(qliq))
    #  print('maxqrain',np.amax(qrain))
    # Hardwired values found from inspecting one file
    max_temp  = 320.0
    min_temp  = 140.0
    max_qv    = 0.025
    max_qt    = 0.025
    max_pres  = 106000.0
    # Note that a lot of the precip fields is zero. So feel free to calculate centiles having removed zero data.
    print('95th, 99th and max qliq',np.percentile(qt,95.0),np.percentile(qt,99.0),np.amax(qt))
    print('95th, 99th and max qliq',np.percentile(qliq,95.0),np.percentile(qliq,99.0),np.amax(qliq))
    print('95th, 99th and max qice',np.percentile(qcf, 95.0),np.percentile(qcf, 99.0),np.amax(qcf))
    max_qliq  = 1.0e-4
    max_qice  = 1.0e-4
    ##   max_qrain = 
    # Normalise/standardise
    temp_li = (temp_li-min_temp) / (max_temp-min_temp)
    qt      = qt   / max_qt
    pres    = pres / max_pres
    qliq    = qliq / max_qliq
    qcf     = qcf  / max_qice
    # Combine all the variables together into a big array: (IN: temp_li, qt, pressure; OUT: qliquid, qice)
    data    = np.append(np.append(np.append(np.append(temp_li,qt,axis=0),pres,axis=0),qliq,axis=0),qcf,axis=0)
    return {'data':data};

####################################################
# For machine learning moments
####################################################

def twodigit_num_to_str(number):
    if number < 10:
        number_str='0'+str(number)
    else:
        number_str=str(number)
    return number_str;

def plot_pcolor2(data1,data2):
    cmap = plt.get_cmap('bwr')
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    im = ax0.pcolormesh(data1,vmin=-1.0,vmax=1.0,cmap=cmap)
    fig.colorbar(im, ax=ax0)
    im = ax1.pcolormesh(data2,vmin=-1.0,vmax=1.0,cmap=cmap)
    fig.colorbar(im, ax=ax1)
    plt.show()
    return;

def plot_pcolor(data):
    plt.figure()
    plt.pcolormesh(data)
    plt.show()
    return;

def read_in_one_field_for_one_subdomain(domain,date,time,var_str,stash,name_str):
    # Read in one stash variable for a single LAM domain, which has data on a 14x14, and on nlev level grid.
    path       = '/scratch/frme/ML/u-cg081/netcdf/'

    domain_str = twodigit_num_to_str(domain)
    time_str   = twodigit_num_to_str(time)
    name       = '_km3p3_RA2M_32x32sampling_hence_14x14_time'
    nc_file    = path+date+'/r'+domain_str+'/'+date+'T0000Z_r'+domain_str+name+'0'+time_str+'_'+var_str+'_'+stash+'.nc'
    fh         = Dataset(nc_file, mode='r')
    data       = fh.variables[name_str][:,:]
    fh.close()
    return data;

def loop_over_lam_domains(date,time,var_str,stash,name_str,nlev,ndomains):
    # Read in data from all 99 domains and create a massive array, with each domain appended to previous ones.
    if nlev>1:
        collapse_along_axis=2
    else:
        collapse_along_axis=1
    domain   = 1
    big_data = read_in_one_field_for_one_subdomain(domain,date,time,var_str,stash,name_str)
    for domain in np.arange(2,ndomains+1,1):
        data     = read_in_one_field_for_one_subdomain(domain,date,time,var_str,stash,name_str)
        big_data = np.append(big_data,data,axis=collapse_along_axis)
        #print('domain,big_data.shape',domain,big_data.shape)
    return big_data;

def normalise(date,time,var_str,stash,flag,name_str,nlev,ndomains):
    big_data=loop_over_lam_domains(date,time,var_str,stash,name_str,nlev,ndomains)
    #
    # Read in the the profiles of maximum and minimum values ever encountered so far.
    file_max='/net/home/h01/frme/cyrilmorcrette-projects/python/wxyz/moments_profiles/'+var_str+'-'+stash+'-max-prof.txt'
    maximum=np.loadtxt(file_max)
    file_min='/net/home/h01/frme/cyrilmorcrette-projects/python/wxyz/moments_profiles/'+var_str+'-'+stash+'-min-prof.txt'
    minimum=np.loadtxt(file_min)
    #
    if flag==0:
        # Use this in training, calc max and from this sample
        if nlev > 1:
            # Get a profile of max and min if data is 3d (result is a (nlev,) vector)
            new_maximum=np.max(np.max(big_data,axis=2),axis=1)
            new_minimum=np.max(np.min(big_data,axis=2),axis=1)
        else:
            # Or just get a single scalar value of max and min if data is 2d (NB in this case max and mins are scalars, not (1,) vector)
            new_maximum=np.max(big_data)
            new_minimum=np.min(big_data)
        # Keep track of max and min value from earlier staored value and one just calculated
        maximum=np.maximum(new_maximum,maximum)
        minimum=np.minimum(new_minimum,minimum)
        # Write it out for comparison when reading in data from next time-stamp
        # NB have to add a blank vector to allow data to be written out even when it is a scalar.
        np.savetxt(file_max, maximum+np.zeros([1,nlev]), fmt='%10.7f')
        np.savetxt(file_min, minimum+np.zeros([1,nlev]), fmt='%10.7f')
    # else flag == 1:
    #    At inferrence, use the stored profiles that have been read in (not modified by current data!)
    #    as in principle, would want to infer from one profile at a time, for which max and min would make no sense.
    # Standardise on level by level basis
    if nlev > 1:
        for k in np.arange(0,nlev):
            if maximum[k]>minimum[k]:
                big_data[k,:,:]=(big_data[k,:,:]-minimum[k])/(maximum[k]-minimum[k])
            else:
                big_data[k,:,:]=big_data[k,:,:]-minimum[k]
    else:
        if maximum>minimum:
            big_data=(big_data-minimum)/(maximum-minimum)
        else:
            big_data=big_data-minimum
    return big_data;

def split_then_unfold(date,time,var_str,stash,flag,name_str,nlev,split_from,split_to,ndomains):
    # Get hold of data for one variable, lots of domains and normalised
    big_data = normalise(date,time,var_str,stash,flag,name_str,nlev,ndomains)
    # Split it into a training portion and a validation portion and reshape it into a curtain (or vector) of data.
    n = split_to - split_from
    if nlev > 1:
        data = big_data[:,split_from:split_to,:]
        data = np.reshape(data, (70, n*14*ndomains) )
    else:
        data = big_data[split_from:split_to,:]
        data = np.reshape(data,  (1, n*14*ndomains) )
    return data;

def get_data(date,time,flag,moment_to_learn,field_to_learn,name_to_learn,split_from,split_to):
    ndomains=99
    nlev=70
    var_str   = 'AVG'
    theta     = split_then_unfold(date,time,var_str,'00004',flag,'unknown',nlev,split_from,split_to,ndomains)
    qv        = split_then_unfold(date,time,var_str,'00010',flag,'unknown',nlev,split_from,split_to,ndomains)
    u_wind    = split_then_unfold(date,time,var_str,'00002',flag,'unknown',nlev,split_from,split_to,ndomains)
    v_wind    = split_then_unfold(date,time,var_str,'00003',flag,'unknown',nlev,split_from,split_to,ndomains)
    w_wind    = split_then_unfold(date,time,var_str,'00150',flag,'unknown',nlev,split_from,split_to,ndomains)
    pressure  = split_then_unfold(date,time,var_str,'00408',flag,'unknown',nlev,split_from,split_to,ndomains)
    #
    toa_sw    = split_then_unfold(date,time,var_str,'01207',flag,'unknown',1,split_from,split_to,ndomains)
    shf       = split_then_unfold(date,time,var_str,'03217',flag,'unknown',1,split_from,split_to,ndomains)
    lhf       = split_then_unfold(date,time,var_str,'03234',flag,'unknown',1,split_from,split_to,ndomains)
    land_mask = split_then_unfold(date,time,var_str,'00030',flag,'unknown',1,split_from,split_to,ndomains)
    avg_orog  = split_then_unfold(date,time,'AVG',  '00033',flag,'unknown',1,split_from,split_to,ndomains)
    std_orog  = split_then_unfold(date,time,'STD',  '00033',flag,'unknown',1,split_from,split_to,ndomains)
    skw_orog  = split_then_unfold(date,time,'SKW',  '00033',flag,'unknown',1,split_from,split_to,ndomains)
    bcu_ppn   = split_then_unfold(date,time,'BCU','0.0_0.0_0.5_PPN',flag,'stratiform_rainfall_flux',1,split_from,split_to,ndomains)
    #
    data      = np.append(theta,qv,axis=0)
    data      = np.append(data,u_wind,axis=0)
    data      = np.append(data,v_wind,axis=0)
    data      = np.append(data,w_wind,axis=0)
    data      = np.append(data,pressure,axis=0)
    #
    data      = np.append(data,toa_sw,axis=0)
    data      = np.append(data,shf,axis=0)
    data      = np.append(data,lhf,axis=0)
    data      = np.append(data,land_mask,axis=0)
    data      = np.append(data,avg_orog,axis=0)
    data      = np.append(data,std_orog,axis=0)
    data      = np.append(data,skw_orog,axis=0)
    data      = np.append(data,bcu_ppn,axis=0)
    #
    output    = split_then_unfold(date,time,moment_to_learn,field_to_learn,flag,name_to_learn,nlev,split_from,split_to,ndomains)
    data      = np.append(data,output,axis=0)
    #
    # Transpose ready for using np.permutations later
    data=np.transpose(data)
    return data;

def learn(date,time,model,keep_rmse_train,keep_rmse_val,moment_to_learn,field_to_learn,name_to_learn,path):
    n_mini_epochs        = 1
    batch                = 1000
    # Flag = 0 means training
    flag                 = 0
    #
    # Get training data
    split_from, split_to = 0, 12
    big_data             = get_data(date,time,flag,moment_to_learn,field_to_learn,name_to_learn,split_from,split_to)
    #
    # Shuffle
    shuffled_big_data    = np.random.permutation(big_data)
    # X is input info, Y is what we try to learn how to predict
    trainX               = shuffled_big_data[:,0:428]
    trainY               = shuffled_big_data[:,428:498]
    #
    # Get validation data
    split_from, split_to = 12, 14
    big_data             = get_data(date,time,flag,moment_to_learn,field_to_learn,name_to_learn,split_from,split_to)
    testX                = big_data[:,0:428]
    testY                = big_data[:,428:498]
    #
    history              = model.fit(trainX, trainY, validation_data=(testX, testY), 
                                     epochs=n_mini_epochs, batch_size=batch, shuffle=True)
    #
    # Store metrics to track progress
    keep_rmse_train=np.append(keep_rmse_train,history.history['root_mean_squared_error'])
    keep_rmse_val  =np.append(keep_rmse_val,  history.history['val_root_mean_squared_error'])
    #
    # Write out some metrics to track progress. You need to read these back in later and plot them.
    fileout=path+'/ml_moments_timeseries_rmse_train_'+moment_to_learn+'_'+field_to_learn+'.txt'
    np.savetxt(fileout, np.ones((1,1))*keep_rmse_train, fmt='%10.7f')
    fileout=path+'/ml_moments_timeseries_rmse_val_'+moment_to_learn+'_'+field_to_learn+'.txt'
    np.savetxt(fileout, np.ones((1,1))*keep_rmse_val, fmt='%10.7f')
    #
    return {'keep_rmse_train':keep_rmse_train, 'keep_rmse_val':keep_rmse_val};    

def infer(date,time,model,moment_to_learn,field_to_learn,name_to_learn):
    print(moment_to_learn,field_to_learn)
    # Flag = 1 means inferrence
    flag                 = 1
    # Learning was done on python rows 0-12, hence do validation on rows 12-14 NOT used in training 
    # (i.e. train on rows 1-12, val on 13,14)...
    split_from, split_to = 12, 14
    # ... Or as a sanity check, can validate on two rows that WERE used in training, as here it OUGHT to do very well
    # and it would be a red flag if it didn't.
    split_from, split_to = 2, 4
    big_data             = get_data(date,time,flag,moment_to_learn,field_to_learn,name_to_learn,split_from,split_to)
    X                    = big_data[:,0:428]
    truth                = big_data[:,428:498]
    #
    pred=model.predict(X)
    #
    truth=np.transpose(truth)
    pred=np.transpose(pred)
    truth=np.reshape(truth,(70*2*14*99))
    pred=np.reshape(pred,(70*2*14*99))
    #
    rmse=np.sqrt(np.mean((pred-truth)**2.0))
    me=np.mean(pred-truth)
    print('rmse=',rmse,' me=',me)
    #
    xmin=0
    xmax=1
    dx=0.02
    ymin=0
    ymax=1
    dy=0.02
    result=histogram_2d(truth,pred,xmin,xmax,dx,ymin,ymax,dy)
    return result;

def histogram_2d(data1,data2,xmin,xmax,dx,ymin,ymax,dy):
    x=np.arange(0,((xmax-xmin)/dx)+2)*dx
    y=np.arange(0,((ymax-ymin)/dy)+2)*dy
    hist=np.zeros([len(x),len(y)])
    n1=len(data1)
    n2=len(data2)
    if (n1!=n2):
        print('Data arrays not the same size!')
    for i in np.arange(0,n1):
        x_index=np.amin([np.amax([np.int(np.ceil(data1[i]/dx)),0]),len(x)-1])
        y_index=np.amin([np.amax([np.int(np.ceil(data2[i]/dy)),0]),len(y)-1])
        #print(x_index,y_index)
        hist[x_index,y_index]=hist[x_index,y_index]+1.0
    p=np.polyfit(data1,data2,1)
    return {'hist':hist, 'x':x, 'y':y, 'p':p};

def loop_over_timesteps(date,model,keep_rmse_train,keep_rmse_val,moment_to_learn,field_to_learn,name_to_learn,path):
    for time in np.arange(0,72,3):
        print('time=',time)
        result=learn(date,time,model,keep_rmse_train,keep_rmse_val,moment_to_learn,field_to_learn,name_to_learn,path)
        keep_rmse_train=result['keep_rmse_train']
        keep_rmse_val=result['keep_rmse_val']
    return {'keep_rmse_train':keep_rmse_train, 'keep_rmse_val':keep_rmse_val};

def loop_over_dates(start_date,end_date,model,keep_rmse_train,keep_rmse_val,moment_to_learn,field_to_learn,name_to_learn,path):
    for single_date in daterange(start_date, end_date):
        print(single_date.strftime("%Y%m%d"))
        result=loop_over_timesteps(single_date.strftime("%Y%m%d"),model,keep_rmse_train,keep_rmse_val,moment_to_learn,field_to_learn,name_to_learn,path)
        keep_rmse_train=result['keep_rmse_train']
        keep_rmse_val=result['keep_rmse_val']
    return {'keep_rmse_train':keep_rmse_train, 'keep_rmse_val':keep_rmse_val};

def loop_over_epochs(start_date,end_date,model,n_superloop,moment_to_learn,field_to_learn,name_to_learn,path):
    # Set up arrays to store the rmse as one loops through multiple epochs.
    keep_rmse_train=np.empty([1,1])+np.nan
    keep_rmse_val  =np.empty([1,1])+np.nan
    for superloop in np.arange(1,n_superloop+1):
        print('Epoch=',superloop)
        result=loop_over_dates(start_date,end_date,model,keep_rmse_train,keep_rmse_val,moment_to_learn,field_to_learn,name_to_learn,path)
        keep_rmse_train=result['keep_rmse_train']
        keep_rmse_val=result['keep_rmse_val']
        weights_file=path+'/ml_moments_latest_saved_weights_'+moment_to_learn+'_'+field_to_learn+'.h5'
        model.save_weights(weights_file)
    return;

