# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:02:10 2022

@author: jcfit

What is the end result? A class which contains for each event, the [t,omni],[t,mu,pads],[t,Hz,wind_waves_spectrum]
so that one can call event.omni, event.pads, event.waves

So what do we need to do? We have as input the folder that contains all the csv files
-lets start simple (do not check if the data has already been processed, to begin with this may not be a speed problem and can be implemented later)
-since all the data is (in theory) identical in form for each data, need 3 functions
omni_csv_to_numpy(csv) which oututs t,omni in numpy array 
pads_csv_to_numpy(csv) which outputs t, mu, pads in numpy array
waves_csv_to_numpy(csv) which outputs t,nu,waves in numpy array

-a cleaning procedure for omni and pads to take away the background fluxes in each channel

import the class from WIND_event_class and save the event data in a class object!

how should we store the data? save the file and alter the name with t0_type_originalname
first lets collect some data and inspect the csv files before trying any nonsense
"""

import sys
from pathlib import Path
#importing the global functions file
import util.global_functions as funcs
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import re
import numpy as np
import cdflib
from matplotlib.gridspec import GridSpec

import scipy

#import warnings
#warnings.simplefilter("ignore") 


'''
When running user_requests all WIND data will be processed AUTOMATICALLY

'''
        
class WIND_functions:
    #takes as input the event_csvs
    def __init__(self,dicts,events_cdf,wind_events):
        #obtaining the file path of util
        self.path_util = os.path.dirname(os.path.realpath(__file__))
        #self.csv_folder='WIND_csv'
        #self.path_csv = os.path.join(self.path_util,self.csv_folder)
        self.cdf_folder='WIND_cdf'
        self.path_cdf=os.path.join(self.path_util,self.cdf_folder)
        #self.event_csv=event_csv
        self.events_cdf=events_cdf
        #defining the wind_events class
        self.wind_events=wind_events
        
        #pulling out the event we are considering
        self.WIND_dict = dicts['WIND_dict']
        self.t_evaluate_background = self.WIND_dict['t_evaluate_background[mins]']
        self.event_considering=self.WIND_dict['event_considering']
        self.window_size = self.WIND_dict['window_size[s]']

        #pulling out the simulated energy arrays
        self.sim_dict=dicts['sim_dict']
        self.simulated_energies = self.sim_dict['energies[keV]']
        #array holding all the WIND energies
        self.WIND_energies = [27,40,67,110,180,310,520]


    #function that extracts the indices in the WIND data of the simulated energies 
    def simmed_energy_indices(self,):
        #first building the WIND_energy_dictionary 27=0, 40=1 and so on..
        WIND_energy_dict = {}
        for i,energy in enumerate(self.WIND_energies):
            WIND_energy_dict[str(energy)]=i

        #extracting the indices of the simulated energies using the constructed dictionary
        simmed_energy_indices=[]
        #also returning a dictionary featuring only the elements of the energies selected!
        simmed_energy_dict = {}
        for i,energy in enumerate(self.simulated_energies):
            simmed_energy_indices.append(WIND_energy_dict[str(energy)])   
            simmed_energy_dict[str(energy)]=i  
        return simmed_energy_indices,simmed_energy_dict


    def datetime64_todatetime(self,datetime64_ar):
        #converts an array of datetime64 in the form %Y-%m-%dT%H:%M:%S.%f000 to datetimes
        #create an array
        t_dt = []
        #convert t to datetime array
        for time64 in datetime64_ar:
            #convert the datetime64 to a string
            time_str=str(time64)
            #use strptime to convert the string to a datetime
            dt = datetime.strptime(time_str,'%Y-%m-%dT%H:%M:%S.%f000')
            #append this datetime to t_dt
            t_dt.append(dt)
        #return the datetime array
        return np.array(t_dt)
    
    #given a datetime array and a datetime t, output the time index closes to T
    def tindex(self,T,datetimes):
        t = []
        for tau in range(len(datetimes)):
            sec = (datetimes[tau]-T).total_seconds()
            t.append(sec)
        #modelus of t, then can find minimum
        t=np.abs(t)
        #True if the value is closest to T
        boole = t==np.min(t)
        #what is the index?
        try:
            ind = np.where(boole)[0]
            return int(ind)
        except:
            ind = np.where(boole)[0][0]
            return int(ind)
        
    #convert 
    def datetime_to_float(self,datetimes):
        stamps=[]
        for t in datetimes:
            stamps.append(t.timestamp())
        return np.array(stamps)
    
    def process_omni_cdf(self,t_bounds,omni_name):
        #calling the path to the cdf folder
        path=self.path_cdf
        #extracting the omnidirecitonal name from the dictionary
        file=omni_name
        #creating the full path to the file to be loaded
        file_path =os.path.join(path,file)
        #the master cdf file contains data for the whole day!
        cdf_file = cdflib.CDF(file_path)
        #convert to an xarray
        data_xarray = cdflib.cdf_to_xarray(file_path, to_datetime=True)
        #extract the dimension information
        n_ener=data_xarray.dims['dim0']

        #extract the data of an arbitrary energy, all times should be identical
        data_t = data_xarray.isel(dim0=0)
        #extract the datetime64 and fluxes for the days worth of data
        t=np.array(data_t.Epoch)
        #convert to datetimes (this will be identical for each energy so just use the last 'data' used)
        t_datetimes = self.datetime64_todatetime(t)
        #return the indices of the start and end of the time period required
        ind_start,ind_end = self.tindex(t_bounds[0],t_datetimes),self.tindex(t_bounds[1],t_datetimes)
        #trim the times and flux data
        t_datetimes=t_datetimes[ind_start:ind_end+1]

        t_res = (t_datetimes[1]-t_datetimes[0]).total_seconds()
        
        #find the time after evaluatating the background
        finish_background=t_datetimes[0]+timedelta(minutes=self.t_evaluate_background)
        #find the index in the array at which the background should complete
        finish_background_ind = self.tindex(finish_background,t_datetimes)
        
        n_times_afterslice = ind_end-ind_start+1
        omni_fluxes=np.empty((n_ener,n_times_afterslice))
        omni_fluxes_smoothed = np.empty((n_ener,n_times_afterslice))
        #assume times are post bins
        #for each energy:
        for n in range(n_ener):
            #extract the data of the nth energy
            data = data_xarray.isel(dim0=n)
            #extract the flux data (sliceing at the appropriate bounds to get the times wanted)
            flux = np.array(data.FLUX)[ind_start:ind_end+1]
            #perform the background correction
            #evaluate the background flux
            background_flux = np.nanmean(flux[0:finish_background_ind+1])
            #correct for the background
            flux_background_corrected=flux-background_flux 
            #apply a smoothing savitsky-golay filter for calculation of t_d and t_r
            #calculate the equivalent
            window_size_num_data_points = int(self.window_size/t_res)
            #print(window_size_num_data_points*t_res)
            #print(self.window_size)
            #raise SystemExit
            flux_smoothed = scipy.signal.savgol_filter(flux_background_corrected,window_size_num_data_points,3)
            #normalise with respect to the peak intensity
            flux_normalised=flux_background_corrected/np.nanmax(flux_smoothed)
            flux_smoothed/=np.nanmax(flux_smoothed)
            '''
            plt.plot(t_datetimes,flux_normalised)
            plt.plot(t_datetimes,flux_smoothed)
            plt.show()
            '''
            #normalise to the peak intensity
            omni_fluxes[n,:]=flux_normalised
            omni_fluxes_smoothed[n,:]=flux_smoothed
        #one less time slice for fluxes since last time will be an UPPER EDGE (assuming the times are post edges)
        #omni_fluxes=omni_fluxes[...,ind_start:ind_end]

        '''
        Extract only the energies that have been chosen to simulate!
        '''
        #finding the indices of the simulated energies
        simmed_energy_indices,simmed_energy_dict=self.simmed_energy_indices()
        #and extracting only those energies for the analysis
        omni_fluxes = omni_fluxes[simmed_energy_indices]
        omni_fluxes_smoothed = omni_fluxes_smoothed[simmed_energy_indices]
        return [t_datetimes,omni_fluxes,omni_fluxes_smoothed],simmed_energy_dict
    
    
    def process_pads_cdf(self,t_bounds,pads_name,add_edge):
        #calling the path to the cdf folder
        path=self.path_cdf
        #calling the name of the exact file within the folder
        file=pads_name
        #creating the file path
        file_path =os.path.join(path,file)
        #the master cdf file contains data for the whole day!
        cdf_file = cdflib.CDF(file_path)
        

        #convert to an xarray
        data_xarray = cdflib.cdf_to_xarray(file_path, to_datetime=True)
        
        #print(data_xarray)
        #print(data_xarray)
        #extract the dimensions of the angular bin edges and the energies
        #n_angs is the number of bins
        n_angs=data_xarray.dims['dim0']
        #n_ener is the number of energies
        n_ener=data_xarray.dims['dim1']
        try:
            n_times_total=data_xarray.dims['record0']
        except:
            n_times_total=data_xarray.dims['Epoch']
        
        #extract the data of an arbitrary energy, all times should be identical
        data_t = data_xarray.isel(dim1=0)
        #extract the datetime64 and fluxes for the days worth of data
        t=np.array(data_t.Epoch)
        #convert to datetimes (this will be identical for each energy so just use the last 'data' used)
        t_datetimes = self.datetime64_todatetime(t)
        #return the indices of the start and end of the time period required
        ind_start,ind_end = self.tindex(t_bounds[0],t_datetimes),self.tindex(t_bounds[1],t_datetimes)
        #trim the times to the required time
        t_datetimes=t_datetimes[ind_start:ind_end+1]

        
        #find the time after evaluatating the background
        finish_background=t_datetimes[0]+timedelta(minutes=self.t_evaluate_background)
        #find the index in the array at which the background should complete
        finish_background_ind = self.tindex(finish_background,t_datetimes)
        #number of times after slicing to the specified time region
        n_times_afterslice = ind_end-ind_start
        
        #creating an empty array to hold the pitch angle distirbutions
        pads = np.empty((n_ener,n_times_afterslice,n_angs))
        #loop through each particle energy
        for n in range(n_ener):
            #extract the data of the nth energy
            data = data_xarray.isel(dim1=n)
            #extract the pad data (days worth!)
            pad = np.array(data.FLUX)
            #extracting the angular edges
            angs = data.PANGLE

            #angs holds the angular bins at each timestep
            angs=np.array(angs)        
            '''
            fix the bad bin handler!
            '''
            #find where the angular bins are not defined (due to errors in mean magnetic field measurement?)
            where_nan = np.isnan(angs)
            
            #may be different for each event, necessary for plotting. Replace nan edges with the added edges
            angs[where_nan]=add_edge

            #should not have any effect on analysis
            #i.e. where the bin edges are unavailable
            #setting all the flux values at these locations to be nans (i.e. dont plot them)
            pad[where_nan]=np.nan
            
            '''
            performing the background correction
            '''
            #first trim the pad to the required time range
            pad = pad[ind_start:ind_end]   
            #building the background pad, over which we will take the mean
            background_pad=pad[0:finish_background_ind+1]
            background_flux = np.nanmean(background_pad)
            
            
            #perform the background correction
            pad-=background_flux
            
            #normalising to the peak intensity
            pad/=np.nanmax(pad)
            #place the pitch angle distribution into the multidim array
            pads[n]=pad
    

        #adding in an edge at 180 deg
        angs=np.insert(angs,8,180,axis=-1)
        #slicing the angles to the required time range
        angs=angs[ind_start:ind_end+1]    
        #swap the axes to allow for plotting
        angs=np.swapaxes(angs,0,1)
        
        #convert the angular bins to pitch angßle cosine bins
        #conversions to muspace, simple after finding out what the bin edges are!
        #mu_beds_wind = np.cos(angs*np.pi/180)     
        #find the location of the maximum (whether it is in positive or negative mu)
        bin_ind_ofmax = np.where(pads==np.nanmax(pads))[-1][0]  


        where_nan = np.isnan(angs)
        #if the magnetic field so aligned that the peak flux happens at mu near -1 flip the bins       
        if bin_ind_ofmax<3:
            #reverse the bins!
            for n in range(n_times_afterslice+1):
                angs[:,n]=angs[::-1,n]

        '''
        Extract only the energies that have been chosen to simulate!
        '''
        #finding the indices of the simulated energies
        simmed_energy_indices,_=self.simmed_energy_indices()
        #extracting only those energies from pads
        pads = pads[simmed_energy_indices]
        return [t_datetimes,angs,pads]


    def process_waves_cdf(self,t_bounds,waves_name):
        #calling the path to the cdf folder
        path=self.path_cdf
        #calling the name of the exact file within the folder
        file=waves_name
        #creating the file path
        file_path =os.path.join(path,file)
        #the master cdf file contains data for the whole day!
        cdf_file = cdflib.CDF(file_path)
        #convert to an xarray
        data_xarray = cdflib.cdf_to_xarray(file_path, to_datetime=True)

        #now extract the rad1 and rad2 voltage data, frequency bins and the time bin for the DAY of data
        freq_rad1 = np.array(data_xarray.Frequency_RAD1)
        freq_rad2 = np.array(data_xarray.Frequency_RAD2)
        volts_rad1 = np.array(data_xarray.E_VOLTAGE_RAD1)
        volts_rad2 = np.array(data_xarray.E_VOLTAGE_RAD2)
        
        times = np.array(data_xarray.Epoch)
        #convert to datetimes (this will be identical for each energy so just use the last 'data' used)
        t_datetimes = self.datetime64_todatetime(times)
        
        '''
        #finding the start and end indices given the event time
        t_start=datetime(year=2002,month=10,day=20,hour=14)
        t_end = datetime(year=2002,month=10,day=20,hour=15,minute=30)
        ind_start=self.tindex(t_start,t_datetimes)
        ind_end = self.tindex(t_end,t_datetimes)
        '''
        ind_start,ind_end = self.tindex(t_bounds[0],t_datetimes),self.tindex(t_bounds[1],t_datetimes)
        #trimming the data
        t_datetimes=t_datetimes[ind_start:ind_end+1]
        #trimming off a value at the end f the time and frequency dimensions
        volts_rad1=volts_rad1[ind_start:ind_end,:-1]
        volts_rad2 = volts_rad2[ind_start:ind_end,:-1]
        
    
        return [t_datetimes,freq_rad1,volts_rad1,freq_rad2,volts_rad2]
    
    
    def process_wind(self):
        #extrcating the data file names of the selected event
        #last entry is the "add_edge" variable, stored specifically for each event.
        cdf_datas=self.events_cdf[self.event_considering][:-1]
        #extracting "add_edge" from the event dictionary
        add_edge = self.events_cdf[self.event_considering][-1]

        #t_bounds is the start and end time (recall the cdf file contains the DAYS worth of data)
        t_bounds,omni_name_cdf,pads_name_cdf,waves_name_cdf=cdf_datas
        #unpack the omnidirectional intensity data from the downloaded cdf
        omni_data_fromcdf,simmed_energy_dict=self.process_omni_cdf(t_bounds,omni_name_cdf)
        #unpacking the pitch angle distribution data. NOTE here, the angle beds are array like
        pads_data_fromcdf=self.process_pads_cdf(t_bounds,pads_name_cdf,add_edge) 
        #unpack the wind/waves data
        waves_data_cdf=self.process_waves_cdf(t_bounds,waves_name_cdf)
        #as for the csv data pack these into a dictionary
        event_data=self.wind_events(omni_data_fromcdf,pads_data_fromcdf,waves_data_cdf,self.simulated_energies)
        #now save the two different data sets! We will compare in plotting....
        data_path=funcs.folder_functions().return_datapath()
        #np.save(os.path.join(data_path,"events_dict"),events_dict,allow_pickle=True)
        #saving event_data and the simmed_energy_dict (latter is needed for event fitting and plotting)
        np.save(os.path.join(data_path,"event_data"),event_data,allow_pickle=True)
        np.save(os.path.join(data_path,"simmed_energy_dict"),event_data,allow_pickle=True)
        #print(events_data.omni_times)
        #return the (unsorted! event_data)
        return event_data,simmed_energy_dict
    
        
   
    
'''
fix NAN times! stairs does not work......
Need to fix UNITS
need to background correct!
Why are the summed pads different from the omnidirectional intensities?
Check other events!
'''


    
'''
NOTES
time arrays are different between omnidirectional and pads data
slightly different structures between directly obtained omnidirectional data and summed pads (in shape and numeric value)
derived energies are different between omnidirectional and pads data?
we ignore angular information derivedfrom the headers and simply assume 8 equidistant angular bins
we ignore the energies derived from the headers and use approximate values
'''
    
   
    
   
    
   
    
   
    
   
    
   
    