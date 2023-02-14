# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:19:32 2022

@author: jcfit

In this script, we will take in the requested event, and perform the 
"""

import sys
from pathlib import Path
#importing the global functions file
import util.global_functions as funcs
import os
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime,timedelta

import scipy
import matplotlib.dates as mdates


###class that will sort the wind data (i.e., find the onset and decay times, asymmetric HWHM, background normalise the data, find the injection time ect.)
### t_e,t_o,HWHM-,HWHM+ will be stored for each energy
class sort_WIND_data:
    def __init__(self,event_data,dicts,simmed_energy_dict):
        self.event_data=event_data
        self.WIND_dict=dicts['WIND_dict']
        self.inj_max_freq = self.WIND_dict['inj_max_freq[kHz]']
        self.thresh = self.WIND_dict['thresh']
        self.inj_manual_override = self.WIND_dict['inj_manual_override']
        self.inj_manual_override_date = self.WIND_dict['inj_manual_override_date']
        self.validate_omni_chars = self.WIND_dict['validate_omni_chars']
        self.simmed_energy_dict=simmed_energy_dict
        
    #given a value and an array, find the index of the bin to which the value is closest
    #ar must be 1D!
    def find_index_1D(self,ar,val):
        #if not already, convert the array to a numpy array
        ar=np.array(ar)
        #take val away from ar
        ar=ar-val
        #take the absolute magnitude of ar
        ar=np.abs(ar)
        return np.where(ar==np.nanmin(ar))[0][0]
    '''
    def estimate_uncertainty_fluxs(self,fluxs,omni_times,t_res,thresh,max_index,string):

            signed_fluxs = np.sign(fluxs-thresh)
            #1 if the flux curve crosses over the 0.5 flux threshold
            diff_fluxs = np.abs(np.diff(signed_fluxs)/2)
            #extracting the indices where this occurs
            indx = np.argwhere(diff_fluxs>0)
            if string=='pre':
                #holds true if that index is pre the maximum
                true_if = indx<max_index
                indx = indx[true_if]
            if string=='post':
                true_if=indx>max_index
                indx=indx[true_if]
            else:
                pass
            
            indx_minimum,indx_maximum = int(np.nanmin(indx)),int(np.nanmax(indx))

            #seperate these into intersections pre and post the maximum
            #converting these indices to times (first premax then postmax)
            return [omni_times[indx_minimum],omni_times[indx_maximum]],(omni_times[indx_maximum]-omni_times[indx_minimum]).total_seconds()
    '''      
        
    def charecterise_event_omni(self,inj_time):
        #extract the omnidirectional intensities and the omnidirectional fluxes
        omni_times = self.event_data.omni_times
        omni_fluxes=self.event_data.omni_fluxes
        omni_fluxes_smoothed = self.event_data.omni_fluxes_smoothed
        
        #finding approximate time resolution
        t_res = (omni_times[30]-omni_times[29]).total_seconds()


        
        n_ener = np.shape(omni_fluxes)[0]

        

        #each key will be the energy in keV, and each element will be the dictionary of ascociated charecteristics
        event_chars_dict = {}
        
        #now that everything is in seconds, should make this into a numpy array
        #CONVENTION last axis will be t_peak,t_r,t_d
        event_chars = np.empty((n_ener,3))
        event_chars_uncertainties = np.empty((n_ener,3))
        
        #for each energy...
        for n in range(n_ener):
            energy = self.event_data.energies[n]
            str_energy = str(energy)+'keV'           
            #create a dictionary that will hold the time charecteristics for the given energy
            energy_chars_dict = {}
            #extract the nth omnidirectional fluxes
            fluxs = omni_fluxes[n]
            fluxs_smoothed=omni_fluxes_smoothed[n]
            
            #converting the times to floats
            #t = funcs.datetime_to_float(omni_times)

            #find the maximum flux
            #peak normalised, so this value will always be one (when written no normalisation had been performed)
            max_flux = np.nanmax(fluxs)
            #divide the flux into pre max and post max i.e. find the index where the maximum flux is)
            #for now don't smooth with zaavitsky golay filter, just raw maximum, may be noisy for low energy channels
            max_index = funcs.find_index_1D(fluxs_smoothed,max_flux)
            
            #find the time of the maximum
            t_max =omni_times[max_index]
            #find half the maximum flux
            #thresh_flux = max_flux/2
            thresh_flux=self.thresh

            
            #change to quarter flux in the future for more asymmetry?
            #thresh_flux=0.5
            #find the index corresponding to the half max flux BEFORE the peak
            ind_half_max_pre = funcs.find_index_1D(fluxs_smoothed[:max_index],thresh_flux)
            #find the ascociated time
            #finding the 'onset' time
            t_r_datetime = omni_times[ind_half_max_pre]
            
            #similarly for post the peak
            ind_half_max_post = funcs.find_index_1D(fluxs_smoothed[max_index:-1],thresh_flux)
            #finding the 'decay' times
            t_d_datetime = omni_times[max_index+ind_half_max_post+1]
            
            #forming the pair of half width at half maximums
            #HWHM_premax = (t_max-t_o_datetime).total_seconds()
            #HWHM_postmax = (t_d_datetime-t_max).total_seconds()
            
            #calculating the peak time (seconds since injection)
            t_peak = (t_max - inj_time).total_seconds()
            #calculating the rise time (from half peak flux, pre maximum)
            t_r = (t_max-t_r_datetime).total_seconds()
            #calculating teh decay time (from half peak flux, post maximum)
            t_d = (t_d_datetime-t_max).total_seconds()

            '''
            estimating the uncertainties using the interesection method
            first for t_r and t_d
            '''

            #returns the temporal boundary times, and the width of the intersection boundaries
            #intersections_postmax,t_diff_postmax = self.estimate_uncertainty_fluxs(fluxs,omni_times,t_res,0.5,max_index,'post')
            #intersections_premax,t_diff_premax = self.estimate_uncertainty_fluxs(fluxs,omni_times,t_res,0.5,max_index,'pre')
            #intersections_max,t_diff_peak = self.estimate_uncertainty_fluxs(fluxs,omni_times,t_res,1.0,max_index,'max')



            
            #inferring teh estimated uncertainties as the maximum between the time binning and half the difference computed from the intersection boundaries
            #set to instrumental resolution
            delta_t_peak = t_res/2
            delta_t_r = t_res
            delta_t_d= t_res

        
            
            
            
            
            if self.validate_omni_chars==True:
                fig,ax=plt.subplots(1,figsize=(7,7))
                WIND_energies = self.event_data.energies
                color=plt.cm.Dark2(n)
                if self.validate_omni_chars==True:
                    ax.plot(omni_times,fluxs,color=color)
                    ax.plot(omni_times,fluxs_smoothed,color='black')
                    ax.axvline(t_r_datetime,color='blue',linestyle='--')
                    ax.axvline(x=t_max,color='red',linestyle='--')
                    ax.axhline(y=thresh_flux,color='grey',linestyle='--')
                    ax.axvline(x=t_d_datetime,color='green',linestyle='--')
                    ax.axvline(x=inj_time,color='grey',linestyle='--')
                    #ax.axvspan(xmin=intersections_premax[0],xmax=intersections_premax[1],color='lightgrey')
                    #ax.axvspan(xmin=intersections_postmax[0],xmax=intersections_postmax[1],color='lightgrey')
                    #ax.axvspan(xmin=intersections_max[0],xmax=intersections_max[1],color='lightpink')
                    ax.axhline(y=0.5,linestyle='--',color='grey')
                    ax.axhline(y=1.0,linestyle='--',color='grey')
                    str_energy=str(WIND_energies[n])+'keV'
                    ax.annotate(str_energy, (0.75,0.7),xycoords='axes fraction',fontsize=10,color=color)
                    
                    '''
                    formatting
                    '''
                    
                    ax.tick_params(labelbottom=True)
                    ax.xaxis.set_tick_params(labelsize=10)
                    ax.set_xlabel('UT Time',fontsize=15)
                    ax.set_ylabel('Normalised Flux', fontsize=15)
                    myFmt = mdates.DateFormatter('%H%M')
                    ax.xaxis.set_major_formatter(myFmt)
                    plt.show(block=True)
                    #saving the event_chars_dict!
                    figure_path = r'\Users\jcfit\Desktop\Transport Modelling\Electron Transport Modelling\Numerical Modelling Scripts\transport_model\util\transport_simulation_figures'
                    #plt.savefig(os.path.join(figure_path,'fig+all_test_figures+'+str(n)+'_.pdf'),format='pdf',bbox_inches='tight')

            #placing the time profile data in a dictionary
            #t_o and t_d are defined for seconds after injection
            energy_chars_dict['t_peak[s]']=t_peak
            energy_chars_dict['t_r[s]']=t_r
            energy_chars_dict['t_d[s]']=t_d

                        
            #placing this dictionary in the global event charecteristics dictionary,keyed by the energy
            event_chars_dict[str_energy]=energy_chars_dict

            
            
            #place equivalently into a numpy array
            event_chars[n]=t_peak,t_r,t_d
            event_chars_uncertainties[n]=delta_t_peak,delta_t_r,delta_t_d
    

     

        #saving the event_chars_dict!
        data_path=funcs.folder_functions().return_datapath()
        #done!   
        #saving both the dictionary and numpy array for convenience
        np.save(os.path.join(data_path,"event_chars_dict"),event_chars_dict,allow_pickle=True)    
        np.save(os.path.join(data_path,"event_chars"),event_chars)  
        np.save(os.path.join(data_path,"event_chars_uncertainties"),event_chars_uncertainties)  

        return
    
    #auto find the injection time
    #only need the data from rad2 (high frequency)
    def find_injection_time(self):
        #extracting the high frequency WAVES data
        frqs=self.event_data.waves_freq_rad2
        volts=self.event_data.waves_volts_rad2

        ts = self.event_data.waves_times 
        
        #find where the frequency array is closest to the requested injection frequency
        where_frq = funcs.find_index_1D(frqs,self.inj_max_freq)
        #slicing the dynamic spectra at the requested frequency
        volts_requested = volts[:,where_frq]      
        #finding the maximum of the slice of the dynamic spectrum requested
        max_volts_requested=np.nanmax(volts_requested)
        

        ind_max_volts = np.where(volts_requested==max_volts_requested)[0][0]

        #finding the time of the maximumfor the requested frequency
        t_max = ts[ind_max_volts]
        
        #calculate the free propagation time in minutes (1AU travel time for the radio burst) 
        #speed of light in ms-1 converted to au/day
        #
        radio_travel_time_days = 1/(299792458*5.77548e-7)
        radio_travel_time_minutes=radio_travel_time_days*1440
        radio_travel_time_minutes = funcs.light_travel_time_1AU()


        inj_time = t_max-timedelta(minutes=radio_travel_time_minutes)

        '''
        #manual override if necessary, incorporate into dictionary for ease and use a manual injection override in the interface
        '''
        #manual override if necessary
        if self.inj_manual_override==True:
            inj_time =self.inj_manual_override_date-timedelta(minutes=radio_travel_time_minutes)
        
        #saving the injection time!
        data_path=funcs.folder_functions().return_datapath()
        #done!
        np.save(os.path.join(data_path,"injection_time"),inj_time,allow_pickle=True) 
        return inj_time
        
    
    def sort_WIND(self):
        #find the injection time based on WIND data
        inj_time = self.find_injection_time()
        #find the onset and decay times for each energy
        self.charecterise_event_omni(inj_time)
        return
        
     
        
    
    
