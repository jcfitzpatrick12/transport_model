# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:19:32 2022

@author: jcfit

In this script, we will build the class using the saved processed data in the folder WIND_processed
"""

import numpy as np

#creating a simple class to handle the event extractin of data
class WIND_events:
    def __init__(self,omni_data,pads_data,waves_data,simulated_energies):
        self.omni_times=omni_data[0]
        self.omni_fluxes=omni_data[1]
        self.omni_fluxes_smoothed = omni_data[2]
        #assume energies in keV, global energy array. We will extract the energies we wish to that are simulated.
        self.energies=np.array(simulated_energies)
        #to extract the pads data
        self.pads_times=pads_data[0]
        self.pads_angbeds=pads_data[1]
        self.pads = pads_data[2]
        #extracting the wind waves data
        self.waves_times=waves_data[0]
        self.waves_freq_rad1=waves_data[1]
        self.waves_volts_rad1=waves_data[2]
        self.waves_freq_rad2=waves_data[3]
        self.waves_volts_rad2=waves_data[4]
        


#class that will hold wind_event charecteristics, i,e, t_e,t_o,HWHM+,HWHM-,
#class wind_events_chars(self,omni_time_data,omni_HWHM_data,)