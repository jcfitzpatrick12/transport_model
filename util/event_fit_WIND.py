# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:08:44 2022

@author: jcfit

in this script we will

-create a class that allows for a fitting of the event based on all the saved event charecteristics and corre-
-sponding simulation charecteristics

TASKS

-set up so to involve as an option before plotting in the UI
-load in the event charecteristics for the chosen event
-load in the simulated charecteristics for ALL the saved kappa values (we )
-build an array that keeps track of which indexed tuple of charecteristics corresponds to which simulation


THESE FUNCTIONS ARE QUITE PARTICULAR THAT THERE ARE THE SAME SHAPE OF TRIALLED SIMULATIONS FOR EACH KAPPA!
"""

import numpy as np
import os
import matplotlib.pyplot as plt

#import warnings
#warnings.simplefilter("error", np.VisibleDeprecationWarning)

'''
adding the project to the system path so to allow for importing transport scripts
importing the global functions file
'''

import sys
from pathlib import Path 
#importing the global functions file
import util.global_functions as funcs
import os

from datetime import datetime,timedelta


class fit_WIND_event:
    
    def __init__(self,dicts):
        self.WIND_dict=dicts['WIND_dict']
        self.fit_chars = self.WIND_dict['fit_chars']
        self.break_energy = self.WIND_dict['break_energy[keV]']
        #self.WIND_energies=np.array([27,40,67,110,180,310,520])
        self.fit_energies=self.WIND_dict['fit_energies[keV]']
        self.data_path = funcs.folder_functions().return_datapath()
        self.sim_dict=dicts['sim_dict']
        self.energies = self.sim_dict['energies[keV]']
        
        
        
    #this function will 
    def build_sim_chars_over_kappa(self,):
        '''
        important! we assume here each file over kappa contains the same numpy shapes of data
        we are also making assumptions on the order that things are saved in.... check!
        '''
        #create an array that contains all file names in a folder
        data_names=os.listdir(self.data_path)
        #loop through each file name
        #picks out the sim_omni CHARECTERISTICS for each simulation (t_o,t_d,HWHM_o,HWHM_d)
        keyword='sim_omni_chars_data_'
        #create a big array that will hold the 
        sim_chars_data_overkappa = []
        for file in data_names:
            #seperating out into chars files and var files
            if keyword in file:
                print(file)
                sim_chars_data = np.load(os.path.join(self.data_path,file))
                sim_chars_data_overkappa.append(sim_chars_data)
                
        try:
            sim_chars_data_overkappa=np.array(sim_chars_data_overkappa)
        
        except:
            #deprecation needs to be upgraded to fatal error for this to work
            raise SystemError('Check simulation dimensions for each kappa! They should be equal!')

        return sim_chars_data_overkappa
    
    #function that takes in the requested chars to be fitted, and outputs the corresponding array indices that they lie at
    def char_to_index(self):
        #create a dictionary that holds the corresponding index for each charecter
        ind_dict={}
        ind_dict['t_peak[s]']=0
        ind_dict['t_r[s]']=1
        ind_dict['t_d[s]']=2
        
        #prepare an array to hold the indices
        inds = []
        #for each requested charecteristics
        for char in self.fit_chars:
            #extract teh index
            ind = ind_dict[char]
            #append to the index array
            inds.append(ind)
        #return that index array
        return inds
    
    def energy_to_index_old(self):
        #create an array with the index of each energy
        inds = np.arange(0,len(self.WIND_energies),1)
        #array with true if the energy is above the threshold
        above_thresh = self.WIND_energies>self.break_energy
        #return the inds of these energies
        inds=inds[above_thresh]
        return inds
    
        
    def energy_to_index(self):
        #create a dictionary that holds the corresponding index for each energy
        ind_dict={}
        for n in range(len(self.energies)):
            energy=self.energies[n]
            ind_dict[str(energy)]=n
        
        #prepare an array to hold the indices
        inds = []
        #for each requested charecteristics
        for energy in self.fit_energies:
            #extract teh index
            ind = ind_dict[str(energy)]
            #append to the index array
            inds.append(ind)
        #return that index array
        return inds
    
    def min_MSE(self,event_chars,sim_chars_data):
        
        #make sure that we have the number of required energies for the simulations and WIND
        n_ener_sims = np.shape(sim_chars_data[...,0,0])[-1]
        n_ener_wind = len(event_chars[...,0])
        if n_ener_sims!=n_ener_wind:
            raise SystemError('Need the same number of simulated energies as WIND energies to fit!')
            
        #decompose into sim_chars and sim_vars as defined on construction
        sim_vars = sim_chars_data[...,0]
        sim_chars = sim_chars_data[...,1]
    
        #extract the equivalent array indices
        char_inds = self.char_to_index()
        
        #slicing the desired charecteristics
        sim_chars=sim_chars[...,char_inds]
        event_chars=event_chars[...,char_inds]
        
        
        #return the indices of the energies above the spectral break
        energy_inds = self.energy_to_index() 
        
        #reassign n_ener_wind to the number of energies above the threshold
        n_ener_wind = len(energy_inds)
        
        

        
        #slicing out the desired energies
        sim_chars=sim_chars[...,energy_inds,:]
        event_chars=event_chars[energy_inds,...]
        
        N=len(self.fit_chars)
        #flatten event_chars over energy and over char tuple
        event_chars = np.reshape(event_chars,(n_ener_wind*N))
        #same for each simulation
        sim_reshape_tup = np.shape(sim_chars)[0:3]+(n_ener_wind*N,)
        sim_chars = np.reshape(sim_chars,sim_reshape_tup)


        square_diff = (sim_chars-event_chars)**2
        
        
        #finding the mean square errors
        MSEs=np.sum(square_diff,axis=-1)
        #print(np.shape(square_diff))
        #find the minimum mean square error
        min_MSE = np.min(MSEs)
        
        where_min=(MSEs==min_MSE)
        
        best_fit_params = sim_vars[where_min][0,0]
        print('The best fit to the time profiles for the charecteristics '+str(self.fit_chars)+' is:')
        print('kappa='+str(best_fit_params[0]))
        print('alpha='+str(best_fit_params[1]))
        print('mfp-='+str(best_fit_params[2]))
        print('mfp+='+str(best_fit_params[3]))
        
        return 
        
        
    def fit_event(self):
        #loading in the event data
        #saving the event_chars_dict!
        #done!   
        #saving both the dictionary and numpy array for convenience   
        event_chars = np.load(os.path.join(self.data_path,"event_chars.npy"))
        #order is t_0,t_d,HWHM_o,HWHM_d
        
        #build the overarching numpy array that contains all the simulated event_charecteristics over kappa 
        #(since these are in seperate files)
        sim_chars_data = self.build_sim_chars_over_kappa()
           

        
        #now we need to fit the tuples sim_chars to event_chars.
        #i.e. find the best tuple [kappa,aplpha,pair] that best describe the data!
        #what statistical method...
        
        '''
        Try minimise the mean square error! simple, how on earth to decide the degrees of freedom? 
        '''
        
        self.min_MSE(event_chars,sim_chars_data)
        return
    

