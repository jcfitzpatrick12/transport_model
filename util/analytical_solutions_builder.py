# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:44:53 2022

@author: jcfit

This script will
"""

import numpy as np
import os
from scipy import special

#import warnings
#warnings.simplefilter("error", np.VisibleDeprecationWarning)

    
#importing the global functions file
import util.global_functions as funcs

#from util.flexi_plotter import plot_functions


class solution_functions:
    #takes in as input the sim_dict and sort_dict and the sampled times to return the z solutions at 
    #also takes in z_beds and t_bins_minutes to extract the ranges over which to build the solutions
    def __init__(self,dicts):
        #loading in the dictionary
        self.plot_dict=dicts['plot_dict']
        self.sim_dict=dicts['sim_dict']
        self.sort_dict=dicts['sort_dict']
        
        #loading in which analytical solution to plot and the needed variables to implement
        self.analytical_toplot=self.plot_dict['analytical_toplot']
        self.energy_toplot=self.plot_dict['energy_toplot[keV]']
        self.mfp_toplot=self.plot_dict['mfp_toplot[AU]']
        #working out the numerated energy to plot (needed to extract the velocity)
        self.ee=np.array(self.sim_dict['energies[keV]'])
        self.n_en = int(np.where(self.energy_toplot==self.ee)[0][0])
        #self.n_en=1
        
        '''
        HOW TO BYPASS IF PLOTTING SIMULATION PARAMETERS THAT ARE NOT IN WORKING MEMORY IN SIM_DICT? CONFLICT!
        '''
        
        #loading in extra values to locate the necessary sampled_ts, z_beds and t_bins_minutes used for that simulation
        self.kappa=self.plot_dict['kappa_toplot']
        self.alpha_toplot=self.plot_dict['alpha_toplot']
        #defining the file name
        self.fname= funcs.file_name_asym(self.alpha_toplot,self.mfp_toplot[0],self.mfp_toplot[1],self.kappa)
        
        #loading in z_obs to build the analytical time profile
        self.z_obs = self.sort_dict['z_observation[AU]']
        #loading in z_init to build the analytical_time profile
        self.z_init=self.sim_dict['z_injection[AU]']
    
    #function takes in files from the file dictionary defined within to plot
    def load_files(self,item):
        file_dict={}
        file_dict['analytical']=['sampled_ts_','z_beds_','t_bins_minutes_']
        #general path to the data
        data_path=funcs.folder_functions().return_datapath()
        #request 'item' named files 
        files=file_dict[item]
        #empty array to pack the data to plot into
        data_toplot=[]
        for file in files:
            #which file to load
            #print(file+self.fname)
            str_toload=file+self.fname+".npy"
            
            #numpy load it and append to data to plot (can be very different datatype!)
            data_toplot.append(np.load(os.path.join(data_path,str_toload)))
        return data_toplot
    
    #build the spatial diffusion coefficient (for [S]ymmetric[D]iffusion[S]olution
    #returns a vectorised D_zz of the form (n_mfp,n_ener,num_tar)
    def Dzz_SDII(self,velocity_toplot):
        #take the forwards mean free path
        if self.mfp_toplot[0]!=self.mfp_toplot[1]:
            raise SystemError('Need symmetric mean free paths.')
        return velocity_toplot*self.mfp_toplot[1]/3
    
    #identical to Dzz_SDII
    def Dzz_SDCI(self,velocity_toplot):
        if self.mfp_toplot[0]!=self.mfp_toplot[1]:
            raise SystemError('Need symmetric mean free paths.')
        return velocity_toplot*self.mfp_toplot[1]/3
        
    #Francescos derivated spatial diffusion coefficient
    def Dzz_FRAN(self,velocity_toplot):
        return (self.mfp_toplot[1]*velocity_toplot)*0.0853*(7/54)
    
    #[sol]ution_[S]ymmetric[D]iffusion[I]nstantaneous[I]njection
    def sol_SDII(self,z,t,velocity_toplot):
        Dzz=self.Dzz_SDII(velocity_toplot)
        return (1/np.sqrt(4*np.pi*Dzz*t))*np.exp(-np.power(z-self.z_init,2)/(4*Dzz*t))
    
    #[sol]ution_[S]ymmetric[D]iffusion[C]onstant[I]njection
    def sol_SDCI(self,z,t,velocity_toplot):
        Dzz=self.Dzz_SDCI(velocity_toplot)
        z_bar = z-self.z_init
        return np.sqrt(Dzz*t/np.pi)*(np.exp((-(z_bar**2)/(4*Dzz*t))))+0.5*(z_bar*special.erf(z_bar/(np.sqrt(4*Dzz*t)))-np.abs(z_bar))
    
    #computing the analytical solutions in the case of strong scattering in the backward direction
    def sol_FRAN(self,z,t,velocity_toplot):
        Dzz=self.Dzz_FRAN(velocity_toplot)
        z_bar=(z-self.z_init)-(velocity_toplot*t)/2
        a=1/(np.sqrt(4*np.pi*Dzz*t))
        b=np.exp((-1*np.power(z_bar,2))/(4*Dzz*t))
        return a*b

    
    #builds an analytical solution based on the general function general_func(z,t)
    def build_solution_t(self,sol_func,tar,velocity_toplot):
        sol_t = sol_func(self.z_obs,tar,velocity_toplot)
        return sol_t/np.nanmax(sol_t)
    
    def build_solution_z(self,sol_func,zar,sampled_ts,velocity_toplot):
        #extracting the sample times
        #recall, sampled_ts are in days!
        ts=sampled_ts[self.n_en]
        #inferring the number of samples from the sample times
        M = len(ts)
        #initalise an array that will hold the z_solutions
        sol_z = np.empty((M,len(zar)))
        for i in range(M):
            t=ts[i]
            #evaluateing the z solution for that time sample
            sol_nonorm = sol_func(zar,t,velocity_toplot)
            sol_z[i]=sol_nonorm/np.trapz(sol_nonorm,zar)
        #returning the solutions at each sample
        return sol_z
    
    #need the min and max z and t values
    #this function builds the requested analytical solution based on energy requested, mean free path requested and the string required
    def build_solution(self,):
        #first load in the carry_over array
        carry_over=funcs.load_carry_over()
        #load in the analytical files (sampled_ts_,z_beds,t_bins_minutes)
        analytical_data = self.load_files('analytical')
        #loading the sample times
        sampled_ts=analytical_data[0]
        
        sampled_ts=np.array(sampled_ts)
        #loading the z_beds
        z_beds = analytical_data[1]
        #loading in the time array
        t_bins_minutes=analytical_data[2]
        
        #placing more grid points for the analytical solutions (time array and distance array)
        zar=np.linspace(z_beds[0],z_beds[-1],1000)
        #conveting tar to days
        tar=np.linspace(t_bins_minutes[0],t_bins_minutes[-1],1000)/1440
             
        #exctracting the velocity to plot
        velocity_toplot = v=carry_over['velocities[AU/day]'][self.n_en]
        #building the solutions_funcs class 
    
        if self.analytical_toplot=='SDII':
            #if the solution chosen is symmetric diffusion instantaneous injection, use the corresponding spatial diffusion coefficient and corresponding solution function
            sol_func = self.sol_SDII
        if self.analytical_toplot=='SDCI':
            #if the solution chosen is symmetric diffusion constant injection, use the corresponding spatial diffusion coefficient and corresponding solution function
            sol_func = self.sol_SDCI
        
        if self.analytical_toplot=='FRAN':
            #if we are considering Francescos derived asymmetric solution
            sol_func = self.sol_FRAN
            
            
        #building the solutions, given the requested analytical solution
        #technically for SDII and SDCI velocity is redundant, however, to keep code neat for now pass into the function
        #first the t profile at z=z_obs
        sol_t= self.build_solution_t(sol_func,tar,velocity_toplot)
        #now build the z solutions (which will be a n_samples,n_z size array
        sol_z=self.build_solution_z(sol_func,zar,sampled_ts,velocity_toplot)
        #output solutions are already normalised to the peak intensity
        #reconvert tar back to days
        tar*=1440
        #return the analytical time (convert back to days)
        return [tar,sol_t,zar,sol_z]



        
                           
                           
                           
                           
                           
                           
                           
                           
                           
                           
                           