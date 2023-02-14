# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:35:03 2022

@author: jcfit
"""

'''
adding the project to the system path so to allow for importing transport scripts
importing the global functions file
'''

import sys
from pathlib import Path
    
#importing the global functions file
import util.global_functions as funcs
from util.global_functions import folder_functions as ffuncs

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib as mpl

import numpy as np
import os

import matplotlib.dates as mdates

from datetime import timedelta
import warnings




#plotting_variables_labels=['mfp_toplot[AU]','alpha_toplot','energy_toplot[keV]','step_toplot','plots_toplot']


#class for figure formatting and saving
class plot_functions:
    def __init__(self,dicts):
        #define the path to util as in global functions
        self.path_util = os.path.dirname(os.path.realpath(__file__))
        
        #global figure parameters
        self.fsize_head=15
        self.fsize_ticks=10
        self.cmap1=plt.get_cmap("Set1")
        self.props=dict(boxstyle='round', facecolor='white', alpha=0.1)
        
        #extracting the relevent dictionaries
        self.plot_dict=dicts['plot_dict']
        self.sort_dict=dicts['sort_dict']
        self.sim_dict=dicts['sim_dict']
        self.WIND_dict=dicts['WIND_dict']
        
        #extracting whether the mean free path used was constant
        self.mfp_const=self.sim_dict['mfp_constant']
        #used in finding the numerated requested energy
        self.ee=np.array(self.sim_dict['energies[keV]'])
        
        
        #extracting values from the plotting dictionary (to be formatted into strings)
        self.alpha_toplot=self.plot_dict['alpha_toplot']
        self.mfp_toplot=self.plot_dict['mfp_toplot[AU]']
        self.energy_toplot=self.plot_dict['energy_toplot[keV]']
        self.kappa=self.plot_dict['kappa_toplot']
        self.step_toplot=self.plot_dict['step_toplot']
        self.n_en = int(np.where(self.energy_toplot==self.ee)[0][0])
        #self.n_en=1
        #extracting which plots have been requested
        self.plots_toplot=self.plot_dict['plots_toplot']
        #defining the file_name (unique for the given request!)
        self.fname= funcs.file_name_asym(self.alpha_toplot,self.mfp_toplot[0],self.mfp_toplot[1],self.kappa)
        self.analytical_toplot=self.plot_dict['analytical_toplot']
        self.plot_analytical=self.plot_dict['plot_analytical']
        self.char_toplot=self.plot_dict['char_toplot']
        self.energy_char_toplot = self.plot_dict['energy_char_toplot']
        
        #extracting the sorting variables to plot
        self.z_obs = self.sort_dict['z_observation[AU]']
        self.z_tol=self.sort_dict['z_tolerance[AU]']
        self.M=self.sort_dict['num_samples']
        self.t_binwidth=self.sort_dict['t_binwidth[s]']
        

        self.event_energy=self.WIND_dict['event_energy_toplot[keV]']
        self.break_energy = self.WIND_dict['break_energy[keV]']
        
        #load in the data path
        self.data_path = funcs.folder_functions().return_datapath()
        

        
        
    
    #function takes in files from the file dictionary defined within to plot
    def load_files_toplot(self,item):
        #file_dict will hold which files are necessary to load for each figure
        file_dict={}
        file_dict['injection']=['t_inj_','inj_function_']
        file_dict['diffusion coef']=['mu_toplot_','D_mumu_toplot_']
        file_dict['zmu']=['z_normbeds_','mu_beds_','hist2D_']
        file_dict['z and mu']=['mu_beds_','h_mu_','ers_mu_','z_beds_','h_z_','ers_z_']
        file_dict['electron flux']=['t_bins_minutes_','omni_counts_','omni_counts_ers_','mu_beds_pads_','pads_','bal_arrival_times_']
        file_dict['plot_WIND_one_energy_comparison']=['t_bins_minutes_','omni_counts_','omni_counts_ers_','ang_beds_pads_','pads_','bal_arrival_times_','t_bins_inj_']
        file_dict['sampled_times[days]'] = ['sampled_ts_']
        file_dict['plot_WIND_all_energy_comparison']=['t_bins_minutes_','omni_counts_','omni_counts_ers_','bal_arrival_times_','sim_omni_chars_','t_bins_inj_']
        file_dict['plot_one_omni_characteristic'] = ['sim_omni_chars_']
        file_dict['plot_all_simulations_one_energy']= ['t_bins_inj_','bal_arrival_times_']
        #general path to the data
        data_path=funcs.folder_functions().return_datapath()
        #request 'item' named files 
        files=file_dict[item]
        #empty array to pack the data to plot into
        data_toplot=[]
        for file in files:
            #which file to load
            str_toload=file+self.fname+".npy"
            #numpy load it and append to data to plot (can be very different datatype!)
            data_toplot.append(np.load(os.path.join(data_path,str_toload),allow_pickle=True))
        return data_toplot

    #creates a dictionary strings for labelling some figures using requested parameters
    def build_string_dictionary(self):
        #finding the sample time in minutes and converting to a string
        sampled_ts_toplot=self.load_files_toplot('sampled_times[days]')[0]
        sampled_ts_toplot=np.array(sampled_ts_toplot)
        
        t_samp = sampled_ts_toplot[self.n_en,self.step_toplot]*1440
        t_samp=np.round(t_samp,2)
        
        #building more strings for plotting
        str_mfp_n = "$\lambda_{\parallel, \oplus}^{-}$"+'='+str(self.mfp_toplot[0])+' [AU]'
        str_mfp_p = "$\lambda_{\parallel, \oplus}^{+}$"+'='+str(self.mfp_toplot[1])+' [AU]'
        str_alpha = "$\\alpha$"+'='+str(self.alpha_toplot)
        str_energy = 'E='+str(self.energy_toplot)+str(' [keV]')
        str_t = "$t_{s}$="+str(t_samp)+" [minutes] "
        str_z_obs = "$z_{obs}$="+str(self.z_obs)+" [AU] "
        str_kappa="$\kappa=$"+str(self.kappa)
        #str_z_tol = "$z_{tol}$="+str(z_tol_toplot)+" [AU] "

        if self.mfp_const==True:
            str_mfp_formula = r'$\lambda_{\parallel}(z,p)=\lambda_{\parallel, \oplus}$ (constant)'
        if self.mfp_const==False:
            str_mfp_formula = r'$\lambda_{\parallel}(z,p)=\lambda_{\parallel, \oplus} \left( \frac{z}{z_{\oplus}} \right)^{\kappa} \left( \frac{p}{p_{0}} \right)^{2\alpha}$'
            
        strings_toplot=[str_mfp_n,str_mfp_p,str_alpha,str_energy,str_t,str_z_obs,str_mfp_formula,str_kappa]
        strings_toplot_labels = ['mfp-_toplot[AU]','mfp+_toplot[AU]','alpha_toplot','energy_toplot[keV]','t_samp[minutes]','z_observation[AU]','mfp_formula','kappa']
        strings_dict=funcs.build_dict(strings_toplot_labels,strings_toplot)
        return strings_dict
        
    
    def plot_injection(self,plot,string_dict,folder_path):
        
        data_toplot=self.load_files_toplot(plot)
        t_inj=data_toplot[0][self.n_en][:-1]*1440
        inj_function=data_toplot[1][self.n_en][:-1]
    
        
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,5))
        ax.set_ylabel('Injection Rate',fontsize=self.fsize_head)
        ax.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax.set_xlabel('Time Elapsed [hrs]',fontsize=self.fsize_head)
        
        ax.plot(t_inj,inj_function,color=self.cmap1(0))
        
        str_energy=string_dict['energy_toplot[keV]']
        plt.show(block=False)
        plt.savefig(os.path.join(folder_path,'fig+injection_function_'+str_energy+'.pdf'),format='pdf',bbox_inches='tight')
        
    def plot_diffusion_coef(self,plot,string_dict,folder_path):   
        data_toplot=self.load_files_toplot(plot)
        mus=data_toplot[0]
        D_mumus=data_toplot[1]
        
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,5))     
        ax.set_ylabel('$D_{\mu \mu}(\mu)$ / Speed [AU/day]',fontsize=self.fsize_head)
        ax.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax.axvline(x=0,linestyle='dashed',color='grey')
        ax.set_xlabel('Pitch Angle Cosine $\mu=cos(\\theta)$',fontsize=self.fsize_head)
        ax.plot(mus,D_mumus,color=self.cmap1(0))
        plt.show(block=False)
        plt.savefig(os.path.join(folder_path,'fig+diffusion_coefficient_.pdf'),format='pdf',bbox_inches='tight')
        
    def plot_zmu(self,plot,string_dict,folder_path):
        
        data_toplot=self.load_files_toplot(plot)        
        z_normbeds=data_toplot[0]
        mu_beds=data_toplot[1]
        H2D_toplot=data_toplot[2][self.n_en,self.step_toplot]
        
        #fix the normalisation!
        fig,ax=plt.subplots(1,figsize=(10,5))
        ax.set_xlabel('z/($v_{E}t_{s}}$)',fontsize=self.fsize_head)
        ax.set_ylabel('Pitch Angle Cosine $\mu=\cos(\\theta)$',fontsize=self.fsize_head)
        ax.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax.axhline(y=0,color='white',linestyle='--')
        im=ax.pcolormesh(z_normbeds, mu_beds, H2D_toplot.T, cmap='viridis',vmin=0,vmax=np.max(H2D_toplot))
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=self.fsize_ticks)
        
        
        
        str_t=string_dict['t_samp[minutes]']
        ax.set_title(str_t,loc='right',fontsize=self.fsize_head)
        
        str_energy=string_dict['energy_toplot[keV]']

        plt.show(block=False)
        plt.savefig(os.path.join(folder_path,'fig+zmu_distributions_'+str_energy+'.pdf'),format='pdf',bbox_inches='tight')
        
    def plot_z_and_mu(self,plot,string_dict,folder_path,sols):

        #extracting the specific times for the given energy
        sampled_ts=self.load_files_toplot('sampled_times[days]')[0]
        #extracting the sampled times of a given energy
        sampled_ts_toplot=sampled_ts[self.n_en]
        
        data_toplot=self.load_files_toplot(plot)
        mu_beds=data_toplot[0]
        h_mu=data_toplot[1][self.n_en]
        ers_mu=data_toplot[2][self.n_en]
        z_beds=data_toplot[3]
        h_z=data_toplot[4][self.n_en]
        ers_z=data_toplot[5][self.n_en]
    
        #creating the error bounds in mu and z
        mu_ers_n = h_mu-ers_mu
        mu_ers_p = h_mu+ers_mu
        z_ers_n = h_z-ers_z
        z_ers_p=h_z+ers_z
    
                
        # setup the normalization and the colormap
        normalize = mcolors.Normalize(vmin=sampled_ts_toplot.min(), vmax=sampled_ts_toplot.max()*1.5)
        colormap = cm.plasma
        
        #creating the Figure
        fig1=plt.figure(figsize=(10,10))
        #ax0 will hold the mu values
        ax0 = plt.subplot(2,1,1) 
        #ax1 will hold the t values
        ax1 = plt.subplot(2,1,2)
   
        
        ax0.set_ylabel('Normalised Counts',fontsize=self.fsize_head)
        ax0.set_xlabel('Pitch Angle Cosine $\mu = \cos (\\theta)$',fontsize=self.fsize_head)
        ax0.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax0.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax0.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        
        for samp in range(1,self.M):
            t=sampled_ts_toplot[samp]
            ax0.stairs(h_mu[samp],mu_beds,color=colormap(normalize(t)))
            #using post edges for the highlighting so need to repeat the last entry!
            #confirmed with stairs, so tranitively confirmed with this highlighting method
            mu_ers_n_samp=np.insert(mu_ers_n[samp],len(mu_ers_n[samp]),mu_ers_n[samp][-1])
            mu_ers_p_samp=np.insert(mu_ers_p[samp],len(mu_ers_p[samp]),mu_ers_p[samp][-1])
            #mu_beds=np.insert(mu_beds,len(mu_beds),mu_beds[-1])
            ax0.fill_between(mu_beds, mu_ers_n_samp, mu_ers_p_samp,step='post', color='k', alpha=0.15)
            
        '''
        and now plotting the z distributions
        '''
        
        ax1.set_ylabel('Normalised Counts',fontsize=self.fsize_head)
        ax1.set_xlabel('Distance [AU]',fontsize=self.fsize_head)
        ax1.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax1.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax1.axvspan(xmin=self.z_obs-self.z_tol,xmax=self.z_obs+self.z_tol,color='lightpink')
        
        if self.plot_analytical==True:
            #extracting the analytical_solutions
            zar=sols[2]
            sol_z=sols[3]
            for samp in range(1,self.M):
                ax1.plot(zar,sol_z[samp],color='grey')
            
            
        #creating the annotations on site, const is to scale the jumps to be plotted well
        const=0.2
        #size of jumps to make
        size_jumps = int(np.round((len(z_beds)/self.M)))
        z_loc_max = z_beds[::size_jumps]/const
    
            
        for samp in range(1,self.M):
            #pick the 4th largest bin to plot
            z_max = sorted(h_z[samp])[-5]
            t=sampled_ts_toplot[samp]
            ax1.stairs(h_z[samp],z_beds,color=colormap(normalize(t)))
            z_ers_n_samp=np.insert(z_ers_n[samp],len(z_ers_n[samp]),z_ers_n[samp][-1])
            z_ers_p_samp=np.insert(z_ers_p[samp],len(z_ers_p[samp]),z_ers_p[samp][-1])
            ax1.fill_between(z_beds, z_ers_n_samp, z_ers_p_samp,step='post', color='k', alpha=0.15)
            #ax1.annotate(str_t,(z_loc_max[samp]+0.1,z_max),fontsize=15,color=colormap(normalize(t)))
            str_t = "t="+'{:.2f}'.format(t*1440)+" [mins] "
            #ax1.annotate(str_t,(0.1,np.max(h_z[1])-np.max(h_z[1])*samp/self.M),fontsize=15,color=colormap(normalize(t)))
            #ax1.annotate(str_t,(0.4,z_max-const*samp*z_max),fontsize=15,color=colormap(normalize(t)))
            ax1.annotate(str_t,(0.15,0.45-samp*0.15/self.M),fontsize=15,color=colormap(normalize(t)),xycoords='figure fraction')
            
        
        str_energy=string_dict['energy_toplot[keV]']
        str_mfp_n=string_dict['mfp-_toplot[AU]']
        str_mfp_p=string_dict['mfp+_toplot[AU]']
        str_alpha=string_dict['alpha_toplot']
        str_kappa=string_dict['kappa']

        str_ar = [str_energy,str_mfp_n,str_mfp_p,str_alpha,str_kappa]
        

        if self.mfp_const==False:
            for i,str in enumerate(str_ar):
                ax1.annotate(str,(0.15,0.85-i*0.03),fontsize=15,xycoords='figure fraction')
        if self.mfp_const==True:
            for i,str in enumerate(str_ar):
                if i<=2:
                    ax1.annotate(str,(0.15,0.85-i*0.03),fontsize=15,xycoords='figure fraction')

                
        #ax1.set_xlim(0,0.25)

        plt.show(block=False)
        plt.savefig(os.path.join(folder_path,'fig+z&mu_distributions_'+str_energy+'.pdf'),format='pdf',bbox_inches='tight')

    def plot_electron_flux(self,plot,string_dict,folder_path,sols):
        
        data_toplot=self.load_files_toplot(plot)
        t_bins_minutes=data_toplot[0]
        omni_counts = data_toplot[1][self.n_en]
        omni_counts_ers=data_toplot[2][self.n_en]
        mu_beds_pads=data_toplot[3]
        #print(mu_beds_pads)
        pads=data_toplot[4][self.n_en]
        bal_arrival_time=data_toplot[5][self.n_en]
        #sim_omni_chars = data_toplot[6][self.n_en]
        #t_bins_inj=data_toplot[7]

        
        
        #creating the error bounds in mu and z
        omni_ers_n = omni_counts-omni_counts_ers
        omni_ers_p = omni_counts+omni_counts_ers
        
        
        fig = plt.figure(constrained_layout=True,figsize=(15,9))

        gs = GridSpec(2, 2, figure=fig,width_ratios=[1,0.14], height_ratios=[1,1],)
        ax1 = fig.add_subplot(gs[0, 0])
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
        ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
        ax3= fig.add_subplot(gs[0,1])
        ax3.axis('off')
        ax4 = fig.add_subplot(gs[1,1])
        ax4.axis('off')
        
        '''
        ax1 is for the pads, ax3 is to hold the colorbar
        '''
               
        ax1.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax1.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        #ax1.set_xlabel('Time Since Injection [Minutes]',fontsize=fsize_head)
        ax1.tick_params(labelbottom=False)
        ax1.set_ylabel('Pitch Angle Cosine $\mu=cos(\Theta)$',fontsize=self.fsize_head)
        str_mfp_formula=string_dict['mfp_formula']
        ax1.set_title(str_mfp_formula,loc='right',fontsize=20,pad=20)
        ax1.axvline(x=bal_arrival_time,color='grey',linestyle='--')
        #need to repeat the final entry of t_bins since pcolormesh demands edges have +1 dimension to image values (verified correct in this case)
        #t_bins_minutes=np.insert(t_bins_minutes,len(t_bins_minutes),t_bins_minutes[-1])
        im=ax1.pcolormesh(t_bins_minutes,mu_beds_pads,pads.T,cmap='rainbow')
        #ax1.imshow((t_bins_minutes,mu_beds_pads),pads.T)
        cbar = fig.colorbar(im, ax=ax3,location='left')
        cbar.ax.tick_params(labelsize=self.fsize_ticks)

        
        '''
        ax2 is for the omnidirectional intensities, ax4 is for plotting the parameters
        '''
        
        ax2.set_xlabel('Time Since Injection [Minutes]',fontsize=self.fsize_head)
        ax2.set_ylabel('Peak Normalised Flux',fontsize=self.fsize_head)
        ax2.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax2.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax2.axvline(x=bal_arrival_time,color='grey',linestyle='--')

        
        ax2.stairs(omni_counts,t_bins_minutes)
        #inserting a repeated last entry for pltoting purposes, so the plot knows to extend the last value
        omni_ers_n=np.insert(omni_ers_n,len(omni_ers_n),omni_ers_n[-1])
        omni_ers_p=np.insert(omni_ers_p,len(omni_ers_p),omni_ers_p[-1])
        ax2.fill_between(t_bins_minutes, omni_ers_n, omni_ers_p,step='post', color='k', alpha=0.15)

        
        if self.plot_analytical==True:
            #extracting the required analytical_solutions to overlay
            tar=sols[0]
            sol_t=sols[1]
            ax2.plot(tar,sol_t,color='black')
        
        str_energy=string_dict['energy_toplot[keV]']
        str_mfp_n=string_dict['mfp-_toplot[AU]']
        str_mfp_p=string_dict['mfp+_toplot[AU]']
        str_alpha=string_dict['alpha_toplot']
        str_z_obs=string_dict['z_observation[AU]']
        str_kappa=string_dict['kappa']
        
        props = self.props

        
        strs = [str_energy,str_mfp_n,str_mfp_p,str_z_obs,str_alpha,str_kappa]
        
        for i in range(0,4):
            string=strs[i]
            ax2.annotate(string,(0.6,0.45-i*0.06),fontsize=15,xycoords='figure fraction')
        
        if self.mfp_const==False:
            for i in range(4,6):
                string=strs[i]
                ax2.annotate(string,(0.6,0.45-i*0.06),fontsize=15,xycoords='figure fraction')
        
        plt.show(block=False)
        plt.savefig(os.path.join(folder_path,'fig+electron_fluxes_'+str_energy+'.pdf'),format='pdf',bbox_inches='tight')
        
        
    #creates a folder to save the required figures, and information about the simulation.
    def create_folder(self):
        #the folder_name convention
        str_folder_name_info='mfp-,mfp+,alpha,kappa'
        root_folder = 'transport_simulation_figures'
        #extracting the specific files MAIN defining parameters, alpha and L
        #create the folder path based on this simulation name
        path_root=os.path.join(self.path_util,'transport_simulation_figures')
        #defining the name of the simulation folder
        list_vars=list([self.mfp_toplot[0],self.mfp_toplot[1],self.alpha_toplot,self.kappa])
        str_folder=str(list_vars)
        #build the folder path if it does not already exist
        folder_path = ffuncs().create_simfolder(path_root,str_folder)
        #clear the text file
        ffuncs().clear_text('folder_name_info.txt',folder_path)
        ffuncs().clear_text('simulation_parameters.txt',folder_path)
        ffuncs().clear_text('sorting_parameters.txt',folder_path)
        #creating the simulation variable to append
        #append the variable to the text file
        ffuncs().append_text(str_folder_name_info,'folder_name_info.txt',folder_path)
        #also append the simulation and sorting dictionaries for future reference
        ffuncs().append_text(str(self.sim_dict),'simulation_parameters.txt',folder_path)
        ffuncs().append_text(str(self.sort_dict),'sorting_parameters.txt',folder_path)
        return folder_path
         
    def plot_WIND_waves(self): 
        '''
        adjust so to only use one colorbar (since both are identical!)
        '''
        #first load in the events dictionary
        event_data=funcs.load_events_data()
        #event_data=events_dict[self.event_considering]
        fig = plt.figure(constrained_layout=True,figsize=(15,9))
        #unpacking the WIND/WAVES data for the event
        waves_times=event_data.waves_times
        waves_freq_rad1=event_data.waves_freq_rad1
        waves_freq_rad2=event_data.waves_freq_rad2
        waves_volts_rad1=event_data.waves_volts_rad1
        waves_volts_rad2=event_data.waves_volts_rad2
     
        #set the y label coords
        labelx = -0.1
        #energy to plot
        en=1

        gs = GridSpec(2, 2, figure=fig,width_ratios=[1,0.14], height_ratios=[1,1],)
        #ax1 and ax2 will be the WIND/WAVES data
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
        #ax1a and 1b will hld the colorbar
        ax1a = fig.add_subplot(gs[0,1])
        ax1a.axis('off')
        ax2a=fig.add_subplot(gs[1,1])
        ax2a.axis('off')
        
              
        min_val=10e-1
        max_val=max(np.max(waves_volts_rad1),np.max(waves_volts_rad2))
        
        ax1.set_ylabel(r'Frequency [kHz]',fontsize=self.fsize_head)
        ax1.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax1.tick_params(labelbottom=False)
        ax1.set_yscale('log')
        im=ax1.pcolormesh(waves_times,waves_freq_rad2,waves_volts_rad2.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        cbar = fig.colorbar(im, ax=ax1a,location='right')
        cbar.ax.tick_params(labelsize=self.fsize_ticks)
        cbar.ax.set_ylabel('Norm_Volt_RAD2', fontsize=self.fsize_head)
        
        #ax.set_xscale('log')
       # ax1.annotate('CSV PADS summed', (0.7,0.8),xycoords='axes fraction',fontsize=self.fsize_head)

        ax2.set_ylabel(r'Frequency [kHz]',fontsize=self.fsize_head)
        ax2.set_xlabel(r'Time (Start Time: '+str(waves_times[0])+')',fontsize=self.fsize_head)       
        ax2.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax2.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        myFmt = mdates.DateFormatter('%H%M')
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.set_yscale('log')
        
        #
        im2=ax2.pcolormesh(waves_times,waves_freq_rad1,waves_volts_rad1.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        cbar2 = fig.colorbar(im2, ax=ax2a,location='right')
        cbar2.ax.tick_params(labelsize=self.fsize_ticks)
        cbar2.ax.set_ylabel('Norm_Volt_RAD1', fontsize=self.fsize_head)

        plt.show(block=False)
        plt.subplots_adjust(wspace=0, hspace=0.01)
        
        
    #function that plots the requested simulation data and WIND data at the specified energy
    #as a test to show simultaneously that the t_o and t_d values are correct.
    def plot_WIND_one_energy_comparison(self,plot,string_dict,folder_path):
        
        inj_time=funcs.load_injection_time()
        '''
        loading in all the required data
        '''
        
        #loading in the simulated electron data
        
        data_toplot=self.load_files_toplot(plot)
        t_bins_minutes=data_toplot[0]
        omni_counts=data_toplot[1][self.n_en]
        omni_counts_ers=data_toplot[2][self.n_en]
        ang_beds_pads=data_toplot[3]
        #print(mu_beds_pads)
        pads_sim=data_toplot[4][self.n_en]
        bal_arrival_time=data_toplot[5][self.n_en]
        
        #creating the error bounds in mu and z
        omni_ers_n = omni_counts-omni_counts_ers
        omni_ers_p = omni_counts+omni_counts_ers
        
        #ballistic arrival times (in minutes since injection)
        bal_arrival_time = inj_time+timedelta(minutes=bal_arrival_time)
        
        #loading in the datetimes bins since (radio-derived) injection
        t_bins_inj=data_toplot[6]
        
        #loading in the WIND data (radio and electron data)
        #first load in the events dictionary
        event_data=funcs.load_events_data()
        #unpacking the WIND/WAVES data for the event
        waves_times=event_data.waves_times
        waves_freq_rad1=event_data.waves_freq_rad1
        waves_freq_rad2=event_data.waves_freq_rad2
        waves_volts_rad1=event_data.waves_volts_rad1
        waves_volts_rad2=event_data.waves_volts_rad2
        
        #loading in the pads data
        event_en=np.where(self.event_energy==event_data.energies)[0][0]
        pads_times=event_data.pads_times
        pads_angbeds=event_data.pads_angbeds
        pads_wind=event_data.pads[event_en]
        
        #loading in the WIND omni data
        WIND_omni_times=event_data.omni_times
        WIND_omni_fluxes=event_data.omni_fluxes[event_en]
        WIND_omni_fluxes_smoothed=event_data.omni_fluxes_smoothed[event_en]
        
        #loading in the event time profile charecteristics
        event_omni_chars = funcs.load_event_chars()
        
        '''
        creating the Figures
        '''
        
        #fig = plt.figure(constrained_layout=True,figsize=(25,15))
        fig = plt.figure(figsize=(15,15))
        gs = GridSpec(5, 2, figure=fig,width_ratios=[1,0.14], height_ratios=[1,1,1,1,1],)
        #ax1 and ax2 will be the WIND/WAVES data
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
        #ax3 and ax4 will hold the WIND pads and simulated pads
        ax3 = fig.add_subplot(gs[2,0],sharex=ax1)
        ax4 = fig.add_subplot(gs[3,0],sharex=ax1)
        #ax5 will hold the WIND omnis and simulated omnis
        ax5 = fig.add_subplot(gs[4,0],sharex=ax1)
        #ax1a and 2a will hld the colorbar for the WIND data
        ax1a = fig.add_subplot(gs[0,1])
        ax1a.axis('off')
        ax2a=fig.add_subplot(gs[1,1])
        ax2a.axis('off')
        #ax3a and ax4a will hold the pads colorbars for wind and simulation respectively
        ax3a=fig.add_subplot(gs[2,1])
        ax3a.axis('off')
        ax4a=fig.add_subplot(gs[3,1])
        ax4a.axis('off')
        
        '''
        finding the radio travel time and expected arrival
        '''
        
        radio_travel_time_minutes=funcs.light_travel_time_1AU()
        radio_arrival_time = inj_time+timedelta(minutes=radio_travel_time_minutes)
        
        '''
        plotting the radio data check rad1 vs rad2
        '''
        
        min_val=10e-1
        max_val=max(np.max(waves_volts_rad1),np.max(waves_volts_rad2))
        
        ax1.set_ylabel(r'Frequency [kHz]',fontsize=self.fsize_head)
        labelx = -0.05
        ax1.yaxis.set_label_coords(labelx, 0.0)
        ax1.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax1.tick_params(labelbottom=False)
        ax1.set_yscale('log')
        im=ax1.pcolormesh(waves_times,waves_freq_rad2,waves_volts_rad2.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        #cbar = fig.colorbar(im, ax=ax1a,location='right')
        cbar = fig.colorbar(im, ax=ax1a,)
        cbar.ax.tick_params(labelsize=self.fsize_ticks)
        cbar.ax.set_ylabel('Norm_Volt_RAD1/-2', fontsize=self.fsize_head,x=0.3,y=0.0)
        ax1.tick_params(labelbottom=False)
            
        ax1.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
        ax1.axvline(x=radio_arrival_time,color='yellow',linewidth=1.5,linestyle='dashed')
        
        #ax.set_xscale('log')
       # ax1.annotate('CSV PADS summed', (0.7,0.8),xycoords='axes fraction',fontsize=self.fsize_head

        #ax2.set_xlabel(r'Time (Start Time: '+str(waves_times[0])+')',fontsize=self.fsize_head)       
        ax2.xaxis.set_tick_params(labelsize=self.fsize_head)
        ax2.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        myFmt = mdates.DateFormatter('%H%M')
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.set_yscale('log')
        im2=ax2.pcolormesh(waves_times,waves_freq_rad1,waves_volts_rad1.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        cbar2 = fig.colorbar(im2, ax=ax2a,)
        cbar2.ax.tick_params(labelsize=self.fsize_ticks)
        #cbar2.ax.set_ylabel('Norm_Volt_RAD1', fontsize=self.fsize_head)
        ax2.tick_params(labelbottom=False)
                    
        ax2.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')

        plt.subplots_adjust(wspace=0, hspace=0.075)
        
        ax2.axvline(x=radio_arrival_time,color='yellow',linewidth=1.5,linestyle='dashed')
        
        
        '''
        WIND pads
        '''
        #take absolute value so to take logarithm of the pitch-angle distributions for testing
        pads_wind=np.abs(pads_wind)
        pads_sim = np.abs(pads_sim)
        min_val=10e-4
        max_val=1.0
        pads_sim[pads_sim==0]=min_val
        
        im=ax3.pcolormesh(pads_times,pads_angbeds,pads_wind.T,cmap='rainbow',norm=mcolors.LogNorm(vmin=min_val,vmax=max_val))
        #im=ax3.pcolormesh(pads_times,pads_angbeds,pads_wind.T,cmap='rainbow')
        cbar = fig.colorbar(im, ax=ax3a,location='right')
        cbar.ax.tick_params(labelsize=self.fsize_ticks)
        cbar.ax.set_ylabel('Peak Normed Flux', fontsize=self.fsize_head,y=0.0)
        ax3.yaxis.set_label_coords(labelx, 0.5)
        ax3.set_ylabel('Pitch-Angle $\\theta$',fontsize=self.fsize_head)
        ax3.tick_params(labelsize=self.fsize_ticks)
        ax3.tick_params(labelbottom=False)
        #ax3.axvline(x=bal_arrival_time,color='purple')              
        ax3.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
        ax3.annotate('WIND PAD', (0.8,0.8),xycoords='axes fraction',fontsize=self.fsize_head,color='white')
        ax3.axhline(y=90,linestyle='--',color='grey')


        
        
        ''' 
        Simulated pads
        '''
        
        ax4.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax4.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        #ax1.set_xlabel('Time Since Injection [Minutes]',fontsize=fsize_head)
        ax4.tick_params(labelbottom=False)
        ax4.set_ylabel('Pitch Angle $\\theta$',fontsize=self.fsize_head)
        str_mfp_formula=string_dict['mfp_formula']
        ax1.set_title(str_mfp_formula,loc='right',fontsize=self.fsize_head,pad=20)
        #need to repeat the final entry of t_bins since pcolormesh demands edges have +1 dimension to image values (verified correct in this case)
        #t_bins_minutes=np.insert(t_bins_minutes,len(t_bins_minutes),t_bins_minutes[-1])
        im=ax4.pcolormesh(t_bins_inj,ang_beds_pads[::-1],pads_sim.T,cmap='rainbow',norm=mcolors.LogNorm(vmin=min_val,vmax=max_val))
        #im=ax4.pcolormesh(t_bins_inj,ang_beds_pads[::-1],pads_sim.T,cmap='rainbow')
        #ax1.imshow((t_bins_minutes,mu_beds_pads),pads.T)
        cbar = fig.colorbar(im, ax=ax4a,location='right')
        cbar.ax.tick_params(labelsize=self.fsize_ticks)
        #cbar.ax.set_ylabel('Peak Normed Flux', fontsize=self.fsize_head)
  
        ax4.yaxis.set_label_coords(labelx,.5)
        ax4.axhline(y=90,linestyle='--',color='grey')
        ax4.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
        
        ax4.annotate('Simulated PAD', (0.8,0.8),xycoords='axes fraction',fontsize=self.fsize_head,color='white')
        
        
        '''
        stacked omnidirectional plot
        ''' 
        
    
        #simulated wind omni data + poisson errors
        ax5.stairs(omni_counts,t_bins_inj,color='lightgrey',fill=True,edgecolor='black')
        #inserting a repeated last entry for pltoting purposes, so the plot knows to extend the last value
        omni_ers_n=np.insert(omni_ers_n,len(omni_ers_n),omni_ers_n[-1])
        omni_ers_p=np.insert(omni_ers_p,len(omni_ers_p),omni_ers_p[-1])
        ax5.fill_between(t_bins_inj, omni_ers_n, omni_ers_p,step='post', color='pink', alpha=0.15,edgecolor='black')
        #plot the expected light travel arrive time
        
        
        #ax5.axvline(x=t_o_sim)
        #ax5.axvline(x=t_d_sim)
        #ax5.axhline(y=0.5)
        
        all_WIND_energies = event_data.energies
        try:
            n=np.where(self.energy_toplot==all_WIND_energies)[0][0]
            color_WIND=plt.cm.Dark2(n)
        except:
            raise SystemError('Error choosing color')
        #WIND data
        ax5.step(WIND_omni_times,WIND_omni_fluxes,where='post',label='WIND',color=color_WIND)
        ax5.plot(WIND_omni_times,WIND_omni_fluxes_smoothed,color='grey')

        ax3.axvline(x=bal_arrival_time,color=color_WIND,linestyle='--') 
        ax4.axvline(x=bal_arrival_time,color=color_WIND,linestyle='--') 
        ax5.axvline(x=bal_arrival_time,color=color_WIND,linestyle='--') 
        #plotting the injection time too
        ax4.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
        
 
        '''
        annotating bottom to see simulation data
        '''
        str_energy=string_dict['energy_toplot[keV]']
        str_mfp_n=string_dict['mfp-_toplot[AU]']
        str_mfp_p=string_dict['mfp+_toplot[AU]']
        str_alpha=string_dict['alpha_toplot']
        str_kappa=string_dict['kappa']

        strs = [str_energy,str_mfp_n,str_mfp_p,str_alpha,str_kappa]

        for i,string in enumerate(strs):
            color='black'
            if i==0:
                color=color_WIND
            ax5.annotate(string, (0.78,0.85-0.19*i),xycoords='axes fraction',fontsize=self.fsize_head,color=color)
        
        #ax5.annotate('WIND', (0.8,0.15),xycoords='axes fraction',fontsize=self.fsize_head)
        
        ax5.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax5.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax5.set_xlabel('UT Time ($t_{inj}$:'+str(inj_time)+')',fontsize=self.fsize_head)
        ax5.axvline(x=inj_time,color='green',linewidth=1.5,linestyle='dashed')
        
        #setting the xlim for neat plotting
        ax4.set_xlim(waves_times[0],pads_times[-1])
        
        plt.show(block=False)
        
        plt.savefig(os.path.join(folder_path,'fig+one_energy_comparison_'+str_energy+'.pdf'),format='pdf',bbox_inches='tight')
    

        return []
    
    def plot_WIND_all_energy_comparison(self,plot,string_dict,folder_path):
        #plot radio first two panels then loop over the omnidirectional fluxes rest of the panels!
        inj_time=funcs.load_injection_time()
        '''
        loading in all the required data
        '''
        
        #loading in the simulated electron data
        
        data_toplot=self.load_files_toplot(plot)
        omni_counts=data_toplot[1]
        omni_counts_ers=data_toplot[2]
        bal_arrival_times=data_toplot[3]
        #creating the error bounds in mu and z
        omni_ers_n = omni_counts-omni_counts_ers
        omni_ers_p = omni_counts+omni_counts_ers     
        #ballistic arrival times (in minutes since injection)
        #bal_arrival_time = inj_time+timedelta(minutes=bal_arrival_time)      
        #loading in the simulated time profile charecteristics
        sim_omni_chars = data_toplot[4]
        #loading in the datetimes bins since (radio-derived) injection
        t_bins_inj=data_toplot[5]
        
        
        '''
        as a verification pull out sim_omni_chars/vars
        '''
        sim_omni_vars = sim_omni_chars[...,0]
        sim_omni_chars = sim_omni_chars[...,1]
        
        #loading in the WIND data (radio and electron data)
        #first load in the events dictionary
        event_data=funcs.load_events_data()
        #unpacking the WIND/WAVES data for the event
        waves_times=event_data.waves_times
        waves_freq_rad1=event_data.waves_freq_rad1
        waves_freq_rad2=event_data.waves_freq_rad2
        waves_volts_rad1=event_data.waves_volts_rad1
        waves_volts_rad2=event_data.waves_volts_rad2
        
        #loading in the WIND omni data
        WIND_omni_times=event_data.omni_times
        WIND_omni_fluxes=event_data.omni_fluxes
        
        #loading in the event time profile charecteristics
        event_omni_chars = funcs.load_event_chars()
        

        #loading in the WIND omni data
        WIND_omni_times=event_data.omni_times
        WIND_omni_fluxes=event_data.omni_fluxes
        WIND_omni_fluxes_smoothed = event_data.omni_fluxes_smoothed
        
        #loading in the event time profile charecteristics
        event_omni_chars = funcs.load_event_chars()
        
        n_ener_WIND = len(WIND_omni_fluxes)
        n_ener_sim = np.shape(omni_counts)[0]
        
                
        if n_ener_WIND!=n_ener_sim:
            warnings.warn('Not a full comparison, simulate all energies!')
            raise SystemError('Need all energies to plot!')
            

            
            
        '''
        first two panels, radio data
        last n_ener_sim panels are the omnidirectional data for the simulation and  wind
        include vertical lines for light arrival and expected ballistic arrival for each energy.
        '''
            
        #converting the bal_arrival_times to datetimes since assumed injection
        bal_arrival_times_dt=[]
        for time in bal_arrival_times:
            bal_arrival_times_dt.append(inj_time+timedelta(minutes=time))
            
        #quick and dirty just reuse some code. obtain the time it takes light to travel 1AU in minutes
        radio_travel_time_minutes=funcs.light_travel_time_1AU()
        radio_arrival_time = inj_time+timedelta(minutes=radio_travel_time_minutes)
        
        
        '''
        creating the Figures
        '''
        
        #fig = plt.figure(constrained_layout=True,figsize=(25,15))
        fig = plt.figure(figsize=(10,25))

        #creating the height ratios array
        hrs = []
        for i in range(n_ener_sim+2):
            hrs.append(1)
        gs = GridSpec(2+n_ener_WIND, 2, figure=fig,width_ratios=[1,0.14], height_ratios=hrs,)
        #ax1 and ax2 will be the WIND/WAVES data
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
        
        #adding axis for radio colorbars
        ax1a = fig.add_subplot(gs[0,1])
        ax1a.axis('off')
        ax2a=fig.add_subplot(gs[1,1])
        ax2a.axis('off')
            
            
            
        '''
        plotting the WIND radio data
        '''
        
        min_val=10e-1
        max_val=max(np.max(waves_volts_rad1),np.max(waves_volts_rad2))
        
        ax1.set_ylabel(r'Frequency [kHz]',fontsize=self.fsize_head)
        labelx = -0.05
        ax1.yaxis.set_label_coords(labelx, 0.0)
        ax1.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax1.tick_params(labelbottom=False)
        ax1.set_yscale('log')
        im=ax1.pcolormesh(waves_times,waves_freq_rad2,waves_volts_rad2.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        #cbar = fig.colorbar(im, ax=ax1a,location='right')
        cbar = fig.colorbar(im, ax=ax1a,)
        cbar.ax.set_ylabel(r'Norm_Volt_RAD-1/-2',fontsize=self.fsize_ticks,y=0.0)
        #labelx = -0.05
        #cbar.ax.set_ylabel_coords(labelx, 0.0)
        #cbar.ax.tick_params(labelsize=self.fsize_ticks)
        #cbar.ax.set_ylabel('Norm_Volt_RAD2', fontsize=self.fsize_head)
        ax1.tick_params(labelbottom=False)
        ax1.axvline(x=radio_arrival_time,color='yellow',linewidth=1.5,linestyle='dashed')
        ax1.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
        #ax2.set_xlabel(r'Time (Start Time: '+str(waves_times[0])+')',fontsize=self.fsize_head)       
        ax2.xaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax2.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        myFmt = mdates.DateFormatter('%H%M')
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.set_yscale('log')
        im2=ax2.pcolormesh(waves_times,waves_freq_rad1,waves_volts_rad1.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        cbar2 = fig.colorbar(im2, ax=ax2a,)
        cbar2.ax.tick_params(labelsize=self.fsize_ticks)
        ax2.tick_params(labelbottom=False)
                    
        ax2.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
        ax2.axvline(x=radio_arrival_time,color='yellow',linewidth=1.5,linestyle='dashed')

        #plt.subplots_adjust(wspace=0, hspace=0.05)
        
        '''
        plotting the omniflux data
        '''
        
        #loading in the event chars as a verification  
        event_chars = np.load(os.path.join(self.data_path,"event_chars.npy"))
        
        #loop through each energy
        for n in range(n_ener_WIND):
            '''
            #add in the omni_chars, as validation of (t_o,t_d,HWHM_premax,HWHM_postmax)
            t_o_sim = inj_time+timedelta(seconds=sim_omni_chars[n,0])
            t_d_sim = inj_time+timedelta(seconds=sim_omni_chars[n,1])
            peak = inj_time+timedelta(seconds=sim_omni_chars[n,0])+timedelta(seconds=sim_omni_chars[n,2])
            t_d_sim = inj_time+timedelta(seconds=sim_omni_chars[n,0])+timedelta(seconds=sim_omni_chars[n,2])+timedelta(seconds=sim_omni_chars[n,3])
            
            t_o_event = inj_time+timedelta(seconds=event_chars[n,0])
            t_d_event = inj_time+timedelta(seconds=event_chars[n,1])
            peak_event = inj_time+timedelta(seconds=event_chars[n,0])+timedelta(seconds=event_chars[n,2])
            t_d_event = inj_time+timedelta(seconds=event_chars[n,0])+timedelta(seconds=event_chars[n,2])+timedelta(seconds=event_chars[n,3])
            '''
            
            
            ax = fig.add_subplot(gs[n+2,0],sharex=ax1)
            ax_spare = fig.add_subplot(gs[n+2,1])
            ax_spare.axis('off')
            
            if n==n_ener_WIND-1:
                '''
                annotating bottom to see simulation data
                '''
                str_mfp_n=string_dict['mfp-_toplot[AU]']
                str_mfp_p=string_dict['mfp+_toplot[AU]']
                str_alpha=string_dict['alpha_toplot']
                str_kappa=string_dict['kappa']

                strs = [str_mfp_n,str_mfp_p,str_alpha,str_kappa]

                for i,string in enumerate(strs):
                    ax_spare.annotate(string, (0.1,0.9-0.25*i),xycoords='axes fraction',fontsize=self.fsize_ticks)

            #extracting the counts and energies for a particular energy
            omni_en=omni_counts[n]
            omni_ers_en = omni_counts_ers[n]
            
            #building the error bounds based on poisson counting statistics
            omni_ers_n = omni_en-omni_ers_en
            omni_ers_p = omni_en+omni_ers_en
            #already have the time bins (derived from radio injection)
            t_bins_inj
            
            #extracting the ballistic arrival time for that energy
            bal_arrival=bal_arrival_times_dt[n]
            
            
            #loading in the WIND omni data
            WIND_omni_times=event_data.omni_times
            WIND_omni_fluxes=event_data.omni_fluxes
            #now get at the wind data (for the given energy)
            WIND_omni_en = WIND_omni_fluxes[n]
            WIND_omni_en_smoothed=WIND_omni_fluxes_smoothed[n]
            #already got the times
           #WIND_omni_times
            
            #plot the injection arrival time
            ax.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
            
            #simulated wind omni data + poisson errors
            ax.stairs(omni_en,t_bins_inj,color='lightgrey',fill=True,edgecolor='black')
            #inserting a repeated last entry for pltoting purposes, so the plot knows to extend the last value
            omni_ers_n=np.insert(omni_ers_n,len(omni_ers_n),omni_ers_n[-1])
            omni_ers_p=np.insert(omni_ers_p,len(omni_ers_p),omni_ers_p[-1])
            ax.fill_between(t_bins_inj, omni_ers_n, omni_ers_p,step='post', color='pink', alpha=0.15,edgecolor='black')
            
            #WIND 
            color=plt.cm.Dark2(n)
            ax.step(WIND_omni_times,WIND_omni_en,where='post',label='WIND',color=color)
            ax.plot(WIND_omni_times,WIND_omni_en_smoothed,color='grey')
            
            #plot the bal_arrival_time
            ax.axvline(x=bal_arrival,color=color,linewidth=1.5,linestyle='dashed')
            
            '''
                        
            #plotting t_o and t_d
            ax.axvline(x=peak_event,color='black',linewidth=1.5,linestyle='dashed')
            ax.axvline(x=t_d_event,color='black',linewidth=1.5,linestyle='dashed')
            ax.axhline(y=0.5)
            '''
            
            ax.yaxis.set_tick_params(labelsize=self.fsize_ticks)
            
            
            ## annotating the approximate energy
            str_energy=str(event_data.energies[n])+'keV'
            ax.annotate(str_energy, (0.75,0.7),xycoords='axes fraction',fontsize=self.fsize_ticks,color=color)
            ax.tick_params(labelbottom=False)
            if n==n_ener_WIND-1:
                ax.tick_params(labelbottom=True)
                ax.xaxis.set_tick_params(labelsize=self.fsize_ticks)
                ax.set_xlabel('UT Time ($t_{inj}$: '+str(inj_time)+')',fontsize=self.fsize_head)
            
        plt.subplots_adjust(wspace=0, hspace=0.05)
        
        plt.show(block=False)
        plt.savefig(os.path.join(folder_path,'fig+all_energies_comparison_.pdf'),format='pdf',bbox_inches='tight')
            

        
        return 
    
    def plot_one_omni_characteristic(self,plot,string_dict,folder_path):
        #loading in the event chars to plot
        event_chars = np.load(os.path.join(self.data_path,"event_chars.npy"))
        #loading in the estimated uncertainties
        event_chars_uncertainties = np.load(os.path.join(self.data_path,"event_chars_uncertainties.npy"))
        #load in the relevent data
        data_toplot=self.load_files_toplot(plot)

        '''
        choosing which energy to plot, currently self.energy_to_index only works if you consider all the WIND energy channels.
        Right now, band-aid fix to only consider
        '''
        en_inds = self.energy_to_index()
        #en_inds = 0

        
        for char in self.char_toplot:
            #extract the sim charecteristics:
            if char=='t_peak':
                ind=0
                str_y = '$t_p$ [s]'
                y_err=self.t_binwidth/2
            if char=='t_r':
                ind=1
                str_y ='$t_r$ [s]'
                y_err = self.t_binwidth
            if char=='t_d':
                ind=2
                str_y='$t_d$ [s]'
                y_err = self.t_binwidth
            sim_chars = data_toplot[0]
            #extracting the omni charecteristic based on which one is requested
            sim_omni_chars=sim_chars[en_inds,ind,1]
            event_omni_chars = event_chars[en_inds,ind]
            event_omni_chars_uncertainty = event_chars_uncertainties[en_inds,ind]

            
            fig,ax =plt.subplots(1,figsize=(7,7))
            
            #wind rough energies
            WIND_energies = np.array([27,40,67,110,180,310,520])
            #extracting the required energies for plotting
            WIND_energies = WIND_energies[en_inds]
            
            #using the defined binwidth sampling for the simulated uncertainties

            '''
            Change if it is the peak to t_res/2!
            '''

            ax.errorbar(WIND_energies,sim_omni_chars,yerr=y_err,color='green',marker='+',capsize=5)
            ax.plot(WIND_energies,sim_omni_chars,color='green')


            ax.errorbar(WIND_energies,event_omni_chars,yerr=event_omni_chars_uncertainty,color='orange',marker='*',capsize=5)
            ax.plot(WIND_energies,event_omni_chars,color='orange')
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            #ax.set_xlim(10,1000)
                    
            
            ax.set_ylabel(str_y,fontsize=self.fsize_head)
            ax.set_xlabel('Energy [keV]',fontsize=self.fsize_head)
            ax.tick_params(labelsize=self.fsize_ticks)
            
           # ax.axvline(x=self.break_energy,linestyle='--',color='grey')
            
            '''
            annotating bottom to see simulation data
            '''
            str_mfp_n=string_dict['mfp-_toplot[AU]']
            str_mfp_p=string_dict['mfp+_toplot[AU]']
            str_alpha=string_dict['alpha_toplot']
            str_kappa=string_dict['kappa']
            str_sim='Simulated'
            str_event='WIND'
            
    
            ax.annotate(str_mfp_n, (0.6,0.9),xycoords='axes fraction',fontsize=self.fsize_head,color='green')
            ax.annotate(str_mfp_p, (0.6,0.8),xycoords='axes fraction',fontsize=self.fsize_head,color='green')
            ax.annotate(str_alpha, (0.6,0.7),xycoords='axes fraction',fontsize=self.fsize_head,color='green')
            ax.annotate(str_kappa, (0.6,0.6),xycoords='axes fraction',fontsize=self.fsize_head,color='green')
            ax.annotate(str_event,(0.6,0.5),xycoords='axes fraction',fontsize=self.fsize_head,color='orange')
            

            plt.show(block=False)
            plt.savefig(os.path.join(folder_path,'fig+char_'+str(char)+'.pdf'),format='pdf',bbox_inches='tight')
        
        
        
        return


    #function to build the array that holds the omnidirectional characteristics for all simulations considered
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


    '''
    could turn this into a general function for variables over kappa (similar to sim_omni_chars)
    for now just use two seperate functions. If i need to repeat this again, will make a general function
    '''
    def build_omni_counts_over_kappa(self,):
        '''
        important! we assume here each file over kappa contains the same numpy shapes of data
        we are also making assumptions on the order that things are saved in.... check!
        '''
        #create an array that contains all file names in a folder
        data_names=os.listdir(self.data_path)
        #loop through each file name
        #picks out the sim_omni CHARECTERISTICS for each simulation (t_o,t_d,HWHM_o,HWHM_d)
        keyword='omni_counts_data_'
        #create a big array that will hold the 
        omni_counts_data_overkappa = []
        for file in data_names:
            #seperating out into chars files and var files
            #also extract the kappa values used
            if keyword in file:
                omni_counts_data = np.load(os.path.join(self.data_path,file))
                omni_counts_data_overkappa.append(omni_counts_data)
                
        try:
            omni_counts_data_overkappa=np.array(omni_counts_data_overkappa)
        
        except:
            #deprecation needs to be upgraded to fatal error for this to work
            raise SystemError('Check simulation dimensions for each kappa! They should be equal!')
    

        return omni_counts_data_overkappa


    #function that outputs the array holding the indices of the neergies we are considering for plotting
    def energy_to_index(self):
        #create a dictionary that holds the corresponding index for each energy
        ind_dict={}
        for i,energy in enumerate(self.ee):
            ind_dict[str(energy)]=i
        
        #prepare an array to hold the indices
        inds = []
        #for each requested charecteristics
        for energy in self.energy_char_toplot:
            #extract teh index
            ind = ind_dict[str(energy)]
            #append to the index array
            inds.append(ind)
        #return that index array
        return inds


    
    '''
    Adapt this to select the chosen energy out of multiply simulated energies?
    '''

    def plot_all_omni_characteristic(self,plot,string_dict,folder_path):

        #loading in the event chars to plot
        event_chars = np.load(os.path.join(self.data_path,"event_chars.npy"))
        event_chars_uncertainties = np.load(os.path.join(self.data_path,"event_chars_uncertainties.npy"))
        #loading in all the computed simulation characteristics and labels
        sim_chars_data = self.build_sim_chars_over_kappa()
        #extracting the energies requested to plot
        en_inds = self.energy_to_index()

        '''
        manually override en_inds for now as we are using only one energy
        '''
        #en_inds = 0
        #decompose into sim_chars and sim_vars as defined on construction
        sim_vars = sim_chars_data[...,0]
        sim_chars = sim_chars_data[...,1]

        #find how many simulations are being considered overall
        multiplied_tup = np.prod(np.shape(sim_chars_data)[0:3])
        #build the reshaped array for easy plotting (flattening over the simulations)
        shape = (multiplied_tup,7,4)

        #extracting the number of simulation paramaters in each category
        n_kappa,n_alpha,n_pairs = np.shape(sim_chars_data)[0:3]


        #extracting the required energy data
        sim_chars = sim_chars[...,en_inds,:]
        sim_vars =sim_vars[...,en_inds,:]
        event_chars=event_chars[en_inds,:]
        event_chars_uncertainties = event_chars_uncertainties[en_inds,:]

        #defining total WIND energies and cutting depending on requested energy channels
        WIND_energies = np.array([27,40,67,110,180,310,520])
        WIND_energies = WIND_energies[en_inds]


        '''
        Building the figure
        '''

        #creating the widths/heights aspect ratios (last axis will be the colorbar)
        wrs = []
        for k in range(n_kappa):
            wrs.append(1)
        wrs.append(0.1)

        fig = plt.figure(figsize=(20,7))
        gs = GridSpec(1, n_kappa+1, figure=fig,width_ratios=wrs,height_ratios=[1])
        #ax1 and ax2 will be the WIND/WAVES data
        
        #extracting the omnidirectional characteristics for the event
        t_p_event = event_chars[0,0]
        t_r_event = event_chars[0,1]
        t_d_event = event_chars[0,2]

        #extracting the uncertainties for the event
        t_p_event_unc = event_chars_uncertainties[0,0]
        t_r_event_unc = event_chars_uncertainties[0,1]
        t_d_event_unc = event_chars_uncertainties[0,2]


        #finding the maximum decay time for plotting
        t_d_sim_max = np.max(sim_chars[...,2])
        t_d_sim_min = np.min(sim_chars[...,2])


        #defining the min max for the colorbar
        #t_p_min must be smaller than that actual minimum so no data points are colored white!
        #Set the event decay time as grey!
        t_d_min_cbar = 0
        t_d_max_cbar = 2

        #defining the diverging colormap (diverging from 1)
        cmap='coolwarm'
        #norm=mcolors.TwoSlopeNorm(vcenter=1,vmin=t_d_min_cbar,vmax=t_d_max_cbar)

        #t_d_max = np.max(sim_chars[...,2])+20
        #t_p_min must be smaller than that actual minimum so no data points are colored white!
        t_p_min = np.min(sim_chars[...,0])-100
        t_p_max = np.max(sim_chars[...,0])+100

        for k in range(n_kappa):
            ax= fig.add_subplot(gs[0, k])
            ax.scatter(t_r_event,t_p_event,c=1,marker='x',cmap=cmap,s=50,vmin=t_d_min_cbar,vmax=t_d_max_cbar)
            ax.errorbar(t_r_event,t_p_event,xerr=t_r_event_unc,yerr=t_p_event_unc,color='grey',lw=0.2,capsize=2)
            for a in range(n_alpha):
                for m in range(n_pairs):
                    #extracting the tuple of omnidirectional characteristics
                    tup = sim_chars[k,a,m,0]
                    #extracting the tuple of the ascociated simulation used to run that particular simulation
                    var_tup =sim_vars[k,a,m,0]

                    #extracting teh omnidirectional characteristics
                    t_p = tup[0]
                    t_r=tup[1]
                    t_d=tup[2]

                    #plotting and placing error bars
                    ax.scatter(t_r,t_p,c=t_d/t_d_event,cmap=cmap,vmin=t_d_min_cbar,vmax=t_d_max_cbar)
                    ax.errorbar(t_r,t_p,xerr=self.t_binwidth,yerr=self.t_binwidth/2,color='grey',lw=0.2,capsize=2)
                    ## annotate the selected variable!

                    str_mfp = '$\lambda_{\parallel}^{+/-}$='+str(var_tup[3])+' [AU]'
                    if var_tup[0]==self.kappa and var_tup[3]==self.mfp_toplot[1]:
                        ax.annotate(str_mfp,xytext=(t_r-100,t_p+100),xy=(t_r,t_p),arrowprops=dict(arrowstyle='-|>'))


            kappa = sim_vars[k,a,m,0,0]
            str_kappa = "$\kappa=$"+str(kappa)
            ax.annotate(str_kappa, (0.1,0.9),xycoords='axes fraction',fontsize=self.fsize_head,color='black')
            #ax.errorbar(WIND_energies,event_omni_chars,yerr=event_omni_chars_uncertainty,color='black',marker='*',capsize=5)
            ax.set_xscale('log')
            #ax.set_yscale('log')
            ax.set_xlim(10,1000)   
            #ax.set_ylim(1000,10000)
            ax.set_ylim(t_p_min,t_p_max)   

            if k==0:
                ax.set_ylabel('$t_p$ [s]',fontsize=self.fsize_head)
            ax.set_xlabel('$t_r$ [s]',fontsize=self.fsize_head)
            ax.tick_params(labelsize=self.fsize_ticks)

            if k>0:
               ax.tick_params(labelleft=False)


            plt.setp(ax.get_yminorticklabels(), visible=True)

        '''
        Setting up the colorbar
        '''
        #tricking matplotlib into making a colorbar using a ghost image
        fig_ghost, ax_ghost = plt.subplots(1,figsize=(7,7))
        #map_ghost = ax_ghost.imshow(np.array([[1,2],[1,2]]),cmap='viridis',norm=mcolors.LogNorm(vmin=t_p_min,vmax=t_p_max))
        map_ghost = ax_ghost.imshow(np.array([[1,2],[1,2]]),cmap=cmap,vmin=t_d_min_cbar,vmax=t_d_max_cbar)
        
        ax=fig.add_subplot(gs[0, -1])
        ax.axis('off')
        cbar = fig.colorbar(map_ghost, ax=ax,location='right',fraction=2.0, pad=2.0,ticks=[0, 1, 2])
        cbar.set_label('$t_d$ / [$t_d$ (event)]',fontsize=self.fsize_head)
        cbar.ax.set_yticklabels(['0', '1', '>2'])
        

        plt.show(block=False)

        
        #ax.plot(WIND_energies,sim_omni_chars,color='green')
        #ax.plot(WIND_energies,event_omni_chars,color='black')

        #cmap = plt.cm.nipy_spectral
        #norm = mcolors.Normalize(vmin=0, vmax=multiplied_tup)   
        #n=0
        #for k in range(n_kappa):
            #for alph in range(n_alpha):
                #for p in range(n_pairs):
                    #n=n+1
                    #tup=sim_omni_chars[k,alph,p]
        #for i,tup in enumerate(flat_sim_omni_chars):
                    #ax.scatter(WIND_energies,tup,color=cmap(norm(n)),marker='.')
                    #ax.plot(WIND_energies,tup,color=cmap(norm(n)))

        
        return None


    '''
    Adapt this to select the chosen energy out of multiply simulated energies?
    '''
    def plot_all_simulations_one_energy(self,plot,string_dict,folder_path):
        omni_counts_data_overkappa = self.build_omni_counts_over_kappa()
        sim_chars_data_overkappa = self.build_sim_chars_over_kappa()

        #we will ALWAYS be considering for these plots one energy, no energy dependence.
        #so to loop over the mean free paths, we will extract the number of mean free paths
        n_mfp = np.shape(omni_counts_data_overkappa)[2]
        n_kappa = np.shape(omni_counts_data_overkappa)[0]
        inj_time=funcs.load_injection_time()
        '''
        loading in all the required data
        '''
        
        #loading in the simulated electron data
        
        data_toplot=self.load_files_toplot(plot)
        #loading in the datetimes bins since (radio-derived) injection
        #these should be identical for ALL simulations considered, so steal them from the "requested" simulation 
        t_bins_inj=data_toplot[0]
        bal_arrival_time=data_toplot[1][self.n_en]
        #ballistic arrival times (in minutes since injection)
        bal_arrival_time = inj_time+timedelta(minutes=bal_arrival_time) 
        #loading in the WIND data (radio and electron data)
        #first load in the events dictionary
        event_data=funcs.load_events_data()
        #loading in the pads data
        event_en=np.where(self.event_energy==event_data.energies)[0][0]
        #unpacking the WIND/WAVES data for the event
        waves_times=event_data.waves_times
        waves_freq_rad1=event_data.waves_freq_rad1
        waves_freq_rad2=event_data.waves_freq_rad2
        waves_volts_rad1=event_data.waves_volts_rad1
        waves_volts_rad2=event_data.waves_volts_rad2
        
        
        #loading in the WIND omni data
        WIND_omni_times=event_data.omni_times
        WIND_omni_fluxes=event_data.omni_fluxes[event_en]
        WIND_omni_fluxes_smoothed=event_data.omni_fluxes_smoothed[event_en]

        '''
        finding the radio travel time and expected arrival
        '''
        
        radio_travel_time_minutes=funcs.light_travel_time_1AU()
        radio_arrival_time = inj_time+timedelta(minutes=radio_travel_time_minutes)

        '''
        building the figures need two panels for the radio data, and n_kappa panels for each set of simulations with the different kappa values
        '''

        #defining the height ratios array, an array of ones of length 2+n_kappa

        hrs = []
        for n in range(2+n_kappa):
            hrs.append(1)
        #fig = plt.figure(constrained_layout=True,figsize=(25,15))
        fig = plt.figure(figsize=(15,15))
        gs = GridSpec(2+n_kappa, 2, figure=fig,width_ratios=[1,0.14], height_ratios=hrs,)
        #ax1 and ax2 will be the WIND/WAVES data
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
        #ax1a and 2a will hld the colorbar for the WIND data
        ax1a = fig.add_subplot(gs[0,1])
        ax1a.axis('off')
        ax2a=fig.add_subplot(gs[1,1])
        ax2a.axis('off')


        '''
        plotting the radio data
        '''
        min_val=10e-1
        max_val=max(np.max(waves_volts_rad1),np.max(waves_volts_rad2))
        
        ax1.set_ylabel(r'Frequency [kHz]',fontsize=self.fsize_head)
        labelx = -0.05
        ax1.yaxis.set_label_coords(labelx, 0.0)
        ax1.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        ax1.tick_params(labelbottom=False)
        ax1.set_yscale('log')
        im=ax1.pcolormesh(waves_times,waves_freq_rad2,waves_volts_rad2.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        #cbar = fig.colorbar(im, ax=ax1a,location='right')
        cbar = fig.colorbar(im, ax=ax1a,)
        cbar.ax.tick_params(labelsize=self.fsize_ticks)
        cbar.ax.set_ylabel('Norm_Volt_RAD1/-2', fontsize=self.fsize_head,x=0.3,y=0.0)
        ax1.tick_params(labelbottom=False)
            
        ax1.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')
        ax1.axvline(x=radio_arrival_time,color='yellow',linewidth=1.5,linestyle='dashed')
        
        #ax.set_xscale('log')
       # ax1.annotate('CSV PADS summed', (0.7,0.8),xycoords='axes fraction',fontsize=self.fsize_head

        #ax2.set_xlabel(r'Time (Start Time: '+str(waves_times[0])+')',fontsize=self.fsize_head)       
        ax2.xaxis.set_tick_params(labelsize=self.fsize_head)
        ax2.yaxis.set_tick_params(labelsize=self.fsize_ticks)
        myFmt = mdates.DateFormatter('%H%M')
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.set_yscale('log')
        im2=ax2.pcolormesh(waves_times,waves_freq_rad1,waves_volts_rad1.T,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val),cmap='turbo')
        cbar2 = fig.colorbar(im2, ax=ax2a,)
        cbar2.ax.tick_params(labelsize=self.fsize_ticks)
        #cbar2.ax.set_ylabel('Norm_Volt_RAD1', fontsize=self.fsize_head)
        ax2.tick_params(labelbottom=False)
                    
        ax2.axvline(x=inj_time,color='grey',linewidth=1.5,linestyle='dashed')

        plt.subplots_adjust(wspace=0, hspace=0.075)
        
        ax2.axvline(x=radio_arrival_time,color='yellow',linewidth=1.5,linestyle='dashed')


        sim_vars = sim_chars_data_overkappa[...,0]
        for n in range(n_kappa):
            kappa = sim_vars[n,0,0,0,0]
            str_kappa = "$\kappa=$"+str(kappa)
            ax = fig.add_subplot(gs[2+n,0],sharex=ax1)

            str_energy = 'E='+str(self.energy_toplot)+str(' [keV]')
            #and another axis for the colorbar (to denote the mean free paths)
            ax.annotate(str_kappa, (0.78,0.85),xycoords='axes fraction',fontsize=self.fsize_head,color='black')


            if n!=n_kappa-1:
                ax.tick_params(labelbottom=False) 

            all_WIND_energies = event_data.energies
            try:
                m=np.where(self.energy_toplot==all_WIND_energies)[0][0]
                color_WIND=plt.cm.Dark2(m)
            except:
                raise SystemError('Error choosing color')
            #WIND data
            ax.step(WIND_omni_times,WIND_omni_fluxes,where='post',label='WIND',color='black')
            ax.plot(WIND_omni_times,WIND_omni_fluxes_smoothed,color='grey')
            ax.axvline(x=bal_arrival_time,color=color_WIND,linestyle='--')  

            #cmap = plt.cm.jet  # define the colormap
            # extract all colors from the .jet map
            cmap = cm.get_cmap('rainbow', n_mfp)  
            ticks = np.array([0,1,2])

            for m in range(n_mfp):
                #assuming symmetric for these plot
                mfp_val = sim_vars[n,0,m,0,2]
                str_mfp = '$\lambda_{\parallel}^{+/-}$='+str(mfp_val)+' [AU]'
                color_sim=cmap(m)
                omni_counts = omni_counts_data_overkappa[n,0,m,0]
                ax.stairs(omni_counts,t_bins_inj,color=color_sim,linestyle='--')
                
                if n==2:
                    ax.annotate(str_mfp, (1.05,0.85-0.3*m),xycoords='axes fraction',fontsize=self.fsize_head,color=color_sim)

            if n==2:
                ax.annotate(str_energy, (1.05,0.85-0.3*n_mfp),xycoords='axes fraction',fontsize=self.fsize_head,color='black')

            
            if n==n_kappa-1:
                ax.set_xlabel('UT Time ($t_{inj}$:'+str(inj_time)+')',fontsize=self.fsize_head)


        plt.show(block=False)
        plt.savefig(os.path.join(folder_path,'fig+all_simulations_'+str_energy+'.pdf'),format='pdf',bbox_inches='tight')
        pass

    def plot_simulation(self,sols):
        #build the string dictionary for the requested simulation data
        string_dict=self.build_string_dictionary()
        #before plotting, create the simulation folder
        folder_path=self.create_folder()
        
        for plot in self.plots_toplot:                
            if plot=='injection':
                self.plot_injection(plot,string_dict,folder_path)
                
            if plot=='diffusion coef':
                self.plot_diffusion_coef(plot,string_dict,folder_path)
                
            if plot=='zmu':
                self.plot_zmu(plot,string_dict,folder_path)
                
            if plot=='z and mu':
                self.plot_z_and_mu(plot,string_dict,folder_path,sols)
                
            if plot=='electron flux':
                self.plot_electron_flux(plot,string_dict,folder_path,sols)
           #if plot=='plot_event':
                #self.plot_event()
            if plot=='plot_WIND_electrons_test':
                self.plot_WIND_electrons_test()
            if plot=='plot_WIND_waves':
                self.plot_WIND_waves()
            if plot=='plot_WIND_one_energy_comparison':
                self.plot_WIND_one_energy_comparison(plot,string_dict,folder_path)
            if plot =='plot_WIND_all_energy_comparison':
                self.plot_WIND_all_energy_comparison(plot,string_dict,folder_path)

            if plot=='plot_one_omni_characteristic':
                self.plot_one_omni_characteristic(plot,string_dict,folder_path)
            if plot=='plot_all_omni_characteristic':
                self.plot_all_omni_characteristic(plot,string_dict,folder_path)
            
            #need radio data, omnidirectional data and to load in all the simulations as in plot_all_omni_characteristic
            if plot=='plot_all_simulations_one_energy':
                self.plot_all_simulations_one_energy(plot,string_dict,folder_path)
                
        return
    

    


























    