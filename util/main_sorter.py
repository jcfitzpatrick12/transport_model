# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:42:53 2022

@author: jcfit

script that takes the sample steps of the simulation and sorts the data ready to be plotted by flexi_plotter script
"""

import numpy as np
import os
import matplotlib.pyplot as plt

#import warnings
import sys
from pathlib import Path   
#importing the global functions file
import util.global_functions as funcs

from datetime import datetime,timedelta

'''
adding the project to the system path so to allow for importing transport scripts
'''

class sorting_functions:
    def __init__(self,dicts):    
        #seperating the dictionaries out
        self.sim_dict=dicts['sim_dict']
        self.sort_dict=dicts['sort_dict']
        self.WIND_dict=dicts['WIND_dict']
        
        
        #extracting info from sim_dicts
        self.Np=self.sim_dict['num_particles']
        self.alpha_vals=np.array(self.sim_dict['alpha_vals'])
        self.kappa=self.sim_dict['kappa']
        self.mfp0_vals = np.array(self.sim_dict['mfp0_vals[AU]'])
        self.ee=np.array(self.sim_dict['energies[keV]'])
        self.t_end = self.sim_dict['t_end[days]']
        self.z_init=self.sim_dict['z_injection[AU]']
        self.inj_set=self.sim_dict['injection_type']
        
        #extracting the info from sort_dict
        self.M=self.sort_dict['num_samples']
        self.z_obs = self.sort_dict['z_observation[AU]']
        self.z_tol=self.sort_dict['z_tolerance[AU]']
        self.t_binwidth=self.sort_dict['t_binwidth[s]']
        self.z_beds = self.sort_dict['z_beds[AU]']
        
        #creating global variables (within this class) that are used in different functions
        self.n_pairs = np.shape(self.mfp0_vals)[0]
        self.n_alph=len(self.alpha_vals)
        self.n_ener = len(self.ee)
        #creating the sim tuple for convenience in defining some arrays
        self.tup=(self.n_alph,self.n_pairs,self.n_ener,self.Np)
        
        self.thresh = self.WIND_dict['thresh']
        
        
    #function that returns the ballistic arrival times in minutes
    def find_bal_times(self,v):
        return 1440*(self.z_obs-self.z_init)/v
    
    #returns histogrammed values with no funny business regarding number of entries!
    #value h[i] in the histogram belongs to [beds[i],beds[i+1]) according to the numpy docs (aside from last which is a closed set)
    def histogram(self,X,beds):
        h,beds=np.histogram(X,beds)
        return h
    
    #returns histogrammed values with no funny business regarding number of entries!
    #value h[i] in the histogram belongs to [beds[i],beds[i+1]) according to the numpy docs (aside from last which is a closed set)
    def histogram_laxis(self,ar,beds):
        #creating an array to hold the new array (histogrammed along the last axis)
        #histogramming along the last axis of the multidim array
        h_np=np.apply_along_axis(lambda x: self.histogram(x,beds), axis=-1, arr=ar)
        #isolating the histogrammed values
        return h_np
    
    #takes in the concanetanated along the last axis and performs this histogram
    def histo2D(self,X,x_beds,y_beds):
        #len(X) will always be even, since it is derived from concatenating two lists of equal length
        N=int(len(X)/2)
        #seperating out the x values
        x_vals = X[0:N]
        #and the y values
        y_vals = X[N:]
        #performing the histogram
        H,_,_ = np.histogram2d(x_vals, y_vals, bins=(x_beds, y_beds))
        return H

    #histogram along the last axis where the last axis holds the (x,y) values of equal lengths
    def histo2D_laxis(self,ar,x_beds,y_beds):
        #concatenate the x and y values so to ass into np.apply_along_axis
        ar_conc=np.concatenate((ar[...,0],ar[...,1]),axis=-1)
        #H=np.apply_along_axis(lambda x: np.histogram2d(x[...,0],x[...,1], bins=(x_beds, y_beds)), axis=-1, arr=ar)
        H=np.apply_along_axis(lambda x: self.histo2D(x,x_beds,y_beds), axis=-1, arr=ar_conc)
        return H
    
    #function that creates the electron distributions at z_obs
    def simulate_electron_fluxes(self,sim_zmu,dt_vals):
        '''
        creating the relevent time bins and arrays
        '''
        #converting t_binwidths to days
        t_binwidth_days = self.t_binwidth/86400
        #finding the number of bins
        n_bins = int(np.round(self.t_end/t_binwidth_days))
        #creating the time bins (days)
        t_bins = np.linspace(0,self.t_end,n_bins)
        
        #plot time elapsed in minutes
        t_binwidth_minutes  = self.t_binwidth/60
        #converting t_bins to minutes
        t_bins_minutes = t_bins*1440 
        '''
        creating the pitch angle distributions and strong counting omnidirectional fluxes
        '''
        z_sim = sim_zmu[...,0]
        mu_sim = sim_zmu[...,1]
        #finding the number of simulation timesteps
        n_steps = np.shape(z_sim)[-1]
        #creating the pitch-angle beds for the pitch angle distributions
        
        #distance between the mu bed edges
        n_mu_bins=14
        
        
        #for comparisons with WIND data, we will convert to pitch-angle theta.
        #so bins will be non-uniform in mu
        #ang_beds_pads = np.linspace(0,180,n_mu_bins+1)
        #rad_beds_pads = np.linspace(0,np.pi,n_mu_bins+1)
        #mu_beds_pads = np.cos(rad_beds_pads)*(-1)
    
        mu_beds_pads = np.linspace(-1,1,n_mu_bins+1)

        n_mu_beds = len(mu_beds_pads)
        #relabelling t_bins_minutes for clarity
        t_beds = t_bins_minutes
        
        #comparison array to preserve shape for vector comparison
        #sets the detection tolerance around the observation location (for pitch angle distributions and strong omni detection)
        comparison_array = np.ones(self.tup)*self.z_tol
        #initialising an array to hold the 
        pitch_angle_dists = np.ones((self.n_alph,self.n_pairs,self.n_ener,n_mu_beds-1,n_steps))
        #constructing the pitch angle distributions at every timestep
        for n in range(n_steps):
            #extracting the z and mu distributions at a given time step
            z_sim_atstep = z_sim[...,n]*1
            mu_sim_atstep = mu_sim[...,n]*1
            #recentering particles around the observation location
            z_sim_atstep-=self.z_obs
            #taking the absolute value
            z_sim_atstep=np.abs(z_sim_atstep)
            detected_bool = np.less_equal(z_sim_atstep,comparison_array)
            #converting undetected particles to inf
            mu_sim_atstep[~detected_bool]=np.inf
            #histogram along the last axis
            #mu_sim_atstep=
            pitch_histo_atstep = self.histogram_laxis(mu_sim_atstep,mu_beds_pads) 
            #place into external array for the given timestep
            pitch_angle_dists[...,n]=pitch_histo_atstep
        
            
        #swapping the angular bin and time axis so that the last axis is the angular bins, necessary for bin averaging
        pitch_angle_dists=np.swapaxes(pitch_angle_dists,-1,-2)
        
        
        #creating the array with the time at each timestep (such a loop will be constant over alpha (and kappa))
        #this array will help locate the relevent pitch angle distributions to average over!
        
        tars = np.empty((self.n_alph,self.n_pairs,self.n_ener)+(n_steps,))
        #this array holds the time at each timestep
        for p in range(self.n_pairs):
            for e in range(self.n_ener):
                #extract the specfic dt value for that combination
                dts=dt_vals[p,e]
                #creating the time array and placing it into the shaped matrix
                c = np.arange(0,n_steps,1)
                tars[:,p,e,:]=c*dts*1440
                    
        #how many timesteps are we looping through. Not, we take the MIDPOINT w.r.t time so only need to loo through n_bins-1?
        n_global = n_bins-1
            
        #averaged pitch angle distributions holds the number of counts. ith time bin holds the counts in i,i+1 time bin
        pitch_angle_dists_avved = np.empty((self.n_alph,self.n_pairs,self.n_ener,n_global,n_mu_beds-1))
        #another comparison array to preserve vector form
        comparison_array = np.ones((self.n_alph,self.n_pairs,self.n_ener,n_steps))*t_binwidth_minutes/2
        #averaging over each time bin!
        for n in range(n_global):
            #preserve the total pitch angle distribution array
            _pitch_angle_dists=pitch_angle_dists*1
            #finding the middle of bin time
            mid_time = t_bins_minutes[n]+t_binwidth_minutes/2
            #centering all times by the midtime and taking absolute value
            tars_prime=np.abs(tars-mid_time)
            #which bins are within this time bin? True if in the time bin
            which_bins = np.less(tars_prime,comparison_array)
            #neglecting all bins outwith the time bin
            _pitch_angle_dists[~which_bins]=np.nan
            #averaging over the pitch angle distributions WITHIN the time slot (neglecting nans)
            pitch_angle_dists_avved[...,n,:]=np.nanmean( _pitch_angle_dists,axis=3)
                
                        
        #relabbeling the averaged pitch angles for convenience
        pads = pitch_angle_dists_avved
        
         
        '''
        see how compares to classic simple sum
        '''
        
        omni_counts = np.nansum(pads,axis=4)
        omni_counts_ers = np.sqrt(omni_counts)
        
        #normalising omni_counts to the peak flux
        
        omni_norms = np.nanmax(omni_counts,axis=-1)
        for n in range(n_global):
            omni_counts[...,n]/=omni_norms
            omni_counts_ers[...,n]/=omni_norms
        
    

        '''
        normalising the pad
        '''
        
        #and normalise the pads so to account for the change of variable!
        
        #normalising pitch angle distributions to represent phase density
        #first sum over all the bins to get the total counts for each combination
        pads_sums = np.nanmax(pads,axis=(-1,-2))
        
        #find the area of each bin
        #pads_area_ofbins = t_binwidth_minutes*(mu_beds[-1]-mu_beds[-2])
        #divide by this norm
        for n in range(n_global):
            for m in range(n_mu_beds-1):
                pads[...,n,m]/=pads_sums


        #find the equivalent pitch-angle beds (simple edge conversion?) 
        ang_beds_pads = np.arccos(mu_beds_pads)*180/np.pi

        '''
        Saving omni_counts
        '''
                #print(fname)
        data_path=funcs.folder_functions().return_datapath()
        #saving the numeric char values
        fname = 'omni_counts_data_kappa='+str(self.kappa)
        #print(container)
        #pack into one array so to not confuse them in unloading in the fitting class!
        np.save(os.path.join(data_path,fname),omni_counts)
        #save the general t_bins_minutes (identical for all the sorted simulations when running different sims for comparisons)
        #this variable is ONLY used for plotting plot_all_simulations_one_energy
        np.save(os.path.join(data_path,'general_t_bins_minutes'),t_bins_minutes)
        return t_bins_minutes,mu_beds_pads,pads,omni_counts,omni_counts_ers,ang_beds_pads
    
    #function that takes the whole simulation array, and samples it at particular timesteps
    def sample_sim(self,sim_zmu,max_steps,dt_vals):
        '''
        sampling the simulation array at specific timesteps
        '''
        #the sample_steps we are going to take (equally sampling then converting to an integer)
        sample_steps = np.round(np.linspace(1,max_steps,self.M))
        #converting sample steps to integers
        sample_steps = sample_steps.astype(int)
        
        #a 1D array containing all the timesteps. From which, create an array with true at the desired timesteps required, False else
        indt = np.arange(0,max_steps+1,1)
        indt[sample_steps]=1
        indt[indt!=1]=0
        indt = np.array(indt,dtype='bool')
        
        #sampled_zmu will hold the simulations at the particular timesteps required:
        sampled_sim = np.ones(self.tup+(self.M,2,))
        #slice the timesteps we want, sampled_zmu now holds the simulation sampled at the various timesteps
        sampled_sim = sim_zmu[:,:,:,:,indt,:]
        #swap particle and timestep axis, in order to make use of histogram_laxis later on (particles being the last axis)
        sampled_sim = np.swapaxes(sampled_sim,-2,-3)
        
        '''
        finding the times of the steps sampled
        '''
        ## now find the times for each timestep for each combination! VERY IMPORTANT, as require this for the analytical comparisons
        #dt_array to deal with elementwise multiplication
        dt_ar = np.ones((self.M,self.n_pairs,self.n_ener))
        dt_ar[:]=dt_vals
        #dt_ar[mfp,e] holds M copies of the timestep corresponding to the combination mfp,e
        dt_ar = np.moveaxis(dt_ar,0,-1)
        
        #any combination of mfp,e should holds the same M timesteps to be sampled from
        sample_ts = np.ones((self.n_pairs,self.n_ener,self.M))
        sample_ts[:,:] = sample_steps
        
        #sample_ts holds the sampling times for each mfp,e combination corresponding to the M timesteps sampled from
        sample_ts = sample_ts*dt_ar
        
        
        return sampled_sim,sample_ts
    
    #bins the z and mu data separately at given timestamps
    def z_and_mu_sampling(self,sampled_sim):
        #create binnings seperately for z and mu!
        #np.arange does not include the end, so 1.1. instead of 1.
        mu_beds = np.linspace(-1,1,21)
        #create z bin edges
        #z_beds = np.linspace(0,1.5,150)
        z_beds = self.z_beds
        #z_beds=np.linspace(0,2,150)
        
        #histogramming pure number counts for z and mu over respective binnings
        h_z_num = self.histogram_laxis(sampled_sim[:,:,:,:,:,0], z_beds)
        h_mu_num= self.histogram_laxis(sampled_sim[:,:,:,:,:,1], mu_beds)
        h_z_num=np.array(h_z_num,dtype='float')
        
        #unless we have instantaneous injection, remove the injection bin (since particles are held here until injection)
        if self.inj_set!='instantaneous':
            # print('removing injection bin')
            z_recentred = np.abs(z_beds-self.z_init)
            # print(z_recentred)
            #place within try, as if this has already been performed will cause an error
            #where is the injection bin (what index?)
            where_bin_inj = np.where(z_recentred==np.min(z_recentred))
            #extracting the index
            bin_ind = where_bin_inj[0][0]
            h_z_num[...,1:,bin_ind-1:bin_ind+1]=np.nan
        
        #plt.step(z_beds,h_z_num[0,0,0,12,:],where='post')
        z_norm = (z_beds[-1]-z_beds[-2])*np.nansum(h_z_num,-1)
        
        
        n_zbeds=np.shape(h_z_num)[-1]
        ers_z = np.sqrt(h_z_num)
        for n in range(n_zbeds):
            h_z_num[...,n]=h_z_num[...,n]/z_norm
            ers_z[...,n] = ers_z[...,n]/z_norm
            
            
        h_mu_num=np.array(h_mu_num,dtype='float')
        mu_norm = (mu_beds[-1]-mu_beds[-2])*np.nansum(h_mu_num,-1)
        n_mubeds=np.shape(h_mu_num)[-1]
        ers_mu=np.sqrt(h_mu_num)
        for n in range(n_mubeds):
            h_mu_num[...,n]=h_mu_num[...,n]/mu_norm
            ers_mu[...,n]=ers_mu[...,n]/mu_norm
            
        return z_beds,h_z_num,ers_z,mu_beds,h_mu_num,ers_mu

        
    #function that performs the joint zmu sampling in 2D at the requested timestep.
    def zmu_sampling(self,sampled_sim,sample_ts,v,mu_beds):
        #using identical mu_beds as in z and mu sampling
        #mu_beds = np.linspace(-1,1,21)
        #first normalise the distances in sampled_zmu to between 0 and 1 via the z/vt_sample normalisation
        shape = np.shape(sampled_sim[...,0])
        
        #create a vectorised v_array to hold the 
        v_ar = np.ones(shape)
        for n in range(self.n_ener):
            v_ar[:,:,n,...]=v[n]
         
        #create a vectorised time array
        sampled_ts_ar=np.ones(shape)
        sampled_ts_ar = np.moveaxis(sampled_ts_ar,-1,1)
        sampled_ts_ar[:,:]=sample_ts
        sampled_ts_ar = np.moveaxis(sampled_ts_ar,1,-1)
        norm = sampled_ts_ar*v_ar
        
        #normalise distances w.r.t v*t for each energy sampled times combination
        sampled_sim[...,0]-=self.z_init
        sampled_sim[...,0]/=norm
        
        #print(np.shape(sampled_sim))
        #concatenate along the last axis to format for histo2D_laxis
        #sampled_sim_conc = np.concatenate((sampled_sim[...,0],sampled_sim[...,1]),axis=-1)
        #creating the normalised z_beds (always between 0 and 1)
        z_normbeds = np.linspace(0,1,100)
        #print(np.shape(sampled_sim_conc))
        
        H2D = self.histo2D_laxis(sampled_sim,z_normbeds,mu_beds)
        #form the 2D histogram
        
        ##normalising the H2D
        z_delta = z_normbeds[-1]-z_normbeds[-2]
        mu_delta = mu_beds[-1]-mu_beds[-2]
        area=z_delta*mu_delta
        H2D/=self.Np
        H2D/=area
        
       # raise SystemExit
        
        return z_normbeds,H2D
    
    def make_Dmumu_plots(self,sim_funcs):
        diffusion = sim_funcs.return_D_mumu_funcs()[0]
        ms=1000
        mus = np.linspace(-1,1,ms)
        D_mumus=np.empty((self.n_pairs,ms))
        
        for p in range(self.n_pairs):
            L_n=self.mfp0_vals[p,0]
            L_p=self.mfp0_vals[p,1]
            bools = mus>0
            for m in range(ms):
                boole = bools[m]
                D_mumus[p,m]=diffusion(mus[m],1,(1-boole)*L_n+boole*L_p)
        return mus,D_mumus
   
    def characterise_omni(self,t_bins_minutes,omni_counts):
        
        #vector array to hold the time profile charecterisers, as we did for the events
        #second entry in the 2 axis holds [t_peak,t_r,t_d,n/a] (and is not identical over energy obiously) last entry is null
        #first entry in the 2 axis holds [kappa,alph,l_n,l_p] and is identical over energy (for convenience)
        #at the same time, create a similarly shaped array that holds the [kappa,alpha,l_n,l_p] tuple!
        sim_omni_chars_data = np.empty((self.n_alph,self.n_pairs,self.n_ener,4,2))
        
        
        #loop through each simulation and extract the tprofile characteristics of each
        for alph in range(self.n_alph):
            alpha = self.alpha_vals[alph]
            for l in range(self.n_pairs):
                l_n = self.mfp0_vals[l,0]
                l_p = self.mfp0_vals[l,1]
                
                #create the tuple
                var_tup=np.array([self.kappa,alpha,l_n,l_p])
                #insert the tuple into the multidim array
                sim_omni_chars_data[alph,l,:,:,0]=var_tup
                for e in range(self.n_ener):

                    fluxs = omni_counts[alph,l,e]

                    try:
                        #find the index of the raw maximum
                        #print(fluxs)
                        max_index = funcs.find_index_1D(fluxs,1.0) 
                        
                        #place in try block, in case no particles reached the virtual detector
                        
                        #finding the time of the maximum (in seconds since injection)
                        t_max =t_bins_minutes[max_index]*60
                        #find half the (peak normalised) maximum flux
                        thresh_flux = self.thresh
                        #find the index corresponding to the half max flux BEFORE the peak
                        ind_half_max_pre = funcs.find_index_1D(fluxs[:max_index],thresh_flux)
                        #find t_o in minutes, and make a conversion to seconds
                        t_r_since_inj = t_bins_minutes[ind_half_max_pre]*60

                        #similarly for post the peak
                        ind_half_max_post = funcs.find_index_1D(fluxs[max_index:],thresh_flux)
                        #finding the 'decay' times
                        t_d_since_inj = t_bins_minutes[max_index+ind_half_max_post]*60
                
                        #forming the pair of half width at half maximums
                        t_peak=t_max
                        t_r = (t_peak-t_r_since_inj)
                        t_d = (t_d_since_inj-t_peak)
                 
                    #if no particles reached the detector, pad with zeros
                    except:
                        t_peak,t_r,t_d=[0,0,0]
                    
                    #placing in the time profile charecterisers into a large array
                    sim_omni_chars_data[alph,l,e,:,1]=t_peak,t_r,t_d,0
                
        #raise SystemExit
        #for this particular value of kappa, save sim_omni_chars (the full array!) into the data folder with a special name
        #this will be loaded in the fitting class
        #print(fname)
        data_path=funcs.folder_functions().return_datapath()
        #saving the numeric char values
        fname = 'sim_omni_chars_data_kappa='+str(self.kappa)
        #print(container)
        #pack into one array so to not confuse them in unloading in the fitting class!
        np.save(os.path.join(data_path,fname),sim_omni_chars_data)
    
        return sim_omni_chars_data
    '''
    split sort simulation into seperate functions so to seperate them for debugging
    '''
    #sort the simulation array (sim_zmu), and sort into the necessary data
    def sort_simulation(self,sim_zmu):
        #loading in the carry over array
        carry_over=funcs.load_carry_over()        
        '''
        extracting the data from the carry_over dictionary
        '''
        #extracting info from carry_over
        max_steps=carry_over['max_steps']
        v=carry_over['velocities[AU/day]']
        dt_vals=carry_over['dt_vals[days]']
        sim_funcs=carry_over['sim_funcs']

        '''
        finding ballistic time of arrival
        '''
        bal_arrival_times=self.find_bal_times(v)
        
        '''
        simulating electron distributions at z_obs
        '''
        t_bins_minutes,mu_beds_pads,pads,omni_counts,omni_counts_ers,ang_beds_pads=self.simulate_electron_fluxes(sim_zmu,dt_vals)


        ##so we now have t_bins_minutes,mu_beds,pads,and omnicounts!
        
        '''
        find the time profile omnidirectional charecterisation quantities t_o,t_d,HWHM_premax,HWHM_postmax for each simulation and save in a large array
        '''
        
        sim_omni_chars_data = self.characterise_omni(t_bins_minutes,omni_counts)
        
    
        '''
        sampling the simulation at equidistant time steps
        '''
        sampled_sim,sample_ts = self.sample_sim(sim_zmu,max_steps,dt_vals)
        #print(np.shape(sample_ts))

        
        '''
        performing z-mu sampling at each sample time
        '''
        
        z_beds,h_z_num,ers_z,mu_beds,h_mu_num,ers_mu=self.z_and_mu_sampling(sampled_sim)
            
        '''
        2D z-mu binning
        '''
        z_normbeds,H2D = self.zmu_sampling(sampled_sim,sample_ts,v,mu_beds)
        
        
        '''
        building the diffusion coefficients (for plotting)
        '''
        mus,D_mumus=self.make_Dmumu_plots(sim_funcs)
        
    
        '''
        saving each sorted data with respective file name
        '''
        print('Saving sorted data.')
        data_path=funcs.folder_functions().return_datapath()
        for alph in range(self.n_alph):
            alpha = self.alpha_vals[alph]
            for l in range(self.n_pairs):
                l_n = self.mfp0_vals[l,0]
                l_p = self.mfp0_vals[l,1]
                fname = funcs.file_name_asym(alpha,l_n,l_p,self.kappa)
                
                # saving the histogrammed z distribution,bins and poisson errors
                np.save(os.path.join(data_path,"h_z_"+fname),h_z_num[alph,l])
                np.save(os.path.join(data_path,"ers_z_"+fname),ers_z[alph,l])
                #np.save(os.path.join(data_path,"beds_z_"+fname),z_beds)
                
                #similarly for mu
                np.save(os.path.join(data_path,"h_mu_"+fname),h_mu_num[alph,l])
                np.save(os.path.join(data_path,"ers_mu_"+fname),ers_mu[alph,l])
                
                #saving the 2D Histogram data (already saved mu_beds)
                np.save(os.path.join(data_path,"hist2D_"+fname),H2D[alph,l])
                #saving the sample times of the z-mu samplings
                #print("sampled_ts_"+fname)
                np.save(os.path.join(data_path,"sampled_ts_"+fname),sample_ts[l])

        
                #saving D_mumu (constant across alpha)
                np.save(os.path.join(data_path,"mu_toplot_"+fname),mus)
                np.save(os.path.join(data_path,"D_mumu_toplot_"+fname),D_mumus[l])
                
                #saving pads, omni fluxes and the time beds for plotting
                np.save(os.path.join(data_path,"pads_"+fname),pads[alph,l])
                np.save(os.path.join(data_path,"ang_beds_pads_"+fname),ang_beds_pads)
                np.save(os.path.join(data_path,"t_bins_minutes_"+fname),t_bins_minutes)

                np.save(os.path.join(data_path,"omni_counts_"+fname),omni_counts[alph,l])
                np.save(os.path.join(data_path,"omni_counts_ers_"+fname),omni_counts_ers[alph,l])
                
                #saving z_obs
                np.save(os.path.join(data_path,"z_obs_"+fname),self.z_obs)
                np.save(os.path.join(data_path,"z_tol_"+fname),self.z_tol)
                
    
                #save z_beds as this will be modified for other plotting with removal of the bin
                np.save(os.path.join(data_path,"z_beds_"+fname),z_beds)
        
                
                #saving the number of phase space samples
                np.save(os.path.join(data_path,"M_"+fname),self.M)
                
                #saving the ballistic arrival times
                np.save(os.path.join(data_path,"bal_arrival_times_"+fname),bal_arrival_times)
                
                        
                #saving the normalised z beds
                np.save(os.path.join(data_path,"z_normbeds_"+fname),z_normbeds)
                #saving the linearly spaced mu beds
                np.save(os.path.join(data_path,"mu_beds_"+fname),mu_beds)
                #saving the pitch angle distribution mu beds
                np.save(os.path.join(data_path,"mu_beds_pads_"+fname),mu_beds_pads)
                
                
                '''
                for now, omit these... not sure what I need sim_omni_
                '''
                #saving the temporal charecteristics in an array
                #np.save(os.path.join(data_path,"t_bins_inj_"+fname),t_bins_inj,allow_pickle=True)
                #save as well sim_omni_chars in the usual fname format for plotting
                np.save(os.path.join(data_path,"sim_omni_chars_"+fname),sim_omni_chars_data[alph,l,:,:,:])
                
                #saving the analytical solution dictionary
        #build the t_bins_inj, with the now saved t_bins_minutes for each simulation
        funcs.update_t_bins_inj()
        print('Sorting Complete. ')
        
        return None
            
            
    
            
    
    
    
    
    
    
    
