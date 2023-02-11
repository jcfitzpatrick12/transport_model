#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:07:02 2022

@author: jimmyfitpatrick

general simulation encomapssing
-asymmetric + symmetric scattering
-uniform non-instantaneous injection
-arbitrary form of diffusion coefficient
-focused transport (if requested)

"""

import numpy as np
import os
from scipy import constants
from scipy import special
import sys
from pathlib import Path
    
#importing the global functions file
import util.global_functions as funcs

'''
Class for extracting the D_mumu functions and mean free path functions
'''

#class that holds functions regarding the mean free path and 
class SIM_funcs:
    def __init__(self, D_mumu_func,mfp_bool,h_val):
      self.D_mumu_func = D_mumu_func
      self.mfp_bool=mfp_bool
      self.h_val = h_val
      
    def return_mfp_func(self):
        #constant meanfree path if the boolean is true
        if self.mfp_bool==True:
            def meanfreepath(p,z,lamda,alph,kappa):
                mfp=lamda
                return mfp
            return meanfreepath
        #use the general form otherwise
        #takes in the ALREADY NORMALISED TO THE LOWEST MOMEMENTUM momentum
        if self.mfp_bool==False:
            def meanfreepath(p,z,lamda,alph,kappa):
                # z coming in au/day momentum already pre-normalised to lowest particle momentum
                mfp= lamda*((np.abs(z/1.2))**kappa)*((p)**(2*alph))
                return mfp
            return meanfreepath
      
    #function that returns the D_mumu and dD_mumu functions
    def return_D_mumu_funcs(self):
        #defining the constant meanfreepath diffusion coefficient
        if self.D_mumu_func=='constant':
            def D_mumu(mu,v,lamda): 
                D = v/(lamda)*mu/mu
                return D
            def dD_mumu(mu,v,lamda):
                dD = 0*mu/mu
                return dD
            return D_mumu,dD_mumu
        
       
        #defining the standard quasilinear diffusion coefficient
        if self.D_mumu_func=='quasilinear':
            def D_mumu(mu,v,lamda): 
                h=self.h_val
                q=5/3
                D = ((3*v)/(lamda*(4-q)*(2-q)*2))*(1-np.power(mu,2))*(np.power(np.abs(mu),q-1)+h)
                return D
            def dD_mumu(mu,v,lamda):
                h=self.h_val
                q=5/3
                A = (3*v)/(2*lamda*(4-q)*(2-q))
                B = 2*mu*(1-np.power(mu,2))
                C = 6*mu*np.power(np.abs(mu),q)*np.power(np.abs(mu),1/3)
                D = 6*mu*h*np.abs(mu)*np.power(np.abs(mu),1/3)
                E = 3*np.abs(mu)*np.power(np.abs(mu),1/3)
                dD = A*((B-C-D)/E)
                #dD = A*(2*mu*(1-mu**2)-6*mu*abs(mu)**(q)*abs(mu)**(1/3)-6*mu*h*abs(mu)*abs(mu)**(1/3))/(3*abs(mu)*abs(mu)**(1/3))
                return dD
            return D_mumu,dD_mumu
        
        #defining the standard isotropic diffusion coefficient
        if self.D_mumu_func=='isotropic':
              def D_mumu(mu,v,lamda): 
                  D = (v/(2*lamda))*(1-mu**2)
                  return D
        
              #and its respective derivative
              def dD_mumu(mu,v,lamda):
                  dD = -(v*mu)/lamda
                  return dD
              return D_mumu,dD_mumu


#primary simulating class
class transport_functions:
    def __init__(self,dicts):
        self.sim_dict=dicts['sim_dict']
        #unpacking the variables of sim_dict
        #unpacking the variables
        self.Np=self.sim_dict['num_particles']
        self.alpha_vals=np.array(self.sim_dict['alpha_vals'])
        self.kappa=self.sim_dict['kappa']
        self.mfp0_vals = np.array(self.sim_dict['mfp0_vals[AU]'])
        self.ee=np.array(self.sim_dict['energies[keV]'])
        self.t_end = self.sim_dict['t_end[days]']
        self.inj_set=self.sim_dict['injection_type']
        self.custom_end = self.sim_dict['custom_end[days]']
        self.z_init=self.sim_dict['z_injection[AU]']
        self.mu_IC_set=self.sim_dict['mu_IC']
        self.D_mumu_set=self.sim_dict['D_mumu_type']
        self.mfp_const=self.sim_dict['mfp_constant']
        self.consider_focusing=self.sim_dict['consider_focusing']
        self.h_val = self.sim_dict['h_val']
        
        
        #extracting the requested temporal resolution, so to compare against
        self.sort_dict=dicts['sort_dict']
        self.requested_t_binwidth = self.sort_dict['t_binwidth[s]']
        
    #returns speed of an electron given its energy (input in keV)
    def energy_vel(self,E):
        m_e = 9.10938356e-31
        # Converting keV to Joules
        E = 1.60218e-16*E
        # Returns velocity in au/day
        return constants.c*np.sqrt(1-((m_e**(2)*constants.c**(4))/(E+m_e*constants.c**2)**2))*5.77548e-7
    
    # returns relativistic momentum of an electron, given particle velocity in au/day
    def rel_momentum(self,v):
        m_e = 9.10938356e-31
        # Returns momentum in units, where v is au/day and mass is kg
        return self.gamma(v)*m_e*(v)
    
    #returns relativistic gamma factor, given particle velocity in au/day
    def gamma(self,v):
        return (1-(v/(constants.c*5.77548e-7))**2)**(-1/2)
    
    ## function that returns the simulation timestep (based on diffusive timescale)
    def finddt(self,l,v):
        return (l/v)/100
    
    #return the index of a value in an 1Darray closest to the input value
    def return_mindex(self,ar,val):
        ar = abs(ar-val)
        minim = np.min(ar)
        ind = np.where(ar==minim)
        return ind[0][0]

    #defining the constant injection function
    def const_inj_function(self,size,dt_spec,t1,t2):
        #constructing the time array for the given dt for a specific energy channel
        t = np.arange(0,size,1)*dt_spec
        #tau1 and tau2 hold the index of the elements that contains times closest to t1 and t2
        tau1 = self.return_mindex(t,t1)
        tau2 = self.return_mindex(t,t2)
        #counting how many time steps between the two times 
        N = tau2-tau1
        #stopping singularity if two points fall on coincident timesteps
        if N==0:
            N=1
        #number of electrons to inject per timestep within the bounds
        n_perstep = self.Np/N
        frac=1
        #if n_perstep is more than 1 just set frac=1
        if n_perstep<1:#if it is less than one
            #if n_perstep is less than 1 then we will need to 'skip' steps
            print('skipping steps!')
            frac=int(np.floor(1/n_perstep))
        #print(n_perstep)
        #this array will hold the number of psuedo-particles to inject at time t (w.r.t the default time array)
        inj = np.zeros(len(t))
        #for each timestep,
        for tau in range(len(t)):
            #if the time falls within the injection period,
            if tau>=tau1 and tau<=tau2:
                #extra line that deals with if n_perstep<1
                boole = tau%frac==0
                #inject relevent number of particles on appropriate timesteps
                inj[tau]=np.ceil(n_perstep)*boole
            else:#otherwise inject no particles
                inj[tau]=0
        #return the time array and injection function
        return t,inj
    
    def return_boole_asym(self,particles_insim,inj_function,tau,tup):
        #label each particle in the simulation from 0,Np based on which index it lies on
        ind = np.ones(tup)
        Np=tup[-1]
        counting = np.arange(0,Np,1)
        ind[:,:,:]=counting

        #add relevent injected particles into the simulation
        particles_insim = particles_insim+inj_function[:,:,:,tau]
        
        #the compare array will be a boolean type array that will release the particle if index<particles_insim
        #this is mediated by the injection function
        compare = np.zeros(np.shape(ind))
        #pull Np axis to the front to place in relevent injected particle numbers
        compare = np.moveaxis(compare,-1,0)
        compare[:,:,:,:] = particles_insim
        #return compare array to shape before 
        compare = np.moveaxis(compare,0,-1)
        #release if the index of each particle is less than the compare
        release = np.where(ind<compare)
        #if the particle has been released, set to one
        ind[release]=1
        #if particle has not been released yet, set to 0
        ind[ind!=1]=0
        #emphasis that ind has become the boolean array we needed
        boole = ind
        return  boole,particles_insim
    
    #function for asymmetric diffusion
    #first two arguments are functions
    def asym_diffusion(self,diffusion,ddiffusion,mu,v,l_n,l_p): 
        #implicately assuming that no particle has mu exactly zero
        #need to apply boolean logic
        #mu is an array
        #arrays that are the same shape as mu, and hold true for the elements satisfying the conditions, and false if not
        true_pos = mu>0
        #true_neg will be inferred from ~true_pos
        true_zero = mu==0
        #converting true_zeroes to nans to not interfere with the decomposition into positve and negative values
        mu[true_zero] = np.nan

        #decomposing the array into positive and negative parts
        #i.e. here, all non-positive entries will be made to be 0 after the multiplication
        pos_mu = true_pos*mu
        #converting all particles with non-positive mu values to nans 
        pos_mu[pos_mu==0]=np.nan
        
        #same for negative mu values
        neg_mu = ~true_pos*mu
        neg_mu[neg_mu==0]=np.nan

        #calling in existing diffusion function, b
        #to allow for antisymmetric diffusion coefficient
        #positive mu values use positive mean free path
        D_p = diffusion(pos_mu,v,l_p)
        #vice versa for negative diffusion values
        D_n = diffusion(neg_mu,v,l_n)

        #finding the nanvalues  after computation and turning back to zeros
        true_nan_pos_D = np.isnan(D_p)
        true_nan_neg_D = np.isnan(D_n)

        #converting these arrays back to 0
        D_p[true_nan_pos_D] = 0
        D_n[true_nan_neg_D] = 0
        
        #identically byt for the derivative
        dD_p = ddiffusion(pos_mu,v,l_p)
        #vice versa for negative diffusion values
        dD_n = ddiffusion(neg_mu,v,l_n)
        
        true_nan_pos_dD = np.isnan(dD_p)
        true_nan_neg_dD = np.isnan(dD_n)
        
        dD_p[true_nan_pos_dD] = 0
        dD_n[true_nan_neg_dD] = 0

        #recompose to form the entire D_mumu array
        D = D_p + D_n
        dD=dD_p+dD_n
        return D,dD
    
    '''
    correcting mu functions via additive boundaries
    '''
    
    '''
    #Boolean arrays for correcting mu vals
    # Returns a boolean array which says true if any vals are above 1
    #takes in the array of mu values for each test particle
    def maxbound(self,muvals):
        trueif = muvals>1
        return trueif

    # Returns a boolean array which says true if any mu vals are below -1
    def minbound(self,muvals):
        trueif = muvals<-1
        #trueif = muvals<0
        return trueif
    # Function that returns True if any of the boolean values in a multi dim array are true, False if there are no trues
    def has_true(self,arr):
        return arr.any()
    '''
    '''
    #should be able to optimise this.
    # Function that corrects mu values by rules of reflective boundaries (-1,1)
    def mucor(self,muvals):
        k=0
        # a returns True for all elements maxbound, and b analogously for elements below 0
        #a = self.maxbound(muvals)
        a=muvals>1
        #b = self.minbound(muvals)
        b=muvals<-1
        while a.any()==True or b.any()==True:
            #debugging loop to stop corrections applying indefinitely for a failsafe
    #         if k>1000:
    #             #print(muvals[a])
    #             #print(muvals[b])
    #             print("too many iterations, stopping")
    #             break  
    #       # applying corrections if any boolean values are true for below 1
            #if b.any() == True:
            muvals[b]= -1+abs(muvals[b]+1)
              
            #if a.any() == True:
            muvals[a]= 1-abs(muvals[a]-1)

            # After a single correction, seeing if any further corrections are necessary. Loops until all values are within range or sequence breaks
            #a = self.maxbound(muvals)
            #b = self.minbound(muvals)
            a=muvals>1
            #b = self.minbound(muvals)
            b=muvals<-1
            # Applying first "above 1" corrections if any boolean values are true for max bound
            k=k+1
        #once corrections are made, return array of mu values
        return muvals
    
    '''
    # Function that corrects mu values by rules of reflective boundaries (-1,1)
    def mucor(self,muvals):
        a=muvals>1
        b=muvals<-1
        while a.any()==True or b.any()==True:
            muvals[b]= -1+abs(muvals[b]+1)
            muvals[a]= 1-abs(muvals[a]-1)
            a=muvals>1
            b=muvals<-1
        #once corrections are made, return array of mu values
        return muvals
    
    def mucor_faster(self,muvals):
        mu_toobig=muvals>=1
        mu_toosmall=muvals<=-1
        #try correct the correctable mu values
        muvals[mu_toobig]= 1-muvals[mu_toobig]%1
        muvals[mu_toosmall]= -1-muvals[mu_toosmall]%-1
        return muvals

        

    #takes in the requested simulation, and outputs the particle simulation array
    def simulate_transport(self):      
        '''
        number of alpha_vals, mean free path pairs and energies being simulated
        '''
        
        #number of meanfreepath pairs
        n_pairs = np.shape(self.mfp0_vals)[0]
        #number of alpha values considered
        n_alph = len(self.alpha_vals)
        #how many energy channels are we considering?              
        n_ener = len(self.ee)
        
        
        sim_tup = (n_alph,n_pairs,n_ener,self.Np)
        
        
        '''
        building paramater arrays for the simulation
        '''
        # To define the mean free path dependent on distance we need to find the momentum of the lowest energy considered
        #finding minimum energy
        lowest_ee = np.min(self.ee)
        #finding minimum velocity [in au/day]
        lowest_vel = self.energy_vel(lowest_ee)
        #finding minimum momementum
        p0=self.rel_momentum(lowest_vel)
        ## Initialising the particle velocities
        #recall, ee holds the energy channels as floats
        v = self.energy_vel(self.ee)
        
        #finding an array of the relativistic momentums, given the particle velocities...
        ## Corresponding momentums for each energy channel
        p = self.rel_momentum(v)

        
        
        '''
        initialising and vectorising simulation parameters
        '''
        
        #reformatting arrays for vector computation
        #reformatting velocities, momentums, mean free path respectively
        vels = np.zeros(sim_tup)
        moms = np.zeros(sim_tup)
        #place v and p into formatted arrays
        for e in range(n_ener):
            vels[:,:,e,:] = v[e] 
            moms[:,:,e,:] = p[e]
            
        #normalising the momentum w.r.t momentum of lowest energy channel.
        mom = moms/p0
        
        #initialising an array to hold the mean_free_paths
        #by convention, first entry on last axis is mfp_n, second is mfp_p
        mfp_ar = np.zeros(sim_tup+(2,))
        # dt_vals will be an intermediate array to help initialise the full timestep array
        #it will hold the timestep for each lambda_parallel,oplus, energy channel combination
        #it is constant across alpha
        dt_vals=np.zeros((n_pairs,n_ener))
        #for each mean_freepath, compute the corresponding  the corresponding timestep
        for l in range(0,n_pairs):
            #extracting the two values of the mean free path
            L_n= self.mfp0_vals[l,0]
            L_p = self.mfp0_vals[l,1]
            #set all the elements in the [l,:,:] entries to these respective values
            mfp_ar[:,l,:,:,0]=L_n
            mfp_ar[:,l,:,:,1]=L_p
            # here we will take minimum mean free path of the set to define the stepsize
            L_min = min(L_n,L_p)
            #finding the diffusive timescale for the L_min v combination
            dt_vals[l,:]=self.finddt(L_min,v)


        #finding the minimum timestep in seconds
        max_dt_seconds = np.max(dt_vals)*86400
        
        if self.requested_t_binwidth<max_dt_seconds:
            raise SystemError('The maximum timesteps in seconds are: '+ str(max_dt_seconds)+' but the requested temporal resolution is: '+str(self.requested_t_binwidth)+'. Increase the requested temporal resolution! Timesteps are too large in some or all of the energies simulated.')

        #full array initialising for vectorising computations
        dt = np.zeros(sim_tup)
        for n in range(self.Np):
            dt[:,:,:,n]=dt_vals
            
        #initialising the alpha vector format array
        alphas = np.zeros(sim_tup)
        for alph in range(n_alph):
            alphas[alph,:,:,:] = self.alpha_vals[alph]
            
        '''
        setting up the primary simulation arrays and related variables
        '''
        #we need to make sure the smallest timestep makes it to t_end
        #number of timesteps for each dt
        n_timesteps = self.t_end/dt
        
        #what is the maximum number of timesteps? i.e the most timesteps needed such that
        #the combination with the smallest timestep makes it to t_end
        max_steps = int(np.max(n_timesteps)+1)
        
        #in the last axis, 0 is the z vals, 1 is the mu vals
        #sim_zmu will hold the position,z, and pitch angle, mu, of each particles in all the simulations in all energy channels
        sim_zmu = np.zeros(sim_tup+(max_steps+1,2))
        
        ## All particles start at z=0.05
        sim_zmu[:,:,:,:,0,0] = self.z_init
        #print(self.z_init)
        
        #picking the initial mu distribution
        if self.mu_IC_set =='uniform+':
            mu_init = np.random.uniform(0,1,sim_tup)
        
        if self.mu_IC_set =='uniform+-':
            mu_init = np.random.uniform(-1,1,sim_tup)
            
        if self.mu_IC_set == 'forward_beamed':
            mu_init = np.random.uniform(-1,1,sim_tup)*1e-4+0.999
        
        
        #placing initial mu distribution into the main simulation array
        sim_zmu[...,0,1] = mu_init[:,:,:,:]
        
        # Construct a ballistic particle that will be UNAFFECTED by scattering to test arrival time
        sim_test = np.zeros((sim_tup[0:-1])+(max_steps+1,))
        #independent of mu naturally
        sim_test[:,:,:,0]=self.z_init
        
        #raise SystemExit
        
        #array to keep track of time for each simulation combination
        t = np.zeros(sim_tup)
        
        '''
        Accounting for non-instantateous injection!
        '''
        #setting the injection boundaries
        if self.inj_set=='instantaneous':
            self.inj_range = [0,self.t_end/10000]
         
        if self.inj_set=='constant':
            self.inj_range=[0,self.t_end]
        
        if self.inj_set=='custom':
            inj_range=[0,self.custom_end]
         
            
        #hold particles at z=init, at each timestep 'release' N particles according to inj function
        #seperate injection functions for mfp,ee combinations (since dt varies for these parameters)
        inj_function = np.zeros((sim_tup[0:-1])+(max_steps+1,))
        t_inj = np.zeros((sim_tup[0:-1])+(max_steps+1,))
        #size of the time array is max_steps+1
        size = max_steps+1
        
        #extracting the injection boundaries
        t1=self.inj_range[0]
        t2=self.inj_range[1]
        #constructing each injection function, for every l and e combination
        #these are the variables which affect timestep size
        for l in range(n_pairs):
            for e in range(n_ener):
                #pulling out the specific timestep for that combination
                dt_spec = dt[0,l,e,0]
                #construct the time arrays(for plotting) and the injection function array
                t_inj[:,l,e],inj_function[:,l,e]=self.const_inj_function(size,dt_spec, t1, t2)
        
        
        #at tau=0 no particles are in the simulation
        #particles_insim will hold how many particles have been injected into the sim each timestep, initially zero
        particles_insim = 0*inj_function[:,:,:,0]
        
        
        '''
        selecting the chosen form of D_mumu and mean free path
        '''
        
        sim_funcs=SIM_funcs(self.D_mumu_set,self.mfp_const,self.h_val)
        diffusion = sim_funcs.return_D_mumu_funcs()[0]
        ddiffusion=sim_funcs.return_D_mumu_funcs()[1]
        meanfreepath_func = sim_funcs.return_mfp_func()
        
        #if consider focusing...
        if self.consider_focusing==True:
            B=1
        if self.consider_focusing==False:
            B=0
            
        
        '''
        Main simulation loop
        -for speed, define the local variables for functions before the main loop
        '''
        
        # t must also now be an array
        print("Running simulation for "+str(self.Np)+" particles.")
        particles_insim = 0*inj_function[:,:,:,0]
        #tau=0
        
        '''
        de-selfing some of the parameters to be used in the loop
        '''
        #scalar
        t_end = self.t_end
        #function
        return_boole_asym = self.return_boole_asym
        #scalar
        kappa=self.kappa
        #function
        asym_diffusion=self.asym_diffusion
        #function
        mucor=self.mucor_faster
        
        '''
        main simulation loop
        '''
        tau=0
        #so incremement tau then use tau-1 elements to define tau
        for n in range(1,max_steps+1):
        #while (np.min(t) < t_end):
            #timekeeping
            if tau%10==0:
                print(str(np.round(100*tau/max_steps+1,2))+'%')
                #print(np.min(t)*1440)
                
            #construct the boolean array and inject particles into the sim using inj_function
            boole,particles_insim = return_boole_asym(particles_insim, inj_function, tau,sim_tup)
            #incremeent by one timestep
            tau=tau+1
            # Increment time by one timestep (t and dt both multdim arrays)   
            t=t+dt  
            #pull out a random variable from a normal distribution centred at the origin variance 1
            W = np.random.normal(0,1,sim_tup)  
            
            #l_n is applied for particles with mu<0
            #l_p will be applied to particles with mu>0
            # finding mean free path for mu<0  
            l_n = meanfreepath_func(mom,sim_zmu[:,:,:,:,tau-1,0],mfp_ar[:,:,:,:,0],alphas,kappa)
            #finding the meanfreepath for mu>0 
            l_p = meanfreepath_func(mom,sim_zmu[:,:,:,:,tau-1,0],mfp_ar[:,:,:,:,1],alphas,kappa)
            
            #define new focusing length 
            L_z = sim_zmu[:,:,:,:,tau-1,0]/2
        
            ## Finding the (now asymmetric) diffusion coefficients (splits into negative and positive particles and applies diffusion formulae with the respective mfps)
            D,dD=asym_diffusion(diffusion,ddiffusion,sim_zmu[:,:,:,:,tau-1,1],vels,l_n,l_p)
            

            #simple ballisitc current dist+vels*dt
            sim_test[:,:,:,tau] = sim_test[:,:,:,tau-1]+vels[:,:,:,0]*dt[:,:,:,0]
            
            
            #computing the stochastic recurcions
        
            #advance particles using boole. If index of particle<sum of injected then boole =1 and particle z advances
            sim_zmu[:,:,:,:,tau,0] = sim_zmu[:,:,:,:,tau-1,0]+ boole*sim_zmu[:,:,:,:,tau-1,1]*vels*dt  
            #sim_zmu[:,:,:,:,tau,0] = sim_zmu[:,:,:,:,tau-1,0]+ vels*dt 
            
            #pitch angle is unaffected by the injection and always varies for all particles
            sim_zmu[:,:,:,:,tau,1] =  sim_zmu[:,:,:,:,tau-1,1] + dt*(dD+B*(vels*(1-sim_zmu[:,:,:,:,tau-1,1]**2))/(2*L_z))+np.sqrt(2*D*dt)*W
            #correct the zmu_values in order to keep within range using the mucor function
            sim_zmu[:,:,:,:,tau,1] = mucor(sim_zmu[:,:,:,:,tau,1])
            
        #after simulating, collect some values needed for sorting and place in a dictionary
        carry_over = [v,dt_vals,max_steps,sim_funcs]
        carry_over_labels = ['velocities[AU/day]','dt_vals[days]','max_steps','sim_funcs']
        carry_over=funcs.build_dict(carry_over_labels,carry_over)
        
        #rather than unpacking as part of the function, numpy save the carry over array
        data_path=funcs.folder_functions().return_datapath()
        np.save(os.path.join(data_path,"carry_over"),carry_over,allow_pickle=True)
        np.save(os.path.join(data_path,"sim_zmu"),sim_zmu)  
    
        #save the parameters that can be saved (prior to sorting!)
        print('Saving relevant variables.')
        data_path=funcs.folder_functions().return_datapath()
        for alph in range(n_alph):
            alpha = self.alpha_vals[alph]
            for l in range(n_pairs):
                l_n = self.mfp0_vals[l,0]
                l_p = self.mfp0_vals[l,1]
                fname = funcs.file_name_asym(alpha,l_n,l_p,self.kappa)
                #saving t_inj and (normalised) injection function
                np.save(os.path.join(data_path,"t_inj_"+fname),t_inj[alph,l])
                np.save(os.path.join(data_path,"inj_function_"+fname),inj_function[alph,l]/self.Np)
                
                #save the injection status and injection locations to remove the injection bin for that run if necessary
                np.save(os.path.join(data_path,"inj_set_"+fname),self.inj_set)
                
                #save the initial_z
                np.save(os.path.join(data_path,"z_init_"+fname),self.z_init)
                
                #save the electron energies
                np.save(os.path.join(data_path,"ee_"+fname),self.ee)
                
                #save whether or not constant mean free path was used (to plot the mean free path function)
                np.save(os.path.join(data_path,"mfp_const_"+fname),self.mfp_const)
                
        print('Simulation complete.')
        #return the simulation array and the variables to carry over for sorting
        #return sim_zmu,carry_over
        return sim_zmu
        
    
        
        
        
        
        
        