 # -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:41:28 2022

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

import warnings

import matplotlib.pyplot as plt



'''
-this class takes in the dictionary of variables needed for each script and unpacks it from the global dictionary
-takes in the dictionary of (primary) functions needed to run the script
'''

class user_query:
    def __init__(self,dicts,funcs_dict):
        self.options=['resimulate','sort','plot','fit event','delete data','exit']
        
        self.sim_dict=dicts['sim_dict']
        self.sort_dict=dicts['sort_dict']
        self.plot_dict=dicts['plot_dict']
             
        self.sim_func = funcs_dict['sim_func']
        self.sort_func=funcs_dict['sort_func']
        self.plot_func=funcs_dict['plot_func']
        self.solution_func=funcs_dict['solution_func']
        self.fit_func= funcs_dict['WIND_fitting_func']     
        
    '''
    general user prompt functions
    '''
    
    #general request action function
    def request_action(self):
        print('Options are: '+str(self.options))
        action = input('What would you like to do? ')
        while not action in self.options:
            print('Please use a valid input: '+str(self.options))
            action = input("Would you like to simulate,sort or plot? ")
        return action
    
    
    #prompts a warning and returns either 'Y' or 'N'
    def warning(self,action):
        options_YN=['Y','N']
        do = input('Are you sure you want to '+action+'? This can overwrite existing data. (Y/N) ')
        while not do in options_YN:
            print('Please use a valid input: '+str(options_YN))
            do = input('Are you sure you want to '+action+'? ')
        return do
    
    '''
    perform either the simulation promp
    '''
    
    #request user if they want to simulate the transport, if yes returns the simulation array and variables that carry over to sim_sorter
    #this routine runs if sim_zmu is not already defined
    def simulate_request(self):
        options_YNPass=['Y','N','Pass']
        prompt_sim=input('Would you like to run the simulation with the requested inputs? (Y/N/Pass) ')
        
        if not prompt_sim in options_YNPass:
            print('Please use a valid input: '+str(options_YNPass))
            prompt_sim=input('Would you like to run the simulation with the requested inputs? ')
            
        if prompt_sim=='Y':
            return self.sim_func()
        
        if prompt_sim=='N':
            raise SystemError('Simulation Cancelled.')
            
        if prompt_sim=='Pass':
            warnings.warn("Warning, cannot sort with no simulation in memory!")
            return None,None
        
        
        
            
    #define the function that runs the simulation with the given dictionary as input WITH A WARNING!
    def do_sim_warning(self,action):
        #prompts a warning
        do=self.warning(action)
        #if yes, simulate the transport, using the parameters of sim_dicto
        #output the simulation array, and variables to carry over from simulation to sorting
        if do=='Y':  
            sim_zmu=self.sim_func()
        #if not, cancel the simulation
        if do=='N':
            raise SystemError('Simulation cancelled.')
        return sim_zmu
    
    #perform the sort, using the simulation array and carry over variables stored in memory
    def do_sort_warning(self,action,sim_zmu):
        #prompt the warning
        do=self.warning(action)
        #if Yes, perform the sort
        if do=='Y':  
            self.sort_func(sim_zmu)
        #if no, cancel the sorting
        if do=='N':
            raise SystemError('Sorting cancelled.')
        return None
    
    #runs the plotting function, with the plotting variables and some of the simulation input variables as input
    #before plotting constructs the requested analytical_solution
    def do_plot(self):
        #update t_bins_inj with the loaded event
        funcs.update_t_bins_inj()
        #build the analytical solutions [tar,sol_t,zar,sol_z]
        sols = self.solution_func()
        #carrying them into the plot_func to plot
        self.plot_func(sols)
        return None
    
    def do_fit(self):
        self.fit_func()
        return None
    
    def do_delete_data(self,action):
        do=self.warning(action)    
        if do=='Y':
            funcs.delete_data()
        else:
            raise SystemError('Deleting cancelled.')

            
    '''
    If simulation array is already prepared, we have the general user request to resimulate, sort or plot
    '''
    #if sim_zmu already exists, further prompt to resimulate, sort or plot.
    def user_request(self,sim_zmu):
        #defining the avaialble user options 
        #request one of the above actions, neglecting other actions
        #every time ust 
        action=self.request_action()  
        while action!='exit':
            #if the user wishes to re-simulate
            if action == 'resimulate':
                #input the action, the describing variables needed for the function, and the available options
                sim_zmu=self.do_sim_warning(action)
                #then ask what to do again
                action=self.request_action()
            #if the user wishes to sort the preexisting simulation data.
            if action=='sort':
                self.do_sort_warning(action,sim_zmu)
                action=self.request_action()
                
            if action=='fit event':
                self.do_fit()
                action=self.request_action()
                
            if action=='plot':  
                #just plot with no warnings
                self.do_plot()
                action=self.request_action()
            if action=='delete data':
                self.do_delete_data(action)
                action=self.request_action()

        if action=='exit':
            return sim_zmu
        

        



        