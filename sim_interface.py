# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:44:07 2022

@author: jcfit

TO RESOLVE:
-for analytical comparisons, change from peak normalising to area normalising ! (temporary fix, plotting 'just z' performs area normalisation
internal within the plotting function.)

"""
'''
importing various packages
'''

import numpy as np

'''
adding the project to the system path so to allow for importing transport scripts
importing the global functions file
'''

import sys
from pathlib import Path
import os

import matplotlib.pyplot as plt
#find the current directory path of the script (primary) script
dir_path = os.path.abspath(os.path.dirname(__file__))
if dir_path in sys.path:
    print('Package already in sys.path.')
else:
    print('Adding package to sys.path')
    sys.path.append(dir_path)
    
#importing the global functions file
import util.global_functions as funcs

import warnings

from datetime import datetime


'''
importing core-functionss
'''

#loading in the user request file
from util.user_requests import user_query

#loading in the transport simulation function
from util.main_simulation import transport_functions

#loading in the sorting functoin
from util.main_sorter import sorting_functions

#loading in the plotting function
from util.main_plotter import plot_functions

#loading in the analytical function
from util.analytical_solutions_builder import solution_functions


'''
importing  WIND functions
'''
#importing in the WIND classes
from util.process_WIND import WIND_functions
from util.event_class_WIND import WIND_events
from util.sort_WIND import sort_WIND_data

#loading in the event_fitting class
from util.event_fit_WIND import fit_WIND_event


'''
Input parameters here

Simulation is currently vectorised over alpha, mean free path pairs and energy
'''

#if sim_zmu is large, loading can take time, set to False and Pass to skip this.
load_existing_sim_zmu = False
# Defining the number of test particles in thde simulation
Np=40000
#choose the alpha_values (vectorised) NO ENERGY DEPENDENCE FOR CONSIDERATION OF ONE ENERGY
alpha_vals = [0.0]
#choosing kappa (non-vectorised, must be a scalar)
kappa=0.0
#first entries are \lamda- second entry is \lamda+. Holds mfp0 as in the attached document
#mfp0_vals = [[1.0,1.0],[1.25,1.25],[1.5,1.5],[1.75,1.75],[2.0,2.0],[2.25,2.25],[2.5,2.5],[2.75,2.75],[3.0,3.0],[3.25,3.25],[3.5,3.5]]
#mfp0_vals = [[30.0,1.25],[1.25,1.25]]
mfp0_vals = [[0.0005,0.05]]

## what electron energies to consider (in keV) (vectorised)res
#CONSTRAINT: must be a member of [27,40,67,110,180,310,520] 
#ee=[67,110]
#ee=[67,110,180]
ee = [27]
#value of h
h_val=0.0
#end time [d]s
t_end=0.1/24
#start injection at t=0 [d], end injection at t2 [d]
#select from options of injection duration
inj_options = ['instantaneous','constant','custom']
inj_set=inj_options[0]

#if custom will use custom end time as specified below (must be less than t_end)
custom_end = t_end

#setting injection distance [AU]
z_init = 0.05

#initial mu distribution
#isotropic+ is uniform ove
mu_IC_options = ['uniform+','uniform+-','forward_beamed']
#choose from above options
mu_IC_set = mu_IC_options[0]

#form of D_mumu
D_mumu_options = ['constant','isotropic','quasilinear']
#choose from above options
D_mumu_set = D_mumu_options[2]

#whether the mean free path varies with energy/distance as in doc or constant (given by mfp_vals)
mfp_const = True
#whether we will consider adiabatic focusing in the stochastic recursions
consider_focusing = False

'''
Input variables for sorting
'''
    
#number of samples to take (equidistant in time, but not necessarily equal across energy channels (since timeteps are different))
M = 4
#position of observation to sample the electron fluxes
z_obs = 0.1
z_tol=0.0005
#time bin widths for the electron fluxes (in seconds, must be larger than the timestep!)
t_binwidth = 10
#ADD z_beds to here! min and max bin edges to plot in the z and mu plots [zmin,zmax,spacing]
z_beds = np.linspace(z_init-0.0,z_init+0.2,100)
#z_beds = np.linspace(1.0,2.0,150)


'''
Input variables to plot
'''

###-###
###-###

#mean free path
mfp_toplot=[0.0005,0.05]
#use the imbalanced case here for the comparisons (first element is the symmetric, second is the asymmetric )
mfp_comp_toplot = [[10.0,10.0],[10.0,10.0]]
#alpha valu
alpha_toplot =0.0
#appa value to
kappa_toplot=0.0
#which energy channel (keV)\
energy_toplot = 27
#step to plot for zmu must satistfy 0<=step_toplot<=M
step_toplot=-1
#choice between 't_o,t_d,HWHM_premax, HWHM_postmax'
char_toplot = ['t_peak','t_r','t_d']
#choose which energies to plot for the omnidirectional characteristics!
energy_char_toplot= [27,40,110]
#which plots to plots
#plots_toplot = ['injection','diffusion coef','zmu','z and mu','electron flux','plot_omni_characteristic','plot_WIND_all_energy_comparison']

#plots_toplot = ['plot_all_simulations_one_energy','plot_WIND_one_energy_comparison','plot_all_omni_characteristic']
#plots_toplot =['diffusion coef','plot_omni_characteristics_imb_comparison','plot_WIND_one_energy_imb_comparison']
#plots_toplot=['plot_omni_characteristics_imb_comparison']

#plots_toplot=['electron flux no pad','just z','z and mu','zmu']
plots_toplot = ['zmu']
#plots_toplot=['electron flux no pad']


#raise a Warning if...
if mfp_toplot not in mfp0_vals:
    warnings.warn("Can cause issues for analytical builder if mfp_toplot not in current sim_zmu!")
    raise SystemError('Watch out!')


#choose between SDII SDCI FRAN 
analytical_toplot='FRAN'
plot_analytical=True


'''
event analysis parameters
'''

#in the future implement manual override for assumed injection


#event to analyse YYYY-mm-dd[a/b/c...]
#which event are we consdering

#event_considering = '1998-08-29'
event_considering='1998-08-29'
#manual_injection_override
inj_manual_override = False
#inj_time = datetime(year=2002,month=12,day=12,hour=0,minute=43)-timedelta(minutes=radio_travel_time_minutes)
#inj_time = datetime(year=2004,month=3,day=16,hour=8,minute=58,second=30)-timedelta(minutes=radio_travel_time_minutes)
#inj_time = datetime(year=2002,month=10,day=20,hour=11,minute=37)-timedelta(minutes=radio_travel_time_minutes)
#inj_manual_override_date = datetime(year=2000,month=3,day=7,hour=21,minute=57)
inj_manual_override_date=datetime(year=2004,month=3,day=16,hour=8,minute=58,second=30)
#plot validate the extracted omni_chars
validate_omni_chars = False
#which energy channel do we want to plot in the preliminary plots
#event_energy_toplot=180
event_energy_toplot=energy_toplot
#how long to evaluate the background? (minutes)
t_evaluate_background = 20
#what is the window size for the data smoothing?
#input in seconds, will convert to number of data points in the code.
window_size = 250
#find the injection time, just set as time of max in WAVES/RAD2 higher frequencies
#modify later to be max intensmity at a given frequency, be wary if there are multiple bursts! simplified treatment
#actually, no just do that. below is the max frequency (kHz)
inj_max_freq = 10000
#threshold flux to gauge the t_o and t_d ect. (fraction of normalise peak (1))
thresh = 1/2.71
#fit any (can be t_o,t_d,HWHM_premax,HWHM_postmax) or all
fit_chars = ['t_peak[s]']
#break energy [keV] used in plotting
break_energy = 45
#explicit energies to consider in the fitting
fit_energies=[67]
####

'''
cdf dictionary with the global days of cdf data
'''


### ONLY NEED A TAIL REDUCTION!

#note the cdfs contain the DAYS worth of data
events_cdf={}
#include the start and end time required in the dictionary in the format [datetime_start,datetime_end]
#events_cdf['']=[[datetime(year=,month=,day=),datetime(year=,month=,day=,hour=)],'','','']


#possible asymemmtric scattering events
#wang 2006
events_cdf['1998-08-29']=[[datetime(year=1998,month=8,day=29,hour=18,minute=15),datetime(year=1998,month=8,day=29,hour=19,minute=45)],'wi_sfsp_3dp_19980829_v01.cdf','wi_sfpd_3dp_19980829_v02.cdf','wi_h1_wav_19980829_v01.cdf',0]
#Tomin 2017
events_cdf['2004-03-16']=[[datetime(year=2004,month=3,day=16,hour=8),datetime(year=2004,month=3,day=16,hour=11)],'wi_sfsp_3dp_20040316_v01.cdf','wi_sfpd_3dp_20040316_v02.cdf','wi_h1_wav_20040316_v01.cdf',180]
#nitta 2006
events_cdf['2000-03-07']=[[datetime(year=2000,month=3,day=7,hour=21,minute=30),datetime(year=2000,month=3,day=7,hour=23,minute=25)],'wi_sfsp_3dp_20000307_v01.cdf','wi_sfpd_3dp_20000307_v02.cdf','wi_h1_wav_20000307_v01.cdf',0]


#Droge-like events (lower profile widths at higher energies difficult to concile with simulations)
events_cdf['1999-08-07']=[[datetime(year=1999,month=8,day=7,hour=16,minute=45),datetime(year=1999,month=8,day=7,hour=18,minute=30)],'wi_sfsp_3dp_19990807_v01(2).cdf','wi_sfpd_3dp_19990807_v02(2).cdf','wi_h1_wav_19990807_v01(2).cdf',0]
#events_cdf['1999-08-07']=[[datetime(year=1999,month=8,day=7,hour=16,minute=45),datetime(year=1999,month=8,day=7,hour=18,minute=30)],'wi_sfsp_3dp_19990807_v01(1).cdf','wi_sfpd_3dp_19990807_v02(1).cdf','wi_h1_wav_19990807_v01(1).cdf',0]
events_cdf['2002-10-19']=[[datetime(year=2002,month=10,day=19,hour=21,minute=00),datetime(year=2002,month=10,day=19,hour=22,minute=15)],'wi_sfsp_3dp_20021019_v01.cdf','wi_sfpd_3dp_20021019_v02.cdf','wi_h1_wav_20021019_v01.cdf',180]
events_cdf['2002-10-20']=[[datetime(year=2002,month=10,day=20,hour=14),datetime(year=2002,month=10,day=20,hour=15,minute=30)],'wi_sfsp_3dp_20021020_v01.cdf','wi_sfpd_3dp_20021020_v02.cdf','wi_h1_wav_20021020_v01.cdf',0]
events_cdf['2002-10-20_b']=[[datetime(year=2002,month=10,day=20,hour=11, minute=00),datetime(year=2002,month=10,day=20,hour=14)],'wi_sfsp_3dp_20021020_v01.cdf','wi_sfpd_3dp_20021020_v02.cdf','wi_h1_wav_20021020_v01.cdf',0]

#ballistic
events_cdf['2005-05-16']=[[datetime(year=2005,month=5,day=16,hour=2, minute=30),datetime(year=2005,month=5,day=16,hour=4)],'wi_sfsp_3dp_20050516_v01.cdf','wi_sfpd_3dp_20050516_v02.cdf','wi_h1_wav_20050516_v01.cdf',180]
#symmetric scattering
events_cdf['2003-09-30']=[[datetime(year=2003,month=9,day=30,hour=8),datetime(year=2003,month=9,day=30,hour=11,minute=30)],'wi_sfsp_3dp_20030930_v01.cdf','wi_sfpd_3dp_20030930_v02.cdf','wi_h1_wav_20030930_v01.cdf',0]

#bad data
events_cdf['1999-03-21']=[[datetime(year=1999,month=3,day=21,hour=16),datetime(year=1999,month=3,day=21,hour=18,minute=30)],'wi_sfsp_3dp_19990321_v01.cdf','wi_sfpd_3dp_19990321_v02.cdf','wi_h1_wav_19990321_v01.cdf',0]
events_cdf['2004-02-28']=[[datetime(year=2004,month=2,day=28,hour=3),datetime(year=2004,month=2,day=28,hour=5,minute=30)],'wi_sfsp_3dp_20040228_v01.cdf','wi_sfpd_3dp_20040316_v02.cdf','wi_h1_wav_20040228_v01.cdf',180]
events_cdf['2002-12-12_a']=[[datetime(year=2002,month=12,day=12,hour=12),datetime(year=2002,month=12,day=12,hour=15)],'wi_sfsp_3dp_20021212_v01.cdf','wi_sfpd_3dp_20021212_v02.cdf','wi_h1_wav_20021212_v01.cdf',0]
#---------------------------------------------------------------------------#


#############################################################################
#------------------------------------#--------------------------------------#
#############################################################################


#---------------------------------------------------------------------------#

'''
Packing the above data into dictionaries, and passed into user query to exectute the relevent scripts
'''


'''
building the dictionaries, i.e. packing away the above inputs to be ported into the corresponding scripts
'''

#sim_variables_text will hold the string title of the simulation variable
sim_variables_labels = ['num_particles','alpha_vals','kappa','mfp0_vals[AU]','energies[keV]','t_end[days]','injection_type','custom_end[days]','z_injection[AU]','mu_IC','D_mumu_type','mfp_constant','consider_focusing','h_val']
#sim_variables will hold the parameter used in the simulation
sim_variables = [Np,alpha_vals,kappa,mfp0_vals,ee,t_end,inj_set,custom_end,z_init,mu_IC_set,D_mumu_set,mfp_const,consider_focusing,h_val]
#building the appropriate dictuonary to hold all thple values
sim_dict=funcs.build_dict(sim_variables_labels,sim_variables)

#sim_variables_text will hold the string title of the simulation variable
sorting_variables_labels = ['num_samples','z_observation[AU]','z_tolerance[AU]','t_binwidth[s]','z_beds[AU]']
#sim_variables will hold the parameter used in the simulation
sorting_variables = [M,z_obs,z_tol,t_binwidth,z_beds]
#building the appropriate dictionary
sort_dict=funcs.build_dict(sorting_variables_labels,sorting_variables)


#ccreating the plotting dictionary
plotting_variables_labels=['mfp_toplot[AU]','alpha_toplot','energy_toplot[keV]','step_toplot','plots_toplot','analytical_toplot','plot_analytical','kappa_toplot','char_toplot','energy_char_toplot','mfp_comp_toplot[AU]']
plotting_variables=[mfp_toplot,alpha_toplot,energy_toplot,step_toplot,plots_toplot,analytical_toplot,plot_analytical,kappa_toplot,char_toplot,energy_char_toplot,mfp_comp_toplot]
plot_dict = funcs.build_dict(plotting_variables_labels,plotting_variables)


#creating the WIND information dictionary
WIND_variables_labels=['event_considering','event_energy_toplot[keV]','t_evaluate_background[mins]','inj_max_freq[kHz]','thresh','fit_chars','break_energy[keV]','fit_energies[keV]','inj_manual_override','inj_manual_override_date','validate_omni_chars','window_size[s]']
WIND_variables = [event_considering,event_energy_toplot,t_evaluate_background,inj_max_freq,thresh,fit_chars,break_energy,fit_energies,inj_manual_override,inj_manual_override_date,validate_omni_chars,window_size]
WIND_dict=funcs.build_dict(WIND_variables_labels,WIND_variables)


'''
create a dictionary, where each element is the corresponding dictionary for each script (the necessary input variables)
-note sort_dict implicately carries over some computed variables from sim_dict
'''

dicts ={}
dicts['sim_dict']=sim_dict
dicts['sort_dict']=sort_dict
dicts['plot_dict']=plot_dict
dicts['WIND_dict']=WIND_dict


'''
also create the dictionary holding the (primary) functions that user_request needs to run the scripts
this is redundannt and overcomplicated probably. Look to remove this in teh future for easier maintenance
'''
funcs_dict={}

funcs_dict['sim_func']=transport_functions(dicts).simulate_transport
funcs_dict['sort_func']=sorting_functions(dicts).sort_simulation
funcs_dict['plot_func']=plot_functions(dicts).plot_simulation
funcs_dict['solution_func']=solution_functions(dicts).build_solution
funcs_dict['WIND_fitting_func']=fit_WIND_event(dicts).fit_event


'''
running WIND loading and sorting scripts
'''
#building the WIND events class that holds the plottables and the extracted data. 
#Saves into data which is then retrievable through funcs.load_events_dict()
#we can perform these outside as these are invariant of the statistical comparisons! Which can be performed as a seperate function in user query
#work out statistical fitting after this.... first find the invariant quantities of each event!


#note, we perform the background correction while extracting! So pads and omnidirectional fluxes are background corrected.
event_data,simmed_energy_dict = WIND_functions(dicts,events_cdf,WIND_events).process_wind()
#sort the WIND data (i.e. extract the tprofile charecteristics)
sort_WIND_data(event_data,dicts,simmed_energy_dict).sort_WIND()


'''
running the simulation
'''

#creating the query class to ask the user what they wish to do given the input variables
query = user_query(dicts,funcs_dict)

#verify whether the simulation has been saved or not (necessary to check if data has been cleared)

if load_existing_sim_zmu==True:
    # data folder may have been cleared, check if sim_zmu can be loaded
    try:
        #return the core data path
        data_path=funcs.folder_functions().return_datapath()
        #load in the sim_zmu currently saved in memory (will raise an error if sim_zmu does notexist)
        sim_zmu=np.load(os.path.join(data_path,"sim_zmu.npy"))  
        #if it can be loaded, note True
        sim_zmu_loaded=True

    #if above raised an error, we could not load sim_zmu
    except:
        sim_zmu_loaded=False

else:
    sim_zmu_loaded=False

#if it does not yet exist, first prompt to simulate, then prompt further user requests
if  sim_zmu_loaded==False:
    print('Sim_zmu not yet saved.')
    #outputs the simulation array,carry over and whether they chose to pass or simulate
    sim_zmu = query.simulate_request()
    #builds the WIND event class and performs the sorting of the class
    #reupdate sim_zmu if necessary
    query.user_request(sim_zmu)
    #code finishes once user wants to exit, so terminate the script
    
#if sim_zmu already exists (implicately carry over will also be already defined)
if sim_zmu_loaded==True:
    print('Simulation array already defined.')
    #in this case automatic yes for having simulated
    sim_zmu=query.user_request(sim_zmu)

#plt.close('All')
print('Ending script.')