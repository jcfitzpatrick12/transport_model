# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:45:40 2022

@author: jcfit

-these are functions that are used globallys

change this into a class too!
"""


import numpy as np
import os
from datetime import datetime,timedelta
import glob


'''
function that defines the global path for the data
'''

#class for building folders to save the figures into
class folder_functions:
    def __init__(self):
        self.path_util = os.path.dirname(os.path.realpath(__file__))

    
    #returns the path where we will store the data
    def return_datapath(self):
        path_util=self.path_util
        datapath= self.create_simfolder(path_util,'data')
        return datapath
    
    
    #creates a folder at a given location
    def create_simfolder(self,root_path,folder_tomake):
        # Parent Directory path
        path_tobuild = os.path.join(root_path,folder_tomake)
        # Path
        #path = os.path.join(parent_dir, sim_folder_name)
        #try to create the directory named as such and such
        if not os.path.exists(path_tobuild):
                os.makedirs(path_tobuild)
                print('Created:', path_tobuild)
        return path_tobuild

        
    #write text into the file text_file located in path
    #if it does not exist, the string text_file will be created
    def append_text(self,text,text_file,path):
        txt_path = os.path.join(path,text_file+'.txt')
        with open(txt_path, "a") as text_file:
            text_file.write(text)
            
    def clear_text(self,text_file,path):
        txt_path = os.path.join(path,text_file+'.txt')
        with open(txt_path, "w") as text_file:
            text_file.write('')


def return_datapath():
    return r'\Users\jcfit\Desktop\Summer Research\Transport Modelling\Electron Transport Modelling\Numerical Modelling Scripts\data'

'''
class to organise the simulation input paramaters
'''

#build_dict takes in two arrays, numerical values and corresponding labels and builds a dictionary
def build_dict(labels,elements):
    dicts={}
    if len(labels)!=len(elements):
        raise SystemError('Error labelling defining simulation parameters')
    N=len(labels)
    for n in range(N):
        dicts[labels[n]]=elements[n]
    return dicts

#function that builds a string from elements of a dictionary
def build_string_fromdicto(dicto):
    long_str=[]
    for n,elem in enumerate(dicto):
        small_str=elem+'='+str(dicto[elem])
        long_str.append(small_str)
    return str(long_str)


'''
file saving functions
'''
def file_name_asym(alpha,l_n,l_p,kappa):
    return str(alpha)+'_'+str(l_n)+'-'+str(l_p)+'_'+str(kappa)+'_'


'''
WIND data extraction functions
'''

#function that converts UT time format into datetime object
def windstr_to_datetime(date_str):
    return datetime.strptime(date_str,'%Y-%m-%dT%H:%M:%S.%fZ')

#function that converts UT time format into datetime object
def datetime_tostring(dt):
    return datetime.strftime(dt,'%Y-%m-%d %H:%M:%S.%f')

#class to manage the files of different events (rather than copy pasting each time)
#event name is just a string, omni_files is a 2 string array [t_name,omni_name]
#pitch angle names is also a 2 string array [pitch_t,name,pitch_name]
class event_files:
    def __init__(self,omni_file,pitch_angle_file):
        self.omni_file = omni_file
        self.pitch_angle_file = pitch_angle_file
    


#function that converts UT time format into datetime object
def windstr_to_datetime(date_str):
    return datetime.strptime(date_str,'%Y-%m-%dT%H:%M:%S.%fZ')

#function that converts UT time format into datetime object
def datetime_tostring(dt):
    return datetime.strftime(dt,'%Y-%m-%d %H:%M:%S.%f')

'''
##we can now load in the events dictionary that contains all the data with the following command:
def load_events_dict():
    data_path=folder_functions().return_datapath()
    events_dict=np.load(os.path.join(data_path,"events_dict.npy"),allow_pickle=True).flat[0]
    return events_dict
'''
##we can now load in the events dictionary that contains all the data with the following command:
## saved in events_data.npy is the (unsorted) data from the selected event
def load_events_data():
    data_path=folder_functions().return_datapath()
    events_dict=np.load(os.path.join(data_path,"event_data.npy"),allow_pickle=True).flat[0]
    return events_dict

#outputs the dictionary and numpy array for convenience
def load_event_chars():
    data_path=folder_functions().return_datapath()
    event_chars_dict=np.load(os.path.join(data_path,"event_chars_dict.npy"),allow_pickle=True).flat[0]
    event_chars=np.load(os.path.join(data_path,"event_chars.npy"))
    return event_chars,event_chars_dict

    
def load_injection_time():
    data_path=folder_functions().return_datapath()
    inj_time=np.load(os.path.join(data_path,"injection_time.npy"),allow_pickle=True).item()
    return inj_time
    
#function to load the most recent carry_over
def load_carry_over():
    data_path=folder_functions().return_datapath()
    carry_over=np.load(os.path.join(data_path,"carry_over.npy"),allow_pickle=True).flat[0]
    return carry_over

#given a value and an array, find the index of the bin to which the value is closest
#ar must be 1D!
def find_index_1D(ar,val):
    #if not already, convert the array to a numpy array
    ar=np.array(ar)
    #take val away from ar
    ar=ar-val
    #take the absolute magnitude of ar
    ar=np.abs(ar)
    try:
        return np.where(ar==np.nanmin(ar))[0][0]
    except:
        return (np.where(ar==np.nanmin(ar))[0])
    

#returns the time in minutes for light to travel 1AU
def light_travel_time_1AU():
    radio_travel_time_days = 1/(299792458*5.77548e-7)
    radio_travel_time_minutes=radio_travel_time_days*1440
    return radio_travel_time_minutes
    
    
    
#function that updates all t_bins_inj to the currently loaded event
def update_t_bins_inj():
    #load the path to the data folder
    data_path=folder_functions().return_datapath()
    #create an array that contains all file names in a folder
    data_names=os.listdir(data_path)
    #loop through each file name
    #picks out the sim_omni CHARECTERISTICS for each simulation (t_o,t_d,HWHM_o,HWHM_d)
    keyword='t_bins_minutes_'
    #create a big array that will hold the 
    sim_chars_data_overkappa = []
    for file in data_names:
        #seperating out into chars files and var files
        if keyword in file:
            #now load this t_bins_minutes and rebuild t_bins_inj with the currently saved injection time
            t_bins_minutes = np.load(os.path.join(data_path,file))
            inj_time=np.load(os.path.join(data_path,"injection_time.npy"),allow_pickle=True).item()
            n_times = len(t_bins_minutes)
            #converting t_bins_minutes to datetimes (recall, t_bins minutes are the bin edges (beds))
            t_bins_inj = []
            for n in range(n_times):
                bed_datetime = inj_time+timedelta(minutes=t_bins_minutes[n])
                t_bins_inj.append(bed_datetime)
            #infer fname from the time files own name
            fname=file[15:-4]
            #and save t_bins_inj using the inferred simulation name
            np.save(os.path.join(data_path,"t_bins_inj_"+fname),t_bins_inj)
    #loads in all the t_bins_minutes, and saves a new file t_bins_inj with the same file name, but updated to the new derived injection time
    return None

#function that deletes all items in data
def delete_data():
    data_path=folder_functions().return_datapath()
    files = glob.glob(os.path.join(data_path,'*'))
    for f in files:
        os.remove(f)       
    return None

    
#function to convert datetime to float
def datetime_to_float(date_ar):
    float_dates = []
    for d in date_ar:
        float_dates.append(d.timestamp())
    return float_dates

#function to take a float and convert it to a datetime object
def float_to_datetime(fl_array):
    date_dates = []
    for fl in fl_array:
        date_dates.append(datetime.fromtimestamp(fl))
    return date_dates
    