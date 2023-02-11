# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:19:08 2022

@author: jcfit
"""

'''

LOOK FOR STATISTICAL METHOD THAT WILL ALLOW FOR EFFECTIVE FITTING
NOTE, 
'''

'''
TO DO, VERY IMPORTANT

WORK OUT WHETHER WIND BINS ARE MIDDLE, POST, PRE AND TRY AND CONVERT TO STAIRS!
-also need to work on a seperate wind_plot class **
-and create a save event folder! for each event **


- plotter that looks at all the energies  
-incorporate (event_fitting_class) as a user interface function, or auto runs?
-let's just auto run
- how to handle different kappa values?


'''

'''
simulation time profile sare saved in main_sorter
wind time profiles are saved in sort_WIND
-normalise the wind pads to peak flux
-convert the pitch angle bins to pitch angle cosine bins

DONE

Now workout the kappa problem, and look into statistical comparison methods
-place a fitting class into user interface
-find a way to load in all kappa event charecteristics separetly then combine into a vector array
-perform a fitting on either t_o or t_d  or HWHM will be the same
-once this is done look for a statistical method

'''


'''
NOTE REVISIT SAVE OF SIM_OMNI_CHARS AT THE BOTTOM OF SORTING. NOT CLEAN CODING BUT BENCH FOR NOW
MAY BUG OUT PLOTTING
'''

'''
To do after dinner....
-check the multidim array is working
-verify t_o and t_d are correct in plotting with the new multidim array
-and correspond to the right simulation parameters

LAST THING
-plot [t_o,t_e,HWHM+,HWHM-] for sim vs event

DONE
'''


'''
CHECKED THE T_0 ects analaysis!


TO DO

Check the pitch angle -> mu bins for the pads
Find a statistical method to compare either the t_o and t_e tuple or all simultaneously



'''


'''
selectively fit either t_o,t_d, ects seperately or all as a whole
first use minimisation of mean square error just to see if it fits anything reasonable.
plot all of the event_chars foe every sim on top of the one selected.
make sure I didnt break anything in mucors
'''

























