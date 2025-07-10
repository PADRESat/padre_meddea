'''Tools to Analyze Data'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import astropy.units as u



def event_rate(event_list, plot_hist=False):
    """
    Take an event_list and find the average event rate and the minimum time interval between consecutive events. 

    Parameters
    ----------
    event_list: Timeseries object

    plot_hist: Boolean
        If True, plot the histogram of the unique times in the event_list.

    Returns
    -------
    delta_t: Float
        Gradient of the time array. 

    avg_event_rate: Float
        Average number of events per second. 

    min_delta_t: Float
        Minimum amount of time between two events. 
    
    """
    time=np.asarray([pd.to_datetime(l).timestamp() for l in np.unique(event_list['time'].value)]) # s
    delta_t=np.gradient(np.unique(time))
    #print(f'Time between events: {delta_t}')
    if plot_hist==True: 
        plt.hist(delta_t, bins=50)
        #plt.title(f'{Time Gradient}')
        plt.ylim([0,1])
        plt.xlabel('Time Between Events [s]')
        plt.ylabel('Number of Events')
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.show()
    avg_event_rate = np.around(1/np.average(delta_t.mean()),3) #round to 3 significant figures.
    print(f'Average Event Rate: {avg_event_rate} ct/s')   
    toto=np.unique(np.sort(delta_t))
    min_delta_t=np.min(toto[toto!=0])*1E6 #microseconds
    print(f'Minimum nonzero time interval between two triggers: %4.2f µs'%(min_delta_t))
    return delta_t, avg_event_rate, min_delta_t


