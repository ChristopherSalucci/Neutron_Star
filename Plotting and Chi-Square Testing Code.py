import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols


#The following is a plotter function that turns the ASCII file into a data frame.


def load_ascii_data(file_path: str):
    data = pd.read_csv(file_path, header=None, delim_whitespace=True, skipinitialspace=True)
    return np.array(data)

ascii_data_original = load_ascii_data("Slow Cooling, No MXB, 250 Data Points.txt") #Stores the original unedited NSCool data to an array



#The first six parameters of the simulation (the ones that were the most important for my project)
v = ascii_data[:,0] #The data point number from 1 to 1500
w = np.log(ascii_data[:,1]) #Time elapsed in years (logscale)
x = np.log(ascii_data[:,2]) #Effective temp. in Kelvin (logscale)
y = np.log(ascii_data[:,3]) #Photon luminosity in ergs/s (logscale)
z = np.log(ascii_data[:,4]) #Neutrino luminosity in ergs/s (logscale)
h = np.log(ascii_data[:,5]) #Radiative transport (logscale)


#The following is an example of a plot. Change the input slices (and corresponding labels too), to plot the different parameters.


#plt.plot(x, y, color="blue", label='Photon Luminosity')
#plt.plot(x, z, color="green", label='Neutrino Luminosity')

#a, b = np.polyfit(x,y,1) #Photon, log-scale regression
#c, d = np.polyfit(x,z,1) #Neutrino, log-scale regression
#coef_phot = np.polyfit(x,y,1)
#poly1d_fn_phot = np.poly1d(coef_phot)
#coef_neut = np.polyfit(x,z,1)
#poly1d_fn_neut = np.poly1d(coef_neut)

#plt.plot(x, poly1d_fn_phot(x), color="blue", linestyle = 'dashed', label='Photon Lum. Regression')
#plt.plot(x, poly1d_fn_neut(x), color="green", linestyle = 'dashed', label='Neutrino Lum. Regression')

#plt.title('Luminosity vs Effective Temperature (with outbursts)')
#plt.xlabel('Log-scale of Effective Temperature (in Kelvin)')
#plt.ylabel('Log-scale of Luminosity (in erg / s)')
#plt.legend()
#plt.show()


#These values are for the goodness of fit testing, and come from Table 2.2.
obs_times_in_MJD = [52197, 52563, 52712.2, 52768, 53560, 53576]
obs_time_in_days = [0, 366, 515.2, 571, 1363, 1379]
obs_time_in_years = [0, 1.00274, 1.41151, 1.56438, 3.73425, 3.77808]
obs_temp = [121, 85, 77, 73, 58, 54]
error_bars = [2, 2, 1, 1.5, 3, 4.5]



#The following plots the real observed 2001 outburst data, without my simulation curve.



#x = obs_time_in_years
#y = obs_temp

#coef = np.polyfit(x,y,1)
#poly1d_fn = np.poly1d(coef)
#plt.plot(x, poly1d_fn(x), color='blue', linestyle='--', label='Linear Fit')
#plt.scatter(x, y, color='blue', label='Observed Data')
#plt.errorbar(x, y, yerr=error_bars, fmt='none')

#plt.xlabel('Time elapsed (in years)')
#plt.ylabel('Effective Temperature ${kT_{eff}}$ (in eV)')
#plt.title('Observations of MXB 1659-29 (after the 2001 outburst)')
#plt.legend()
#plt.show()



#The following is an example of the residuals guide, used to guess at improvements to the NSCool control files.



#import statsmodels.api as sm
#from statsmodels.formula.api import ols

#d = {"Time":x,"Temp":y} #First create a dictionary
#data = pd.DataFrame(d) #Then a dataframe
#linear_model = ols('Temp~Time',data=data).fit()
#print(linear_model.summary())
#fig = plt.figure(figsize=(14,8))
#fig = sm.graphics.plot_regress_exog(linear_model,'Time',fig=fig)



#The following is the final curve fitting figure required to pass. Figure 3.10 comes from this.



df = np.array(pd.read_csv(r"C:\Users\chris\OneDrive\Desktop\PHYS 449\MXB Final Data.txt", header=None, delim_whitespace=True, skipinitialspace=True))

w = (df[1293:1368,1])-(df[1293,1]) #Time elapsed from peak event, in years
x = (8.62*10**-6)*(df[1293:1368,2]) #Eff. temp. in eV using Boltzmann const.
plt.plot(w, x, color="red", label='Simulated Model of Eff. Temp.')

p = obs_time_in_years #Overlay the same scatter plot from before
q = obs_temp
coef = np.polyfit(p,q,1)
poly1d_fn = np.poly1d(coef)
plt.plot(p, poly1d_fn(p), color='blue', linestyle='--', label='Linear Fit')

plt.scatter(p, q, color='blue', label='Observed Data')
plt.errorbar(p, q, yerr=error_bars, fmt='none')
plt.xlabel('Time elapsed (in years since outburst ended)')
plt.ylabel('Effective Temperature ${kT_{eff}}$ (in eV)')
plt.title('Observations of MXB 1659-29 (after the 2001 outburst)')
plt.legend()
plt.show()



'''


#Everything after here is preliminary work on the MCMC algorithm.


#What should be noted is that Q_imp is going to vary according to a uniform distribution over 0 to 1,
#as a fraction of its current value. These are the priors, and they are uniformly distributed.

#The posteriors are going to be normally distributed, symmetrically around the log_10 (L_v / T^6) value.

#Every update via the likelihood function (also Gaussian) is going to result in a narrower normal distribution.

#The correct answer that the Brown et al. paper determined was 38.2.

#The following are the basic renormalization functions that will be used.



'''