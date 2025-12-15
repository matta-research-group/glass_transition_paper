# Script to analyze density vs temperature data from NPT simulations to predict glass transition temperature (Tg)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
# For Piecewise Linear Fit
import pwlf

directory= os.getcwd()
results_dir = f'{directory}/{results}'
os.chdir(results_dir)

csv_files = glob.glob(os.path.join(os.getcwd(), '*NPT*.csv'))

# Concatenate all csv files into a single dataframe
df = pd.DataFrame()
for i in csv_files:
    df_tmp = pd.read_csv(i)
    df = pd.concat([df,df_tmp], axis = 0)
# Sort by time
df.sort_values('Time (ps)')

x = np.array(df['Temperature (K)'])
y = np.array(df['Density (g/mL)'])
# Piecewise Linear Fit on density vs temperature data
pwlf = pwlf.PiecewiseLinFit(x, y)
breaks = pwlf.fit(2)
# Determine Tg
tg = breaks[1]
# Predict values for piecewise linear fit
x_hat = np.linspace(x.min(), x.max(), 100)
y_hat = pwlf.predict(x_hat)
# Plot Tg and Piecewise Linear Fit
plt.figure()
plt.plot(x, y, 'o', markersize=1.5)
plt.plot(x_hat, y_hat, '-')
plt.xlabel('Temperature (K)')
plt.ylabel('Density (g/mL)')
plt.vlines(x = breaks[1], ymax=1.3, ymin=1.125, colors='g', linestyles='--')
# Save plot
plt.savefig('Tg_plot.png')
plt.show()

