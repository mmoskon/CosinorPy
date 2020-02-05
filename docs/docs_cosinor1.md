## ```cosinor1``` module
The main functions of ```cosinor1``` module are devoted to single-component cosinor analysis and visualization.

### ```cosinor1.fit_cosinor(X, Y, period, test='', save_to = '', plot_on = True)```
Fits the single-component cosinor to given data with specified period.
#### Parameters
* ```X,Y```: iteratives containing timepoints and measurements
* ```period```: fitting period
* ```test```: if specified this name will be used when saving the files and plotting the graphs
* ```save_to```: if specified plots will be saved to a file with given name; if empty, plots will be displayed
* ```plot_on```: plot the results?
#### Returns
* ```tuple``` that contains fitting results, estimated amplitude, estimated acrophase and statistics



### ```cosinor1.population_fit_cosinor(df_pop, period, save_to='', alpha = 0.05, plot_on = True, plot_individuals = True, plot_measurements=True, plot_margins=True)```
Fits the single-component population-mean cosinor to given data.
#### Parameters
* ```df_pop```: dataframe containing the population data; each individual (replication) should be named with ```_rep#``` postfix, where ```#``` is the number of the replicaiton
* ```period```: fitting period
* ```save_to```: if specified plots will be saved to a file with given name; if empty, plots will be displayed
* ```alpha```: significance level used for the estimation of confidence intervals
* ```plot_on```: plot the results?
* ```plot_individuals```: plot each individual or only population mean
* ```plot_measurements```: plot raw measurements?
* ```plot_margins```: plot confidence intervals?
#### Returns
* ```dictionary``` that contains the following keys:
 * ```test```: population name
 * ```names```: names of assessed parameters
 * ```values```: values of assessed parameters
 * ```means```: population-mean values of assessed parameters
 * ```confint```: confidence intervals
 * ```p-value```: p-value of the null hypothesis (population mean amplitude equals zero)

### ```cosinor1.fit_group(df, period = 24, save_folder='', plot_on=True)```
Perfroms the fitting of all measurements in the dataframe.
#### Parameters
* ```df```: dataframe containing the measurements
* ```period```: fitting period
* ```save_folder```: if specified plots will be saved to a given folder; if empty, plots will be displayed
* ```plot_on```: plot the results?
#### Returns
* dataframe of the fitting results with statistics and assessed parameters

### ```cosinor1.population_fit_group(df, period = 24, save_folder='', plot_on=True)```
Performs the population-mean fitting of all measurements in the dataframe. Parameters and returns are the same as with the  ```cosinor1.fit_group```.

### ```cosinor1.test_cosinor_pairs(df, pairs, period = 24, folder = '', prefix='', plot_measurements=True, legend=True, df_best_models = -1)```
Perform the single-component cosinor analysis of differential expression between the given pairs of measurements.
#### Parameters
* ```df```: dataframe containing the measurements
* ```pairs```: list of tuples containing the pairs to compare
* ```period```: fitting period
* ```folder```: folder to which the plots are stored; if empty plots will be displayed instead of stored
* ```prefix```: prefix to the file names in which plots will be stored 
* ```plot_measurements```: if True raw measurement are plotted
* ```legend```: display the legend on the plots?
* ```df_best_models```: if dataframe is given as df_best_models periods of these models will be used for fitting (instead of the given period).
#### Returns
Dataframe containing the results.

### ```cosinor1.population_test_cosinor_pairs(df, pairs, period=24)```
Perform the single-component population-mean cosinor analysis of differential expression between the given pairs of populations.
#### Parameters
* ```df```: dataframe containing the measurements
* ```pairs```: list of tuples containing the pairs to compare
* ```period```: fitting period
#### Returns
Dataframe containing the results.




