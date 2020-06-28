## ```cosinor``` module
The main functions of ```cosinor``` module are devoted to single- or multi-component cosinor analysis and visualization.


### ```cosinor.periodogram(X, Y, per_type='per', sampling_f = '', logscale = False, name = '', save_to = '', prominent = False, max_per = 240)```
Plots the periodogram with threshold of significance marked.
#### Parameters
* ```X```: iterable of timepoints 
* ```Y```: iterable of measurements

### ```cosinor.plot_data(df, names = [], folder = '', prefix = '')```
Plots the raw data.
#### Parameters
* ```df```: pandas dataframe of measurements
* ```names```: list of names to plot; if empty all names will be plotted
* ```folder```: folder to which the files are stored; if empty plots will be displayed instead of stored
* ```prefix```: prefix to the file names in which plots will be stored


### ```cosinor.plot_data_pairs(df, names, folder = '', prefix ='')```
Plots the raw data pairs (two groups of measurement on the same plot).
#### Parameters
* ```df```: pandas dataframe of measurements
* ```names```: pairs of names to plot together
* ```folder```: folder to which the files are stored; if empty plots will be displayed instead of stored
* ```prefix```: prefix to the file names in which plots will be stored


### ```cosinor.plot_phases(acrs, amps, tests, period=24, colors = ("black", "red", "green", "blue"), folder = "", prefix="", legend=True, CI_acrs = [], CI_amps = [], linestyles = [], title = "", labels = [])```
Plots the phases in a polar coordinate system.
#### Parameters
* ```acrs```: acrophase value or list of acrophases
* ```amps```: amplitude or list of amplitudes
* ```test```: list of names of tests
* ```period```: period value
* ```colors```: colors to use; if list number of phases to plot is larger than 4, black color is used in default setting
* ```folder```: folder to which the files are stored; if empty plots will be displayed instead of stored
* ```prefix```: prefix to the file names in which plots will be stored
* ```legend```: plot legend?
* ```CI_acrs```: list of acrophase confidence intervals to plot
* ```CI_amps```: list of amplitude confidence intervals to plot
* ```linestyles```: linestyles to use with each plot
* ```title```: if not empty used this label as a plot title
* ```labels```: labels assigned to each plot


### ```cosinor.fit_me(X, Y, n_components = 2, period = 24, model_type = 'lin', lin_comp = False, alpha = 0, name = '', save_to = '', plot=True, plot_residuals=False, plot_measurements=True, plot_margins=True, return_model = False, color = False, plot_phase = True)```
Perform the basic Cosinor regression
#### Parameters
* ```X```: iterable of timepoints 
* ```Y```: iterable of measurements
* ```n_components```: number of cosinor components to use
* ```period```: period to fit
* ```model_type```: by default linear model is used; if working with count data ```poisson```, ```gen_poisson``` (generalized Poisson) or ```nb```(negative binomial) can be specified.
* ```lin_comp```: if ```True``` linear componet (y ~ x) will also be included in the model
* ```alpha```: dispersion parameter - only with ```nb``` models
* ```name```: name to diplay
* ```folder```: folder to which the files are stored; if empty plots will be displayed instead of stored
* ```prefix```: prefix to the file names in which plots will be storedorms the fitting process on all measurement in the dataframe.
* ```plot```: if False plotting is not performed
* ```plot_residuals```: if True qq-plot will be made
* ```plot_measurements```: if True raw measurement are plotted
* ```plot_margins```: if True confidence intervals are plotted
* ```return_model```: if True model will be returned
* ```color```: the color to use when plotting; if False, black color will be used
* ```plot_phase```: if True phase plot will be made
#### Returns
* tuple which includes
  * ```model```: if ```return_model``` was set to True
  * ```statistics```
  * ```rhythm_params```: parameters describing the rhythmicity
  * ```X_test```: timepoints for additional plotting
  * ```Y_test```: measurement for additional plotting


### ```cosinor.population_fit(df, n_components = 2, period = 24, lin_comp = False, names = [], folder = '', prefix='', plot_measurements = True)```
Population fit (accordign to Cornelissen). For parameters and returns see ```cosinor_fit.fit_me```


### ```cosinor.fit_group(df, n_components = 2, period = 24, lin_comp = False, names = [], folder = '', prefix='', plot_measurements = True, plot = True)```
Perform the multi-component cosinor fit to the measurement in the dataframe. Calls ```cosinor.fit_me```.
#### Parameters
* ```df```: pandas dataframe of measurements
* ```n_components```: number of cosinor components or list with numbers of cosinor components; if list of components is given, cosinor fits will be performed on cosinor models with different numbers of cosinor components
* ```period```: period or list of periods to fit
* ```lin_comp```: if ```True``` linear componet (y ~ x) will also be included in the model
* ```names```: list of groups names to fit; if empty all groups will be used
* ```folder```: folder to which the files are stored; if empty plots will be displayed instead of stored
* ```prefix```: prefix to the file names in which plots will be stored
* ```plot_measurements```: if True raw measurement are plotted
* ```plot```: if False plotting is not performed
#### Returns
* dataframe with statistics and parameters of each fit

### ```cosinor.population_fit_group(df, n_components = 2, period = 24, lin_comp = False, names = [], folder = '', prefix='', plot_measurements = True)```
Perform the multi-component population-mean cosinor fit to the measurements in the dataframe. Parameters and returns are the same as in ```cosinor.fit_group```. Calls ```cosinor.population_fit```.

### ```cosinor.get_best_fits(df_results, criterium = 'R2_adj', reverse = False, n_components = [])```
Accepts fitting results and returns the best fits for each model complexity (number of components). Returns the best fit among all fits with a given number of cosinor components. Can be used to find the periods that best fit the data for a cosinor model.
#### Parameters
* ```df_results```: pandas dataframe returned by* ```df```: pandas dataframe of measurements
* ```criterium```: criterium for finding the best models
* ```reverse```: if false lower is better
* ```n_components```: list with numbers of cosinor components to analyse; if list is empty all components in the dataframe will be used
#### Returns
* dataframe with the best fitting model (the model with the best fitting period) for certain number of components

### ```cosinor.get_best_models(df, df_models, n_components = [1,2,3], lin_comp = False, criterium='p', reverse = True)```
Find the best models for each measurement. Get the best models for all fits (accepts the models obtained with ```fit_group``` and calls ```get_best_fits``` for each experiment).
#### Parameters
* ```df```: pandas dataframe of measurements
* ```df_models```: pandas dataframe of fitted models as returned by ```cosinor.fit_group```
* ```n_components```: number of cosinor components or list with numbers of cosinor components; if list of components is given, cosinor fits will be performed on cosinor models with different numbers of cosinor components
* ```lin_comp```: if ```True``` linear componet (y ~ x) is included in the models
* ```criterium```: criterium for finding the best models
* ```reverse```: if false lower is better
#### Returns
* dataframe with the best fitting models

### ```cosinor.get_best_models_population(df, df_models, n_components = [1,2,3], lin_comp = False, criterium = 'RSS', reverse = True)```
Find the best population-mean models for each measurement. See description of ```cosinor.get_best_models```.

### ```cosinor.plot_df_models(df, df_models, plot_residuals=True, folder ="")```
Plots the given models with the given data.

### ```cosinor.plot_df_models_population(df, df_models, plot_residuals=True, folder ="")```
Plots the given population-mean models with the given data.

### ```cosinor.compare_pairs(df, pairs, n_components = 3, period = 24, lin_comp = False, folder = '', prefix = '', plot_measurements=True)```
Perform the LimoRhyde analysis of differential expression between the given pairs of measurements.
#### Parameters
* ```df```: pandas dataframe of measurements
* ```pairs```: list of pairs to compare
* ```n_components```: number of cosinor components to be used in the model for comparison
* ```lin_comp```: if ```True``` linear componet (y ~ x) is included in the models
* ```folder```: folder to which the plots are stored; if empty plots will be displayed instead of stored
* ```prefix```: prefix to the file names in which plots will be stored 
* ```plot_measurements```: if True raw measurement are plotted
#### Returns
* dataframe with the results of comparison

### ```cosinor.compare_nonlinear(X1, Y1, X2, Y2, test1 = '', test2 = '', min_per = 18, max_per=36, compare_phase = False, compare_period = False, compare_amplitude = False, save_to = '', plot_residuals=False)```
Perform the analysis of differential expression on the basis of non-linear cosinor model.