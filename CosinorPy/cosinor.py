import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.multitest as multi
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as multi
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import t
from scipy.optimize import curve_fit
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def plot_components(X, Y, n_components = 3, period = 24, name = '', save_to = ''):
    
    A = np.sin((X/period)*np.pi*2)
    B = np.cos((X/period)*np.pi*2)
    C = np.sin((X/(period/2))*np.pi*2)
    D = np.cos((X/(period/2))*np.pi*2)
    E = np.sin((X/(period/3))*np.pi*2)
    F = np.cos((X/(period/3))*np.pi*2)
    #G = np.sin((X/(period/4))*np.pi*2)
    #H = np.cos((X/(period/4))*np.pi*2) 
    
    
    fig, axs = plt.subplots(n_components, 2, constrained_layout=True)
    fig.suptitle(name, fontsize=16)    
    
    
    
    axs[0,0].plot(A, Y,'.')
    axs[0,0].set(xlabel = 'sin((x/'+str(period)+') * 2$\pi$)')
       
    axs[0,1].plot(B, Y,'.')
    axs[0,1].set(xlabel = 'cos((x/'+str(period)+') * 2$\pi$)')
    
    
    if n_components >= 2:      
        axs[1,0].plot(C, Y,'.')
        axs[1,0].set(xlabel = 'sin((x/'+str(period/2)+') * 2$\pi$)')
        axs[1,1].plot(D, Y,'.')
        axs[1,1].set(xlabel = 'cos((x/'+str(period/2)+') * 2$\pi$)')
        
    if n_components == 3:        
        axs[2,0].plot(E, Y,'.')
        axs[2,0].set(xlabel = 'sin((x/'+str(period/3)+') * 2$\pi$)')
        axs[2,1].plot(F, Y,'.')
        axs[2,1].set(xlabel = 'cos((x/'+str(period/3)+') * 2$\pi$)')
  
    if n_components == 4:        
        axs[3,0].plot(E, Y,'.')
        axs[3,0].set(xlabel = 'sin((x/'+str(period/4)+') * 2$\pi$)')
        axs[3,1].plot(F, Y,'.')
        axs[3,1].set(xlabel = 'cos((x/'+str(period/4)+') * 2$\pi$)')
  
   
    for ax in axs.flat:
        ax.set(ylabel = 'y')

    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()
    
def periodogram_df(df, per_type='per', sampling_f = '', logscale = False, name = '', folder = '', prominent = False, max_per = 240):
    names = list(df.test.unique())
    names.sort()

    for name in names:
        x, y = np.array(df[df.test == name].x), np.array(df[df.test == name].y)
        if folder:
            save_to = folder + "\\per_" + name
        else:
            save_to = ""

        periodogram(x,y, per_type = per_type, sampling_f = sampling_f, logscale = logscale, name=name, save_to = save_to, prominent = prominent, max_per=max_per)


def periodogram(X, Y, per_type='per', sampling_f = '', logscale = False, name = '', save_to = '', prominent = False, max_per = 240):
    
    
    if per_type == 'per' or per_type == 'welch':
    
        X_u = np.unique(X)
        Y_u = []
        for x_u in X_u:
            #y_u.append(np.mean(y[t == x]))
            Y_u.append(np.median(Y[x_u == X]))
        
       
        
        if not sampling_f:
            sampling_f = 1/(X[1]-X[0])
        
        Y = Y_u
        
  
    
    if per_type == 'per':
        # Fourier
        f, Pxx_den = signal.periodogram(Y,sampling_f)
    elif per_type =='welch':
        # Welch
        f, Pxx_den = signal.welch(Y,sampling_f)
    else:
        # Lomb-Scargle
        min_per = 2
        #max_per = 50
        
        f = np.linspace(1/max_per, 1/min_per, 10)
        Pxx_den = signal.lombscargle(X, Y, f)
        
        

    # significance
    # Refinetti et al. 2007
    p_t = 0.05
    
    N = len(Y)
    T = (1 - (p_t/N)**(1/(N-1))) * sum(Pxx_den) # threshold for significance
    
    if f[0] == 0:
        per = 1/f[1:]
        Pxx = Pxx_den[1:]
    else:
        per = 1/f
        Pxx = Pxx_den
    
    
    Pxx = Pxx[per <= max_per]
    per = per[per <= max_per]
    
    
    
    if logscale:
        plt.semilogx(per, Pxx, 'ko')
        plt.semilogx(per, Pxx, 'k--', linewidth=0.5)
        plt.semilogx([min(per), max(per)], [T, T], 'k--', linewidth=1)
    else:
        plt.plot(per, Pxx, 'ko')
        plt.plot(per, Pxx, 'k--', linewidth=0.5)
        plt.plot([min(per), max(per)], [T, T], 'k--', linewidth=1)


    peak_label = ''

    if prominent:    
        locs, heights = signal.find_peaks(Pxx, height = T)
        
        if locs:        
            heights = heights['peak_heights']
            s = list(zip(heights, locs))
            s.sort(reverse=True)
            heights, locs = zip(*s)
            
            heights = np.array(heights)
            locs = np.array(locs)
            
            """
            if logscale:
                plt.semilogx(per[locs[:10]], heights[:10], 'x')
            else:
                plt.plot(per[locs[:10]], heights[:10], 'x')
            """
                
            peak_label = ', max peak=' + str(per[locs[0]])            
    
    else:
        locs = Pxx >= T
        if any(locs):          
            heights, locs = Pxx[locs], per[locs]                   
            HL = list(zip(heights, locs))
            HL.sort(reverse = True)
            heights, locs = zip(*HL)
            
            peak_label = ', peaks=\n'
            
            locs = locs[:11]
            for loc in locs[:-1]:
                peak_label += "{:.2f}".format(loc) + ','
            peak_label += "{:.2f}".format(locs[-1])
                

    plt.xlabel('period [hours]')
    plt.ylabel('PSD')
    plt.title(name + peak_label)
    
    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()


def get_best_fits(df_results, criterium = 'R2_adj', reverse = False, n_components = []):
    df_best = pd.DataFrame(columns = df_results.columns)
    names = np.unique(df_results.test)
    
    for name in names:
        if n_components:
            for n_comp in n_components:
                if reverse:
                    M = df_results[(df_results.test == name) & (df_results.n_components == n_comp)][criterium].min()
                else:
                    M = df_results[(df_results.test == name) & (df_results.n_components == n_comp)][criterium].max()
                df_best = df_best.append(df_results[(df_results.test == name) & (df_results.n_components == n_comp) & (df_results[criterium] == M)], ignore_index = True)
        
        else:
            M = df_results[df_results.test == name][criterium].max()
            df_best = df_best.append(df_results[(df_results.test == name) & (df_results[criterium] == M)], ignore_index = True)
    
    return df_best

def plot_data(df, names = [], folder = '', prefix = ''):
    if not names:
        names = np.unique(df.test) 
        
    for test in names:
        X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)  
        
        plt.plot(X,Y,'.')
        plt.title(test)
       
        #test = test.replace("$","")
       
        #fig = plt.gcf()
        #fig.set_size_inches(11,8)               
        
       
        if folder:
            plt.savefig(folder+'\\'+prefix+test+'.png')
            plt.savefig(folder+'\\'+prefix+test+'.pdf')
            
            plt.close()
        else:
            plt.show()
    
    
def plot_data_pairs(df, names, folder = '', prefix =''):
        
    for test1, test2 in names:
        X1, Y1 = np.array(df[df.test == test1].x), np.array(df[df.test == test1].y)  
        X2, Y2 = np.array(df[df.test == test2].x), np.array(df[df.test == test2].y)  
        
        plt.plot(X1,Y1,'ko', markersize=1, label=test1)
        plt.plot(X2,Y2,'ro', markersize=1, label=test2)
        plt.legend()
        plt.title(test1 + ' vs. ' + test2)
       
        if folder:
            plt.savefig(folder+'\\'+prefix+test1+'_'+test2+'.png')
            plt.savefig(folder+'\\'+prefix+test1+'_'+test2+'.pdf')
            plt.close()
        else:
            plt.show()
def fit_group(df, n_components = 2, period = 24, lin_comp = False, names = [], folder = '', prefix='', plot_measurements = True, plot = True):
    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'RSS', 'R2', 'R2_adj', 'log-likelihood', 'period(est)', 'amplitude', 'acrophase', 'mesor'])

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    if not names:
        names = np.unique(df.test) 

    for test in names:
        for n_comps in n_components:
            for per in period:            
                if n_comps == 0:
                    per = 100000
                X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)        
                if folder:
                    save_to = folder+'\\'+prefix+test+'_compnts='+str(n_comps) +'_per=' + str(per)
                else:
                    save_to = ''
                
                results, statistics, rhythm_param, X_full, Y_full = fit_me(X, Y, n_components = n_comps, period = per, model_type = 'lin', lin_comp = lin_comp, name = test, save_to = save_to, plot_measurements = plot_measurements, plot=plot)
            
                df_results = df_results.append({'test': test, 
                                            'period': per,
                                            'n_components': n_comps,
                                            'p': statistics['p'], 
                                            'p_reject': statistics['p_reject'],
                                            'RSS': statistics['RSS'],
                                            'R2': results.rsquared,
                                            'R2_adj': results.rsquared_adj,
                                            'ME': statistics['ME'],
                                            'resid_SE': statistics['resid_SE'],
                                            'log-likelihood': results.llf,        
                                            'period(est)': rhythm_param['period'],
                                            'amplitude': rhythm_param['amplitude'],
                                            'acrophase': rhythm_param['acrophase'],
                                            'mesor': rhythm_param['mesor']}, ignore_index=True)
                if n_comps == 0:
                    break
    
    df_results.q = multi.multipletests(df_results.p, method = 'fdr_bh')[1]
    df_results.q_reject = multi.multipletests(df_results.p_reject, method = 'fdr_bh')[1]
    
    
    return df_results

def population_fit_group(df, n_components = 2, period = 24, lin_comp = False, names = [], folder = '', prefix='', plot_measurements = True):
    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'RSS', 'period(est)', 'amplitude', 'acrophase', 'mesor'])

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    if not names:
        names = np.unique(df.test) 

    names = list(map(lambda x:x.split('_rep')[0], names))
    
    for name in set(names):
        for n_comps in n_components:
            for per in period:            
                if n_comps == 0:
                    per = 100000
                    
                    
                df_pop = df[df.test.str.startswith(name)]   

                if folder:
                    params, statistics, statistics_params, rhythm_params, results = population_fit(df_pop, n_components = n_comps, period = per, lin_comp= lin_comp, save_to=folder+'\\'+prefix+name+'_compnts='+str(n_comps) +'_per=' + str(per))
                else:
                    params, statistics, statistics_params, rhythm_params, results = population_fit(df_pop, n_components = n_comps, period = per, lin_comp= lin_comp, plot_on = True)
                    
                            
                df_results = df_results.append({'test': name, 
                                            'period': per,
                                            'n_components': n_comps,
                                            'p': statistics['p'], 
                                            'p_reject': statistics['p_reject'],
                                            'RSS': statistics['RSS'],
                                            'ME': statistics['ME'],
                                            'resid_SE': statistics['resid_SE'],
                                            'period(est)': rhythm_params['period'],
                                            'amplitude': rhythm_params['amplitude'],
                                            'acrophase': rhythm_params['acrophase'],
                                            'mesor': rhythm_params['mesor']}, ignore_index=True)
                if n_comps == 0:
                    break
    
    df_results.q = multi.multipletests(df_results.p, method = 'fdr_bh')[1]
    df_results.q_reject = multi.multipletests(df_results.p_reject, method = 'fdr_bh')[1]
        
    return df_results

def get_best_models_population(df, df_models, n_components = [1,2,3], lin_comp = False, criterium = 'RSS', reverse = True):    
    names = np.unique(df_models.test)   
    df_best = pd.DataFrame(columns = df_models.columns)
    df_models = get_best_fits(df_models, criterium = criterium, reverse = reverse, n_components=n_components)
    for test in names:
        n_points = df[df.test.str.startswith(test)].x.shape[0] # razlika med get_best_models in get_best_models_population
        df_test_models = df_models[df_models.test == test]
        df_test_models = df_test_models.sort_values(by=['n_components'])
    
        i = 0
        for new_row in df_test_models.iterrows():            
            if i == 0:
                best_row = new_row
                i = 1
            else:
                RSS_reduced = best_row[1].RSS
                RSS_full = new_row[1].RSS

                DF_reduced = n_points - (best_row[1].n_components * 2 + 1)
                DF_full = n_points - (new_row[1].n_components * 2 + 1)

                if lin_comp:
                    DF_reduced -= 1
                    DF_full -= 1                
                #print (test, old_row[1].n_components, new_row[1].n_components)
                if compare_models(RSS_reduced, RSS_full, DF_reduced, DF_full) < 0.05:
                    best_row = new_row
        df_best = df_best.append(best_row[1], ignore_index=True)
    return df_best


def remove_lin_comp(X, Y, n_components = 1, period = 24):
    
    X_fit = generate_independents(X, n_components = n_components, period = period, lin_comp = True)
    model = sm.OLS(Y, X_fit)
    results = model.fit()
    
    #plt.plot(X,Y, '.')
    
    X_lin = np.zeros(X_fit.shape)
    X_lin[:,1] = X_fit[:,1]
    Y_lin = results.predict(X_lin)
    Y = Y-Y_lin
    
    #plt.plot(X, results.fittedvalues, X, Y_lin)
    #plt.plot(X,Y, 'x')
    
    """
    X_fit = generate_independents(X, n_components = n_components, period = period, lin_comp = False)
    model = sm.OLS(Y, X_fit)
    results = model.fit()
    plt.plot(X, results.fittedvalues, color="black")
    """
    
    #plt.show()
    
    return X, Y
    


def generate_independents(X, n_components = 3, period = 24, lin_comp = False):
    """
    ###
    # prepare the independent variables
    ###
    """
    A1 = np.sin((X/period)*np.pi*2)
    B1 = np.cos((X/period)*np.pi*2)
    A2 = np.sin((X/(period/2))*np.pi*2)
    B2 = np.cos((X/(period/2))*np.pi*2)
    A3 = np.sin((X/(period/3))*np.pi*2)
    B3 = np.cos((X/(period/3))*np.pi*2)
    A4 = np.sin((X/(period/4))*np.pi*2)
    B4 = np.cos((X/(period/4))*np.pi*2)
   
    if n_components == 0:
        X_fit = X        
    elif n_components == 1:
        X_fit = np.column_stack((A1, B1))        
    elif n_components == 2:
        X_fit = np.column_stack((A1, B1, A2, B2))        
    elif n_components == 3:
        X_fit = np.column_stack((A1, B1, A2, B2, A3, B3))        
    else:
        X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4))

    if lin_comp and n_components:
        X_fit = np.column_stack((X, X_fit))
    
    X_fit = sm.add_constant(X_fit, has_constant='add')
    
    return X_fit
    
def population_fit(df_pop, n_components = 2, period = 24, model_type = 'lin', lin_comp= False, plot_on = True, plot_measurements=True, plot_individuals=True, plot_margins=True, save_to = ''):
 
    params = -1

    tests = df_pop.test.unique()
    k = len(tests)
    for test in tests:
        x,y = np.array(df_pop[df_pop.test == test].x), np.array(df_pop[df_pop.test == test].y)
        results, statistics, rhythm_params, X_test, Y_test, model = fit_me(x, y, n_components = n_components, period = period, model_type = model_type, lin_comp=lin_comp, plot = False, return_model = True, plot_phase=False)
        if type(params) == int:
            params = results.params
        else:
            params = np.vstack([params, results.params])
        if plot_on and plot_individuals:
            plt.plot(x, results.fittedvalues,'k', label=test)
        if plot_on and plot_measurements:
            plt.plot(x,y,'ko', markersize=1)
    


    # parameter statistics: means, variances, stadndard deviations, confidence intervals, p-values
    #http://reliawiki.com/index.php/Multiple_Linear_Regression_Analysis
    if k > 1:
        means = np.mean(params, axis=0)
        variances = np.sum((params-np.mean(params, axis=0))**2, axis = 0)/(k-1) # np.var(params, axis=0) # var deli s k in s (k-1)
        sd = variances**0.5
        se = sd/(k**0.5)
        T0 = means/se
        p_values = 2 * (1 - stats.t.cdf(abs(T0), k-1))
        t = abs(stats.t.ppf(0.05/2,df=k-1))
        lower_CI = means - ((t*sd)/(k**0.5))
        upper_CI = means + ((t*sd)/(k**0.5))
        #results, statistics, rhythm_params, X_test, Y_test, model = fit_me(x, y, n_components = n_components, period = period, model_type = model_type, lin_comp=lin_comp, plot = False, return_model = True)
        results.initialize(model, means)
    else:
        means = params
        sd = np.zeros(len(params))
        sd[:] = np.nan
        se = np.zeros(len(params))
        se[:] = np.nan
        lower_CI = means
        upper_CI = means
        p_values = np.zeros(len(params))
        p_values[:] = np.nan

    x,y = df_pop.x, df_pop.y
    xy = list(zip(x,y))
    xy.sort()
    x,y = zip(*xy)
    x,y = np.array(x), np.array(y)
    X_fit = generate_independents(x, n_components = n_components, period = period, lin_comp = lin_comp)
    Y_fit = results.predict(X_fit)
    
    if plot_on:
        plt.plot(x,Y_fit,'r', label="population fit")
        plt.legend()
        plt.xlabel('time [h]')
        plt.ylabel('measurements')

        

    if plot_on and plot_margins:
        sdev, lower, upper = wls_prediction_std(results, exog=X_fit, alpha=0.05)
        plt.fill_between(x, lower, upper, color='#888888', alpha=0.1)                   
    

    
    statistics = calculate_statistics(x, y, Y_fit, n_components, period, lin_comp) 
    statistics_params = {'values': means,
                        'SE': se,
                        'CI': (lower_CI, upper_CI),
                        'p-values': p_values} 
    if plot_on:
        pop_name = "_".join(test.split("_")[:-1])
        plt.title(pop_name + ', p-value=' + "{0:.5f}".format(statistics['p']))

        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()
    






    X_test = np.linspace(0, 100, 1000)
    X_fit_eval_params = generate_independents(X_test, n_components = n_components, period = period, lin_comp = lin_comp)
    if lin_comp:
        X_fit_eval_params[:,1] = 0    
    Y_eval_params = results.predict(X_fit_eval_params)    
    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
    
    return params, statistics, statistics_params, rhythm_params, results

def fit_me(X, Y, n_components = 2, period = 24, model_type = 'lin', lin_comp = False, alpha = 0, name = '', save_to = '', plot=True, plot_residuals=False, plot_measurements=True, plot_margins=True, return_model = False, plot_phase = True):
    """
    ###
    # prepare the independent variables
    ###
    """
    A1 = np.sin((X/period)*np.pi*2)
    B1 = np.cos((X/period)*np.pi*2)
    A2 = np.sin((X/(period/2))*np.pi*2)
    B2 = np.cos((X/(period/2))*np.pi*2)
    A3 = np.sin((X/(period/3))*np.pi*2)
    B3 = np.cos((X/(period/3))*np.pi*2)
    A4 = np.sin((X/(period/4))*np.pi*2)
    B4 = np.cos((X/(period/4))*np.pi*2)



    X_test = np.linspace(0, 100, 1000)
    A1_test = np.sin((X_test/period)*np.pi*2)
    B1_test = np.cos((X_test/period)*np.pi*2)
    A2_test = np.sin((X_test/(period/2))*np.pi*2)
    B2_test = np.cos((X_test/(period/2))*np.pi*2)
    A3_test = np.sin((X_test/(period/3))*np.pi*2)
    B3_test = np.cos((X_test/(period/3))*np.pi*2)
    A4_test = np.sin((X_test/(period/4))*np.pi*2)
    B4_test = np.cos((X_test/(period/4))*np.pi*2)
        
    if n_components == 0:
        X_fit = X
        X_fit_test = X_test
        lin_comp = True
    elif n_components == 1:
        X_fit = np.column_stack((A1, B1))
        X_fit_test = np.column_stack((A1_test, B1_test))      
    elif n_components == 2:
        X_fit = np.column_stack((A1, B1, A2, B2))
        X_fit_test = np.column_stack((A1_test, B1_test, A2_test, B2_test))                      
    elif n_components == 3:
        X_fit = np.column_stack((A1, B1, A2, B2, A3, B3))
        X_fit_test = np.column_stack((A1_test, B1_test, A2_test, B2_test, A3_test, B3_test))
    else:
        X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4))
        X_fit_test = np.column_stack((A1_test, B1_test, A2_test, B2_test, A3_test, B3_test, A4_test, B4_test))
    
    X_fit_eval_params = X_fit_test
    
    if lin_comp and n_components:
        X_fit = np.column_stack((X, X_fit))
        X_fit_eval_params = np.column_stack((np.zeros(len(X_test)), X_fit_test))
        X_fit_test = np.column_stack((X_test, X_fit_test))                              


    #if model_type == 'lin':
    X_fit = sm.add_constant(X_fit, has_constant='add')
    X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')
    """
    ###
    # fit
    ###
    """       
    if model_type == 'lin':
        model = sm.OLS(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson':
        model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        results = model.fit()
    elif model_type =='gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(Y,X_fit, p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)
    elif model_type == 'nb_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(Y,X_fit,p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)
    else:
        #exposure = np.zeros(len(Y))
        #exposure[:] = np.mean(Y)
        #model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(), exposure = exposure)
        if alpha:
            model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        else:
            model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial())
        results = model.fit()
    
    if model_type =='gen_poisson':
        Y_fit = results.predict(X_fit)
    else:
        Y_fit = results.fittedvalues
    
    statistics = calculate_statistics(X, Y, Y_fit, n_components, period, lin_comp)
    
    Y_test = results.predict(X_fit_test)
    Y_eval_params = results.predict(X_fit_eval_params)
    
    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
    
    """
    ###
    # plot
    ###
    """
    if plot:
        
        
        ###
        if plot_measurements:                     
            plt.plot(X,Y, 'ko', markersize=1, label = 'data')
        #plt.plot(X, results.fittedvalues, label = 'fit')
        
        
        plt.plot(X_test, Y_test, 'k', label = 'fit')
        if plot_measurements:
            plt.axis([min(min(X),0), 1.1*max(max(X),period), 0.9*min(min(Y), min(Y_test)), 1.1*max(max(Y), max(Y_test))])
        else:
            plt.axis([min(X_test), 50, min(Y_test)*0.9, max(Y_test)*1.1])
        #plt.title(name + ', components=' + str(n_components) +' , period=' + str(period) + '\np-value=' + str(statistics['p']) + ', p-value(gof)=' + str(statistics['p_reject']))
        #plt.title(name + ', components=' + str(n_components) +' , period=' + str(period) + '\np-value=' + str(statistics['p']))
        plt.title(name + ', p-value=' + "{0:.5f}".format(statistics['p']))
        plt.xlabel('time [h]')
        plt.ylabel('measurements')
        #fig = plt.gcf()
        #fig.set_size_inches(11,8)               
        

        if plot_margins:
            sdev, lower, upper = wls_prediction_std(results, exog=X_fit_test, alpha=0.05)
            plt.fill_between(X_test, lower, upper, color='#888888', alpha=0.1)
        
        
        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()

        if plot_residuals:          
            resid = results.resid
            fig = sm.qqplot(resid)
            plt.title(name)
            if save_to:
                plt.savefig(save_to+'_resid.png')                
                plt.close()
            else:
                plt.show()
        
        if plot_phase:
            per = rhythm_params['period']
            amp = rhythm_params['amplitude']
            phase = rhythm_params['acrophase']
            if save_to:
                plot_phases([phase], [amp], [name], period=per, folder="\\".join(save_to.split("\\")[:-1]))
            else:
                plot_phases([phase], [amp], [name], period=per)

    if return_model: 
        return results, statistics, rhythm_params, X_test, Y_test, model
    else:    
        return results, statistics, rhythm_params, X_test, Y_test


def phase_to_radians(phase, period=24):
    return -(phase/period)*2*np.pi

def acrophase_to_hours(acrophase, period=24):
    return -period * acrophase/(2*np.pi)


#r = np.linspace(0,2*np.pi,1000)
#X = np.sin(r)

def plot_phases(acrs, amps, tests, period=24, colors = ("black", "red", "green", "blue"), folder = "", prefix="", legend=True, CI_acrs = [], CI_amps = []):

    acrs = np.array(acrs)
    amps = np.array(amps)
    
    if colors and len(colors) < len(tests):
        colors += ("black",) * (len(tests)-len(colors))

    x = np.arange(0, 2*np.pi, np.pi/4)
    x_labels = list(map(lambda i: 'CT ' + str(i) + " ", list((x/(2*np.pi) * period).astype(int))))
    x_labels[1::2] = [""]*len(x_labels[1::2])

    ampM = max(amps)
    amps /= ampM
    
    acrs = -acrs
    
    ax = plt.subplot(111, projection='polar')        
    ax.set_theta_offset(0.5*np.pi)
    ax.set_theta_direction(-1) 

    for i, (acr, amp, test, color) in enumerate(zip(acrs, amps, tests, colors)):
        
        """
        if "LDL" in test:
            color = "#FF0000"
        elif "HDL" in test:
            color = "#0000FF"
        elif "CHOL" in test:
            color = "#00FF00"
        elif "control" in test.lower():
            color = "#000000"
        else:
            color = "#0000FF"            
        """

        ax.plot([acr, acr], [0, amp], label=test, color=color)
    
        ax.annotate("", xy=(acr, amp), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, linewidth=2) )
        
        if CI_acrs and CI_amps:
            amp_l, amp_u = np.array(CI_amps[i])/ampM
            amp_l = max(0, amp_l)
            amp_u = min(1, amp_u)
                   
            acr_l, acr_u = -np.array(CI_acrs[i])
             
            if acr_l - acr_u > 2*np.pi:
                plt.fill_between(np.linspace(0, np.pi*2, 1000), amp_l, amp_u, color=color, alpha=0.1)
            elif acr_u < acr_l:
                acr_l, acr_u = acr_u, acr_l
                plt.fill_between(np.linspace(acr_l, acr_u, 1000), amp_l, amp_u, color=color, alpha=0.1)



    ax.set_rmax(1)
    ax.set_rticks([0.5])  # Less radial ticks
    ax.set_yticklabels([""])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(True)
    ax.set_facecolor('#f0f0f0')
       
      
    name = "_".join(tests)
    ax.set_title(name, va='bottom')

    if legend:
        ax.legend()
    if folder:
        plt.savefig(folder+"\\"+prefix+name+"_phase.pdf")
        plt.savefig(folder+"\\"+prefix+name+"_phase.png")
        plt.close()
    else:
        plt.show()
         
def evaluate_rhythm_params(X,Y):
    m = min(Y)
    M = max(Y)
    A = M - m
    MESOR = m + A/2
    AMPLITUDE = A/2
    
    PERIOD = 0
    PHASE = 0
    
    locs, heights = signal.find_peaks(Y, height = M * 0.99)
    heights = heights['peak_heights'] 
    
    if len(locs) >= 2:
        PERIOD = X[locs[1]] - X[locs[0]]
        PERIOD = int(round(PERIOD))
    
    if len(locs) >= 1:
       PHASE = X[locs[0]]
    
    if PERIOD:
        ACROPHASE = phase_to_radians(PHASE, PERIOD)
    else:
        ACROPHASE = np.nan
    
    return {'period':PERIOD, 'amplitude':AMPLITUDE, 'acrophase':ACROPHASE, 'mesor':MESOR}
    
def calculate_statistics(X, Y, Y_fit, n_components, period, lin_comp = False):
    # statistics according to Cornelissen (eqs (8) - (9))
    MSS = sum((Y_fit - Y.mean())**2)
    RSS = sum((Y - Y_fit)**2)

    n_params = n_components * 2 + 1
    if lin_comp:
        n_params += 1            
    N = Y.size

    F = (MSS/(n_params - 1)) / (RSS/(N - n_params)) 
    p = 1 - stats.f.cdf(F, n_params - 1, N - n_params);
    #print("p-value(Cornelissen): {}".format(p))
    
    # statistics of GOF according to Cornelissen (eqs (14) - (15))
    # TODO: ali bi bilo potrebno popraviti za lumicycle - ko je več zaporednih meritev v eni časovni točki?
    #X_periodic = (X % period).astype(int)
    X_periodic = np.round_(X % period,2)                                    
    
    X_unique = np.unique(X_periodic)
    n_T = len(X_unique)
    
    SSPE = 0
    for x in X_unique:
        Y_i_avg = np.mean(Y[X_periodic == x])
        SSPE += sum((Y[X_periodic == x] - Y_i_avg)**2)
    SSLOF = RSS-SSPE
    #print('RSS: ', RSS)
    #print('SSPE: ', SSPE)
    #print('SSLOF: ', SSLOF)
    if lin_comp:
        F = (SSLOF/(n_T-1-(2*n_components + 1)))/(SSPE/(N-n_T))
        p_reject = 1 - stats.f.cdf(F, n_T-1-(2*n_components + 1), N-n_T)
    else:    
        try:
            F = (SSLOF/(n_T-1-2*n_components))/(SSPE/(N-n_T))
            p_reject = 1 - stats.f.cdf(F, n_T-1-2*n_components, N-n_T)
        except:
            F = np.nan
            p_reject = np.nan
        
    
    # Another measure that describes goodnes of fit
    # How well does the curve describe the data?
    # signal to noise ratio
    # fitted curve: signal
    # noise: 
    stdev_data = np.std(Y, ddof = 1)
    stdev_fit = np.std(Y_fit, ddof = 1)
    SNR = stdev_fit / stdev_data
    
    # Standard Error of residuals, margin of error
    # https://stats.stackexchange.com/questions/57746/what-is-residual-standard-error
    DoF = N - n_params
    resid_SE = np.sqrt(RSS/DoF)
    # https://scientificallysound.org/2017/05/16/independent-t-test-python/
    # https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/margin-of-error/
    critical_value = t.ppf(1-0.025, DoF)
    ME = critical_value * resid_SE
    
    
    return {'p':p, 'p_reject':p_reject, 'SNR':SNR, 'RSS': RSS, 'resid_SE': resid_SE, 'ME': ME}

def calculate_statistics_nonlinear(X, Y, Y_fit, n_params, period):
    # statistics according to Cornelissen (eqs (8) - (9))
    MSS = sum((Y_fit - Y.mean())**2)
    RSS = sum((Y - Y_fit)**2)
    N = Y.size

    F = (MSS/(n_params - 1)) / (RSS/(N - n_params)) 
    p = 1 - stats.f.cdf(F, n_params - 1, N - n_params);
    
    X_periodic = np.round_(X % period,2)                                    
    
    X_unique = np.unique(X_periodic)
    n_T = len(X_unique)
    
    SSPE = 0
    for x in X_unique:
        Y_i_avg = np.mean(Y[X_periodic == x])
        SSPE += sum((Y[X_periodic == x] - Y_i_avg)**2)
    SSLOF = RSS-SSPE
    #print('RSS: ', RSS)
    #print('SSPE: ', SSPE)
    #print('SSLOF: ', SSLOF)
    F = (SSLOF/(n_T-n_params))/(SSPE/(N-n_T))
    p_reject = 1 - stats.f.cdf(F, n_T-n_params, N-n_T)
    
    
    # Another measure that describes goodnes of fit
    # How well does the curve describe the data?
    # signal to noise ratio
    # fitted curve: signal
    # noise: 
    stdev_data = np.std(Y, ddof = 1)
    stdev_fit = np.std(Y_fit, ddof = 1)
    SNR = stdev_fit / stdev_data
    
    
    # Standard Error of residuals, margin of error
    # https://stats.stackexchange.com/questions/57746/what-is-residual-standard-error
    DoF = N - n_params
    resid_SE = np.sqrt(RSS/DoF)
    # https://scientificallysound.org/2017/05/16/independent-t-test-python/
    # https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/margin-of-error/
    critical_value = t.ppf(1-0.025, DoF)
    ME = critical_value * resid_SE
    
    
    return {'p':p, 'p_reject':p_reject, 'SNR':SNR, 'RSS': RSS, 'resid_SE': resid_SE, 'ME': ME}
    
    




"""
primerjava med rezimi:
- LymoRhyde (Singer:2019)
"""
def compare_pairs(df, pairs, n_components = 3, period = 24, lin_comp = False, folder = '', prefix = '', plot_measurements=True):
    
    df_results = pd.DataFrame()

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    for test1, test2 in pairs: 
        for per in period:
            for n_comps in n_components:                                
                if folder:
                    save_to = folder + '\\' + prefix + test1 + '-' + test2 + '_per=' + str(per) + '_comps=' + str(n_comps)
                else:
                    save_to = ''
                
                pvalues, params, results = compare_pair_df(df, test1, test2, n_components = n_comps, period = per, lin_comp = lin_comp, save_to = save_to, plot_measurements=plot_measurements)
                
                d = {}
                d['test'] = test1 + ' vs. ' + test2
                d['period'] = per
                d['n_components'] = n_comps
                for i, (param, p) in enumerate(zip(params, pvalues)):
                    d['param' + str(i+1)] = param
                    d['p' + str(i+1)] = p
                
                
                df_results = df_results.append(d, ignore_index=True)
  
    
    
    for i, (param, p) in enumerate(zip(params, pvalues)):        
        df_results['q'+str(i+1)] = multi.multipletests(df_results['p'+str(i+1)], method = 'fdr_bh')[1]
    
    
    columns = df_results.columns
    columns = columns.sort_values()
    columns = np.delete(columns, np.where(columns == 'period'))
    columns = np.append(['period'], columns)
    columns = np.append([columns[-1]], columns[:-1])
    
    df_results = df_results.reindex(columns, axis=1)
    
    return df_results

    #return multi.multipletests(P, method = 'fdr_bh')[1]



def compare_pair_df(df, test1, test2, n_components = 3, period = 24, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot_measurements=True, plot_residuals=False):
    df_pair = df[(df.test == test1) | (df.test == test2)].copy()
    df_pair['h_i'] = 0
    df_pair.loc[df_pair.test == test2, 'h_i'] = 1
    
    
    X = df_pair.x
    Y = df_pair.y
    H_i = df_pair.h_i
    
    """
    ###
    # prepare the independent variables
    ###
    """
    A1 = np.sin((X/period)*np.pi*2)
    B1 = np.cos((X/period)*np.pi*2)
    A2 = np.sin((X/(period/2))*np.pi*2)
    B2 = np.cos((X/(period/2))*np.pi*2)
    A3 = np.sin((X/(period/3))*np.pi*2)
    B3 = np.cos((X/(period/3))*np.pi*2)
    A4 = np.sin((X/(period/4))*np.pi*2)
    B4 = np.cos((X/(period/4))*np.pi*2)
   
    
    X_i = H_i * X              
    A1_i = H_i * A1
    B1_i = H_i * B1
    A2_i = H_i * A2
    B2_i = H_i * B2
    A3_i = H_i * A3
    B3_i = H_i * B3
    A4_i = H_i * A4
    B4_i = H_i * B4
                  
    if n_components == 1:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, H_i))
            idx_params = [3]
        else:
            X_fit = np.column_stack((A1, B1, A1_i, B1_i, H_i))
            idx_params = [3, 4, 5]
            
           
        
    elif n_components == 2:        
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, H_i))
            idx_params = [5]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A1_i, B1_i, A2_i, B2_i, H_i))
            #idx_params = [5, 6, 7, 8, 9]
            idx_params = [5, 6, 7, 8]
            
                    
    elif n_components == 3:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, H_i))
            idx_params = [7]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, H_i))
            #idx_params = [7, 8, 9, 10, 11, 12, 13]
            idx_params = [7, 8, 9, 10, 11, 12]
    else:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4, H_i))
            idx_params = [9]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4, A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, A4_i, B4_i, H_i))
            #idx_params = [9, 10, 11, 12, 13, 14, 15, 16, 17]
            idx_params = [9, 10, 11, 12, 13, 14, 15, 16]
           
        
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))
        idx_params = np.array(idx_params) + 2                                
    
    X_fit = sm.add_constant(X_fit, has_constant='add')

    """
    ###
    # fit
    ###
    """       
    if model_type == 'lin':
        model = sm.OLS(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson':
        model = sm.GLM(Y, X, family=sm.families.Poisson())
        results = model.fit()
    elif model_type =='gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X)
        results = model.fit()
    elif model_type == 'poisson_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(Y,X, p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)
    elif model_type == 'nb_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(Y,X,p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)
    else:
        #exposure = np.zeros(len(Y))
        #exposure[:] = np.mean(Y)
        #model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(), exposure = exposure)
        if alpha:
            model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(alpha=alpha))
        else:
            model = sm.GLM(Y, X, family=sm.families.NegativeBinomial())
        results = model.fit()
    
    
    """
    ###
    # plot
    ###
    """
    
    
    ###
    if plot_measurements:
        plt.plot(df_pair[df_pair.test == test1].x, df_pair[df_pair.test == test1].y, 'ko', markersize=1)
        plt.plot(df_pair[df_pair.test == test2].x, df_pair[df_pair.test == test2].y, 'ro', markersize=1)
    #plt.plot(X, results.fittedvalues, label = 'fit')
    
    if model_type =='gen_poisson':
        Y_fit = results.predict(X)
    else:
        Y_fit = results.fittedvalues
    
    
    
    X1 = X[H_i == 0]
    #Y_fit1 = Y_fit[H_i == 0]
    #L1 = list(zip(X1,Y_fit1))
    #L1.sort()
    #X1,Y_fit1 = list(zip(*L1))  
    X2 = X[H_i == 1]
    #Y_fit2 = Y_fit[H_i == 1]
    #L2 = list(zip(X2,Y_fit2))
    #L2.sort()
    #X2,Y_fit2 = list(zip(*L2))  
    
    
    
    #plt.plot(X1, Y_fit1, 'k', label = 'fit '+test1)    
    #plt.plot(X2, Y_fit2, 'r', label = 'fit '+test2)    
    
    ### plot with higher density
    
    n_points = 1000
    X_full = np.linspace(min(min(X1),min(X2)), max(max(X1), max(X2)), n_points)
    
    X_fit_full = generate_independents_compare(X_full, X_full, n_components1 = n_components, period1 = period, n_components2 = n_components, period2 = period, lin_comp= lin_comp)
    Y_fit_full = results.predict(X_fit_full)
    
    plt.plot(X_full, Y_fit_full[0:n_points], 'k', label = test1)    
    plt.plot(X_full, Y_fit_full[n_points:], 'r', label = test2)    
    ### end of plot with higher density
    
    
    p = min(results.pvalues[idx_params])
    #plt.title(test1 + ' vs. ' + test2 + '\ncomponents=' + str(n_components) +' , period=' + str(period)+', p-value=' + str(p))
    plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p))
    plt.xlabel('time [h]')
    plt.ylabel('measurements')
    plt.legend()
    
    #fig = plt.gcf()
    #fig.set_size_inches(11,8)
    
    if save_to:
        plt.savefig(save_to+'.png')
        plt.savefig(save_to+'.pdf')
        plt.close()
    else:
        plt.show()
    
    if plot_residuals:
        
        resid = results.resid
        fig = sm.qqplot(resid)
        plt.title(test1 + ' vs. ' + test2)
        save_to_resid = save_to.split(".")[0] + '_resid' + save_to.split(".")[1]
        if save_to:
            plt.savefig(save_to_resid)
            plt.close()
        else:
            plt.show()
    
    
    return (results.pvalues[idx_params], results.params[idx_params], results)


def compare_pair(X1, Y1, X2, Y2, test1 = '', test2 = '', n_components = 3, period = 24, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot_measurements=True, plot_residuals=False):
    
    
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
       
    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))
    H_i = np.concatenate((H1, H2))
    
    """
    ###
    # prepare the independent variables
    ###
    """
    A1 = np.sin((X/period)*np.pi*2)
    B1 = np.cos((X/period)*np.pi*2)
    A2 = np.sin((X/(period/2))*np.pi*2)
    B2 = np.cos((X/(period/2))*np.pi*2)
    A3 = np.sin((X/(period/3))*np.pi*2)
    B3 = np.cos((X/(period/3))*np.pi*2)
    A4 = np.sin((X/(period/4))*np.pi*2)
    B4 = np.cos((X/(period/4))*np.pi*2)
   
    X_i = H_i * X
    A1_i = H_i * A1
    B1_i = H_i * B1
    A2_i = H_i * A2
    B2_i = H_i * B2
    A3_i = H_i * A3
    B3_i = H_i * B3
    A4_i = H_i * A4
    B4_i = H_i * B4
                  
    if n_components == 1:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, H_i))
            idx_params = [3]
        else:
            X_fit = np.column_stack((A1, B1, A1_i, B1_i, H_i))
            #idx_params = [3, 4, 5]
            idx_params = [3, 4]
            
           
        
    elif n_components == 2:        
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, H_i))
            idx_params = [5]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A1_i, B1_i, A2_i, B2_i, H_i))
            #idx_params = [5, 6, 7, 8, 9]
            idx_params = [5, 6, 7, 8]
            
                    
    elif n_components == 3:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, H_i))
            idx_params = [7]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, H_i))
            #idx_params = [7, 8, 9, 10, 11, 12, 13]
            idx_params = [7, 8, 9, 10, 11, 12]
    else:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4, H_i))
            idx_params = [9]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4, A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, A4_i, B4_i, H_i))
            #idx_params = [9, 10, 11, 12, 13, 14, 15, 16, 17]
            idx_params = [9, 10, 11, 12, 13, 14, 15, 16]
           
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))
        idx_params = np.array(idx_params) + 2
        
    X_fit = sm.add_constant(X_fit, has_constant='add')

    """
    ###
    # fit
    ###
    """       
    if model_type == 'lin':
        model = sm.OLS(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson':
        model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        results = model.fit()
    elif model_type =='gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(Y,X_fit, p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)
    elif model_type == 'nb_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(Y,X_fit,p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)
    else:
        #exposure = np.zeros(len(Y))
        #exposure[:] = np.mean(Y)
        #model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(), exposure = exposure)
        if alpha:
            model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        else:
            model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial())
        results = model.fit()
    
    
    """
    ###
    # plot
    ###
    """
    
    
    ###
    if plot_measurements:
        plt.plot(X1, Y1, 'ko', markersize=1, label = test1)
        plt.plot(X2, Y2, 'ro', markersize=1, label = test2)
    #plt.plot(X, results.fittedvalues, label = 'fit')
    
    if model_type =='gen_poisson':
        Y_fit = results.predict(X_fit)
    else:
        Y_fit = results.fittedvalues
    
  
    
    X1 = X[H_i == 0]
    Y_fit1 = Y_fit[H_i == 0]
    L1 = list(zip(X1,Y_fit1))
    L1.sort()
    X1,Y_fit1 = list(zip(*L1))  
    X2 = X[H_i == 1]
    Y_fit2 = Y_fit[H_i == 1]
    L2 = list(zip(X2,Y_fit2))
    L2.sort()
    X2,Y_fit2 = list(zip(*L2))  
    
    
    plt.plot(X1, Y_fit1, 'k', label = 'fit '+test1)    
    plt.plot(X2, Y_fit2, 'r', label = 'fit '+test2)    
    
    p = min(results.pvalues[idx_params])
    #plt.title(test1 + ' vs. ' + test2 + '\ncomponents=' + str(n_components) +' , period=' + str(period)+', p-value=' + str(p))
    plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p))
    plt.xlabel('time [h]')
    plt.ylabel('measurements')
    plt.legend()
    
    #fig = plt.gcf()
    #fig.set_size_inches(11,8)
    
    
    if save_to:
        plt.savefig(save_to+'.png')
        plt.savefig(save_to+'.pdf')
        plt.close()
    else:
        plt.show()
        
    if plot_residuals:
        
        resid = results.resid
        fig = sm.qqplot(resid)
        plt.title(test1 + ' vs. ' + test2)
        save_to_resid = save_to.split(".")[0] + '_resid' + save_to.split(".")[1]
        if save_to:
            plt.savefig(save_to_resid)
            plt.close()
        else:
            plt.show()
    
        
    return (results.pvalues[idx_params], results.params[idx_params], results)

def generate_independents_compare(X1, X2, n_components1 = 3, period1 = 24, n_components2 = 3, period2 = 24, lin_comp = False, non_rhythmic=False):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    
    X = np.concatenate((X1, X2))
    H_i = np.concatenate((H1, H2))
   
    A1 = np.sin((X/period1)*np.pi*2)
    B1 = np.cos((X/period1)*np.pi*2)
    A2 = np.sin((X/(period1/2))*np.pi*2)
    B2 = np.cos((X/(period1/2))*np.pi*2)
    A3 = np.sin((X/(period1/3))*np.pi*2)
    B3 = np.cos((X/(period1/3))*np.pi*2)
    A4 = np.sin((X/(period1/4))*np.pi*2)
    B4 = np.cos((X/(period1/4))*np.pi*2)
    
    X_i = H_i * X
    A1_i = H_i * np.sin((X/period2)*np.pi*2)
    B1_i = H_i * np.cos((X/period2)*np.pi*2)
    A2_i = H_i * np.sin((X/(period2/2))*np.pi*2)
    B2_i = H_i * np.cos((X/(period2/2))*np.pi*2)
    A3_i = H_i * np.sin((X/(period2/3))*np.pi*2)
    B3_i = H_i * np.cos((X/(period2/3))*np.pi*2)
    A4_i = H_i * np.sin((X/(period2/4))*np.pi*2)
    B4_i = H_i * np.cos((X/(period2/4))*np.pi*2)
   
    if n_components1 == 1:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, H_i))            
        else:
            X_fit = np.column_stack((A1, B1))            
    elif n_components1 == 2:        
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, H_i))            
        else:
            X_fit = np.column_stack((A1, B1, A2, B2))            
    elif n_components1 == 3:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, H_i))            
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3))            
    else:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4, H_i))            
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4))            
        
    if n_components2 == 1:
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, H_i))))            
    elif n_components2 == 2:        
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, A2_i, B2_i, H_i))))                                    
    elif n_components2 == 3:
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, H_i))))
    else:
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, A4_i, B4_i, H_i))))
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))    
    X_fit = sm.add_constant(X_fit, has_constant='add')

    return X_fit

def compare_pair_extended(X1, Y1, X2, Y2, test1 = '', test2 = '', n_components1 = 3, period1 = 24, n_components2 = 3, period2 = 24, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot_residuals=False):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
       
    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))
    H_i = np.concatenate((H1, H2))
    
    """
    ###
    # prepare the independent variables
    ###
    """
    A1 = np.sin((X/period1)*np.pi*2)
    B1 = np.cos((X/period1)*np.pi*2)
    A2 = np.sin((X/(period1/2))*np.pi*2)
    B2 = np.cos((X/(period1/2))*np.pi*2)
    A3 = np.sin((X/(period1/3))*np.pi*2)
    B3 = np.cos((X/(period1/3))*np.pi*2)
    A4 = np.sin((X/(period1/4))*np.pi*2)
    B4 = np.cos((X/(period1/4))*np.pi*2)
   
    X_i = H_i * X
    A1_i = H_i * np.sin((X/period2)*np.pi*2)
    B1_i = H_i * np.cos((X/period2)*np.pi*2)
    A2_i = H_i * np.sin((X/(period2/2))*np.pi*2)
    B2_i = H_i * np.cos((X/(period2/2))*np.pi*2)
    A3_i = H_i * np.sin((X/(period2/3))*np.pi*2)
    B3_i = H_i * np.cos((X/(period2/3))*np.pi*2)
    A4_i = H_i * np.sin((X/(period2/4))*np.pi*2)
    B4_i = H_i * np.cos((X/(period2/4))*np.pi*2)
                  
    if n_components1 == 1:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, H_i))
            idx_params = [-1]
        else:
            X_fit = np.column_stack((A1, B1))            
    elif n_components1 == 2:        
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, H_i))
            idx_params = [-1]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2))            
    elif n_components1 == 3:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, H_i))
            idx_params = [-1]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3))            
    else:
        if non_rhythmic:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4, H_i))
            idx_params = [-1]
        else:
            X_fit = np.column_stack((A1, B1, A2, B2, A3, B3, A4, B4))            
        
    if n_components2 == 1:
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, H_i))))
            #idx_params = [-3, -2, -1]
            idx_params = [-3, -2]
    elif n_components2 == 2:        
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, A2_i, B2_i, H_i))))                        
            #idx_params = [-5, -4, -3, -2, -1]            
            idx_params = [-5, -4, -3, -2]            
    elif n_components2 == 3:
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, H_i))))                                    
            #idx_params = [-7, -6, -5, -4, -3, -2, -1]
            idx_params = [-7, -6, -5, -4, -3, -2]
    else:
        if not non_rhythmic:
            X_fit = np.column_stack((X_fit, np.column_stack((A1_i, B1_i, A2_i, B2_i, A3_i, B3_i, A4_i, B4_i, H_i))))
            #idx_params = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
            idx_params = [-9, -8, -7, -6, -5, -4, -3, -2]
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))    
    X_fit = sm.add_constant(X_fit, has_constant='add')

    """
    ###
    # fit
    ###
    """       
    if model_type == 'lin':
        model = sm.OLS(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson':
        model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        results = model.fit()
    elif model_type =='gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit)
        results = model.fit()        
    elif model_type == 'poisson_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(Y,X_fit, p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)    
    elif model_type == 'nb_zeros':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(Y,X_fit,p=2)
        #results = model.fit()
        results = model.fit(method='bfgs', maxiter=5000, maxfun=5000)
    else:
        #exposure = np.zeros(len(Y))
        #exposure[:] = np.mean(Y)
        #model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(), exposure = exposure)
        if alpha:
            model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        else:
            model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial())
        results = model.fit()
    
    
    """
    ###
    # plot
    ###
    """
    
    
    ###
    
    plt.plot(X1, Y1, 'ko', markersize=1, label = test1)
    plt.plot(X2, Y2, 'ro', markersize=1, label = test2)
    #plt.plot(X, results.fittedvalues, label = 'fit')
    
    if model_type =='gen_poisson':
        Y_fit = results.predict(X_fit)
    else:
        Y_fit = results.fittedvalues

    X1 = X[H_i == 0]
    Y_fit1 = Y_fit[H_i == 0]
    L1 = list(zip(X1,Y_fit1))
    L1.sort()
    X1,Y_fit1 = list(zip(*L1))  
    X2 = X[H_i == 1]
    Y_fit2 = Y_fit[H_i == 1]
    L2 = list(zip(X2,Y_fit2))
    L2.sort()
    X2,Y_fit2 = list(zip(*L2))  
    plt.plot(X1, Y_fit1, 'k', label = 'fit '+test1)    
    plt.plot(X2, Y_fit2, 'r', label = 'fit '+test2)    
    
    # if generalized poisson last parameter is alpha!!
    if model_type =='gen_poisson':
        idx_params = np.array(idx_params) - 1
    
    p = min(results.pvalues[idx_params])
    plt.title(test1 + ' vs. ' + test2 + ', components1=' + str(n_components1) + ', components2=' + str(n_components2) +'\nperiod1=' + str(period1) + ', period2=' + str(period2)+', p-value=' + str(p))
    plt.xlabel('time [h]')
    plt.ylabel('measurements')
    plt.legend()
    
    if save_to:
        plt.savefig(save_to+'.png')
        plt.savefig(save_to+'.pdf')
        plt.close()
    else:
        plt.show()
    
    #fig = plt.gcf()
    #fig.set_size_inches(11,8)
    
    if plot_residuals:
        
        resid = results.resid
        fig = sm.qqplot(resid)
        plt.title(test1 + ' vs. ' + test2)
        save_to_resid = save_to.split(".")[0] + '_resid' + save_to.split(".")[1]
        if save_to:
            plt.savefig(save_to_resid)
            plt.close()
        else:
            plt.show()
    
    
    """
    print(test1, 'vs', test2)
    print('components1=' + str(n_components1) + ', components2=' + str(n_components2))
    print(results.pvalues[idx_params])
    print(results.summary())
    print("######################")
    """
    return (results.pvalues[idx_params], results.params[idx_params], results)
    



# compare two models according to the F-test
# http://people.reed.edu/~jones/Courses/P24.pdf
# https://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?reg_howtheftestworks.htm  
def get_best_models(df, df_models, n_components = [1,2,3], lin_comp = False, criterium='p', reverse = True):
       
    names = np.unique(df_models.test)   
    df_best = pd.DataFrame(columns = df_models.columns)
    df_models = get_best_fits(df_models, n_components = n_components, criterium=criterium, reverse = reverse)


    for test in names:  
        n_points = df[df.test == test].x.shape[0]
        df_test_models = df_models[df_models.test == test]
        df_test_models = df_test_models.sort_values(by=['n_components'])
        i = 0
        for new_row in df_test_models.iterrows():            
            if i == 0:
                best_row = new_row
                i = 1
            else:
                RSS_reduced = best_row[1].RSS
                RSS_full = new_row[1].RSS

                DF_reduced = n_points - (best_row[1].n_components * 2 + 1)
                DF_full = n_points - (new_row[1].n_components * 2 + 1)

                if lin_comp:
                    DF_reduced -= 1
                    DF_full -= 1                
                #print (test, old_row[1].n_components, new_row[1].n_components)
                if compare_models(RSS_reduced, RSS_full, DF_reduced, DF_full) < 0.05:
                    best_row = new_row
                    
                
                    
                    
        df_best = df_best.append(best_row[1], ignore_index=True)
    
    return df_best

def plot_df_models(df, df_models, plot_residuals=True, folder =""):
    for row in df_models.iterrows():
        test = row[1].test
        n_components = row[1].n_components
        period = row[1].period
        X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)   
        
        if folder:
            fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = folder+'\\'+test+'_compnts='+str(n_components) +'_per=' + str(period), plot_residuals = plot_residuals)
        else:
            fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = "", plot_residuals = plot_residuals)
    
def plot_df_models_population(df, df_models, folder=""):
    for row in df_models.iterrows():
        pop = row[1].test
        n_components = row[1].n_components
        period = row[1].period
        #X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)  
        df_pop = df[df.test.str.startswith(pop)]
        if folder:
            save_to = folder+'\\'+pop+'_pop_compnts='+str(n_components) +'_per=' + str(period)
        else:
            save_to = ""
        population_fit(df_pop, n_components = n_components, period = period, save_to = save_to)

def compare_models(RSS1, RSS2, DF1, DF2):
    if DF2 < DF1:
        F = ((RSS1 - RSS2)/(DF1 - DF2))/(RSS2/DF2)
        return 1 - stats.f.cdf(F, DF1 - DF2, DF2)
    else:
        F = ((RSS2 - RSS1)/(DF2 - DF1))/(RSS1/DF1)
        return 1 - stats.f.cdf(F, DF2 - DF1, DF1)


def ct_response(y, mu):
    return ((y-mu)**2 - y) / mu

def ct_test(count, poiss_results):

    mu = poiss_results.mu
    y = count
    ct = ct_response(y, mu)

    ct_data=pd.DataFrame()
    ct_data['ct_resp'] = ct
    ct_data['mu'] = mu
    ct_results = smf.ols('ct_resp ~ mu - 1', ct_data).fit()
    alpha_ci95 = ct_results.conf_int(0.05).loc['mu']
    print('\nC-T dispersion test: alpha = {:5.3f}, 95% CI = ({:5.3f}, {:5.3f})'.format(ct_results.params[0], alpha_ci95.loc[0], alpha_ci95.loc[1]))
    alpha = ct_results.params[0]
    
    return alpha       


"""
primerjava med rezimi:
- ANOVA. Razmerje med variancami povprecij skupin in varianco na celotnem vzorcu.
- Verjetno v redu, ce menjava rezima vpliva na amplitudo, ne pa, ce menjava vpliva na fazo ali periodo
"""
def compare_ANOVA(df, pairs, n_components = 3, period = 24):
    # https://pythonfordatascience.org/anova-python/
    # http://www.statistik.si/storitve/statisticne-analize-testi/anova-analiza-variance/
    # https://www.youtube.com/watch?v=-yQb_ZJnFXw
    P = []

    for test1, test2 in pairs: 
        """
        df2 = df[(df['test'] == test1) | (df['test'] == test2)].copy()
        df2['A'] = np.sin((df2['x']/period)*np.pi*2)
        df2['B'] = np.cos((df2['x']/period)*np.pi*2)
        if n_components >= 2:
            df2['C'] = np.sin((df2['x']/(period/2))*np.pi*2)
            df2['D'] = np.cos((df2['x']/(period/2))*np.pi*2)
        if n_components >= 3:
            df2['E'] = np.sin((df2['x']/(period/3))*np.pi*2)
            df2['F'] = np.cos((df2['x']/(period/3))*np.pi*2)
        """
        P.append(stats.f_oneway(df['y'][df['test'] == test1], df['y'][df['test'] == test2]).pvalue)

        #results = smf.ols('y ~ test', data = df[(df['test'] == test1) | (df['test'] == test2)]).fit()
        #print(results.summary())

    return multi.multipletests(P, method = 'fdr_bh')[1]

#https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
#https://pythonfordatascience.org/anova-2-way-n-way/
def compare_ANOVA2(df, pairs):
    P = []
	
    for test1, test2 in pairs:
        data = df[(df['test'] == test1) | (df['test'] == test2)]
        formula = 'y ~ x + test + x:test'
        model = smf.ols(formula, data).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        
        P.append(aov_table['PR(>F)']['x:test'])  
        #P.append(model.pvalues[-1])
	
	
    return multi.multipletests(P, method = 'fdr_bh')[1]
	
def test_phase(X1, Y1, X2, Y2, phase, period = 0, test1 = '', test2 = ''):
    X2 -= phase
    if period:
        X1 %= period
        X2 %= period
    
    # test ali so variance enake - nasprotno od ANOVA
    # t-test v vsaki tocki. V vseh tockah moras zavrniti hipotezo, da sta porazdelitvi razlicni


def amp_phase_fit(predictor, Amp, Per, Phase, Base, dAmp, dPhase, dBase):
    X = predictor[0]
    H = predictor[1]
    return (Amp + H*dAmp)*np.cos((2*np.pi*X/Per) + Phase + H * dPhase) + Base + H*dBase

def phase_fit(predictor, Amp, Per, Phase, Base, dPhase, dBase):
    X = predictor[0]
    H = predictor[1]
    return Amp*np.cos((2*np.pi*X/Per) + Phase + H * dPhase) + Base + H*dBase
    
def amp_fit(predictor, Amp, Per, Phase, Base, dAmp, dBase):
    X = predictor[0]
    H = predictor[1]
    return (Amp + H*dAmp)*np.cos((2*np.pi*X)/Per + Phase) + Base + H*dBase

def per_fit(predictor, Amp, Per, Phase, Base, dPer, dBase):
    X = predictor[0]
    H = predictor[1]
    return Amp*np.cos((2 * np.pi *X)/(Per+ H*dPer) + Phase) + Base + H*dBase

def all_fit(predictor, Amp, Per, Phase, Base, dAmp, dPer, dPhase, dBase):
    X = predictor[0]
    H = predictor[1]
    return (Amp + H*dAmp) * np.cos((2 * np.pi *X)/(Per + H*dPer)  + Phase + H * dPhase) + Base + H*dBase

def compare_phase_pairs(df, pairs, min_per = 18, max_per = 36, folder = '', prefix='', plot_residuals=False):
    df_results = pd.DataFrame()

    for test1, test2 in pairs: 
        if folder:
            save_to = folder + '\\' + prefix + test1 + '-' + test2
        else:
            save_to = ''
        
        X1, Y1 = np.array(df[df.test == test1].x), np.array(df[df.test == test1].y)
        X2, Y2 = np.array(df[df.test == test2].x), np.array(df[df.test == test2].y)
        
        statistics, d = compare_nonlinear(X1, Y1, X2, Y2, test1 = test1, test2 = test2, min_per = min_per, max_per=max_per, compare_phase = True, compare_period = False, compare_amplitude = False, save_to = save_to, plot_residuals=plot_residuals)
        
        
        d['test'] = test1 + ' vs. ' + test2
        d['p'] = statistics['p']
        d['p_reject'] = statistics['p_reject']
        d['ME'] = statistics['ME']
        d['resid_SE'] = statistics['resid_SE']
        
        
        
        df_results = df_results.append(d, ignore_index=True)

   
    
    for v in d:
        if v.startswith('p'):
            if v == "p_reject":
                q = "q_reject"
            elif v == "p":
                q = "q"
            else:
                q = v[2:].replace(")","")
                q = "q("+str(q)+")"
            
            df_results[q] = multi.multipletests(df_results[v], method = '')[1]
    
 
    columns = df_results.columns
    columns = columns.sort_values()
    columns = np.delete(columns, np.where((columns == 'test')|(columns == 'p')|(columns == 'q')|(columns == 'p_reject')|(columns == 'q_reject')))
    columns = ['test', 'p', 'q', 'p_reject', 'q_reject'] + list(columns)
    
    df_results = df_results.reindex(columns, axis=1)
    
    return df_results

    #return multi.multipletests(P, method = 'fdr_bh')[1]


def compare_nonlinear_pairs(df, pairs, min_per = 18, max_per = 36, folder = '', prefix='', plot_residuals=False):
    df_results = pd.DataFrame()

    for test1, test2 in pairs: 
        if folder:
            save_to = folder + '\\' + prefix + test1 + '-' + test2
        else:
            save_to = ''
        
        X1, Y1 = np.array(df[df.test == test1].x), np.array(df[df.test == test1].y)
        X2, Y2 = np.array(df[df.test == test2].x), np.array(df[df.test == test2].y)
        
        statistics, d = compare_nonlinear(X1, Y1, X2, Y2, test1 = test1, test2 = test2, min_per = min_per, max_per=max_per, compare_phase = True, compare_period = False, compare_amplitude = True, save_to = save_to, plot_residuals=plot_residuals)
        
        d['test'] = test1 + ' vs. ' + test2
        d['p'] = statistics['p']
        d['p_reject'] = statistics['p_reject']
        d['ME'] = statistics['ME']
        d['resid_SE'] = statistics['resid_SE']
        
        
        
        df_results = df_results.append(d, ignore_index=True)

   
    
    for v in d:
        if v.startswith('p'):
            if v == "p_reject":
                q = "q_reject"
            elif v == "p":
                q = "q"
            else:
                q = v[2:].replace(")","")
                q = "q("+str(q)+")"
            
            df_results[q] = multi.multipletests(df_results[v], method = 'fdr_bh')[1]
    
 
    columns = df_results.columns
    columns = columns.sort_values()
    columns = np.delete(columns, np.where((columns == 'test')|(columns == 'p')|(columns == 'q')|(columns == 'p_reject')|(columns == 'q_reject')))
    columns = ['test', 'p', 'q', 'p_reject', 'q_reject'] + list(columns)
    
    df_results = df_results.reindex(columns, axis=1)
    
    return df_results

    #return multi.multipletests(P, method = 'fdr_bh')[1]




def compare_nonlinear(X1, Y1, X2, Y2, test1 = '', test2 = '', min_per = 18, max_per=36, compare_phase = False, compare_period = False, compare_amplitude = False, save_to = '', plot_residuals=False):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    
    Y = np.concatenate((Y1, Y2))    
    
    X = np.concatenate((X1, X2))    
    H = np.concatenate((H1, H2))
    
    predictor = np.array([X,H])
    
    X_full = np.linspace(min(min(X1), min(X2)), max(max(X1), max(X2)), 100)
    H1 = np.zeros(X_full.size)
    H2 = np.ones(X_full.size)
    
    minimum_bounds = {'Amp':0, 
                      'Per':min_per, 
                      'Phase':0, 
                      'Base':0, 
                      'dAmp':0, 
                      'dPer':0, 
                      'dPhase':-np.pi, 
                      'dBase':0}
    maximum_bounds = {'Amp':max(Y), 
                      'Per':max_per, 
                      'Phase':2*np.pi, 
                      'Base':max(Y), 
                      'dAmp':max(Y), 
                      'dPer':max_per/2, 
                      'dPhase':np.pi, 
                      'dBase':max(Y)}
    
    if compare_phase and compare_amplitude:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dAmp', 'dPhase', 'dBase']        
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]               
        popt, pcov = curve_fit(amp_phase_fit, predictor, Y, bounds=(min_bounds, max_bounds))        
        
        Y1_full = amp_phase_fit(np.array([X_full, H1]), *popt)
        Y2_full = amp_phase_fit(np.array([X_full, H2]), *popt)    
        Y_fit = amp_phase_fit(np.array([X, H]), *popt)              
    elif compare_phase:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dPhase', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(phase_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = phase_fit(np.array([X_full, H1]), *popt)
        Y2_full = phase_fit(np.array([X_full, H2]), *popt)    
        Y_fit = phase_fit(np.array([X, H]), *popt)
    elif compare_period:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dPer', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(per_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = per_fit(np.array([X_full, H1]), *popt)
        Y2_full = per_fit(np.array([X_full, H2]), *popt)    
        Y_fit = per_fit(np.array([X, H]), *popt)
    elif compare_amplitude:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dAmp', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(amp_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = amp_fit(np.array([X_full, H1]), *popt)
        Y2_full = amp_fit(np.array([X_full, H2]), *popt)    
        Y_fit = amp_fit(np.array([X, H]), *popt)
    else:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dAmp', 'dPer', 'dPhase', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(all_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = all_fit(np.array([X_full, H1]), *popt)
        Y2_full = all_fit(np.array([X_full, H2]), *popt)    
        Y_fit = all_fit(np.array([X, H]), *popt)
    
    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), popt[1])    
    
    # Compute standard errors of parameter estimates
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # http://reliawiki.com/index.php/Multiple_Linear_Regression_Analysis
    perr = np.sqrt(np.diag(pcov))
    DoF = len(Y) - len(popt)
    p=np.zeros(len(popt))
    for i in range(len(perr)):
        T0 = popt[i]/perr[i]
        p[i] = 2 * (1 - stats.t.cdf(abs(T0), DoF))
    
    p_dict = {}
    for param, val, p_val in zip(parameters, popt, p):
        p_dict[param] = val
        p_dict["p("+param+")"] = p_val
    
    
    
    plt.plot(X1, Y1, 'ko', markersize=1, label = test1)
    plt.plot(X2, Y2, 'ro', markersize=1, label = test2)
   
    Y_fit1 = Y_fit[H == 0]
    Y_fit2 = Y_fit[H == 1]
    
    plt.plot(X_full, Y1_full, 'k', label = 'fit '+test1)    
    plt.plot(X_full, Y2_full, 'r', label = 'fit '+test2)    
    
    
     
    plt.title(test1 + ' vs. ' + test2)
    plt.xlabel('time [h]')
    plt.ylabel('measurements')
    plt.legend()
    
    if save_to:
        plt.savefig(save_to+'.png')
        plt.savefig(save_to+'.pdf')
        plt.close()
    else:
        plt.show()
    
    if plot_residuals:
        
        resid = Y-Y_fit
        fig = sm.qqplot(resid)
        plt.title(test1 + ' vs. ' + test2)
        save_to_resid = save_to + '_resid' 
        if save_to:
            plt.savefig(save_to_resid)
            plt.close()
        else:
            plt.show()
    
    
    return statistics, p_dict
    
   


