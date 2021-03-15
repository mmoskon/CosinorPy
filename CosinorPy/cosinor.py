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
from scipy.stats import percentileofscore

import copy
import itertools
from matplotlib.lines import Line2D
from random import sample

import os

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
    
def periodogram_df(df, folder = '', **kwargs):
    names = list(df.test.unique())
    names.sort()  

    for name in names:
        x, y = np.array(df[df.test == name].x), np.array(df[df.test == name].y)
        if folder:
            save_to = os.path.join(folder, "per_" + name)            
        else:
            save_to = ""

        periodogram(x,y, save_to = save_to, name=name, **kwargs)


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
    elif per_type == 'lombscargle':
        # Lomb-Scargle
        min_per = 2
        #max_per = 50
        
        f = np.linspace(1/max_per, 1/min_per, 10)
        Pxx_den = signal.lombscargle(X, Y, f)
    else:
        print("Invalid option")
        return
        

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
    
    
    try:
        if logscale:
            plt.semilogx(per, Pxx, 'ko')
            plt.semilogx(per, Pxx, 'k--', linewidth=0.5)
            plt.semilogx([min(per), max(per)], [T, T], 'k--', linewidth=1)
        else:
            plt.plot(per, Pxx, 'ko')
            plt.plot(per, Pxx, 'k--', linewidth=0.5)
            plt.plot([min(per), max(per)], [T, T], 'k--', linewidth=1)
    except:
        print("Could not plot!")
        return

    peak_label = ''

    if prominent:    
        locs, heights = signal.find_peaks(Pxx, height = T)
        
        if any(locs):        
            heights = heights['peak_heights']
            s = list(zip(heights, locs))
            s.sort(reverse=True)
            heights, locs = zip(*s)
            
            heights = np.array(heights)
            locs = np.array(locs)
               
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
    df_best = pd.DataFrame(columns = df_results.columns, dtype=float)
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
            plt.savefig(os.path.join(folder, prefix+test+'.png'))
            plt.savefig(os.path.join(folder, prefix+test+'.pdf'))
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
            plt.savefig(os.path.join(folder,prefix+test1+'_'+test2+'.png'))
            plt.savefig(os.path.join(folder,prefix+test1+'_'+test2+'.pdf'))

            plt.close()
        else:
            plt.show()


#def fit_group(df, n_components = 2, period = 24, folder = '', prefix='', lin_comp = False, names = [], plot_measurements = True, plot = True, x_label = "", y_label = ""):
def fit_group(df, n_components = 2, period = 24, names = "", folder = '', prefix='', **kwargs):
    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'RSS', 'R2', 'R2_adj', 'log-likelihood', 'period(est)', 'amplitude', 'acrophase', 'mesor', 'peaks', 'heights', 'troughs', 'heights2'], dtype=float)

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    if not any(names):
        names = np.unique(df.test) 

    for test in names:
        for n_comps in n_components:
            for per in period:            
                if n_comps == 0:
                    per = 100000
                X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)        
                if folder:                    
                    save_to = os.path.join(folder,prefix+test+'_compnts='+str(n_comps) +'_per=' + str(per))
                else:
                    save_to = ''
                
                #results, statistics, rhythm_param, X_full, Y_full = fit_me(X, Y, n_components = n_comps, period = per, lin_comp = lin_comp, model_type = 'lin', name = test, save_to = save_to, plot_measurements = plot_measurements, plot=plot, x_label = x_label, y_label = y_label)
                results, statistics, rhythm_param, _, _ = fit_me(X, Y, n_components = n_comps, period = per, name = test, save_to = save_to, **kwargs)
            
                try:
                    R2, R2_adj = results.rsquared,results.rsquared_adj
                except:
                    R2, R2_adj = np.nan, np.nan

                df_results = df_results.append({'test': test, 
                                            'period': per,
                                            'n_components': n_comps,
                                            'p': statistics['p'], 
                                            'p_reject': statistics['p_reject'],
                                            'RSS': statistics['RSS'],
                                            'R2': R2, 
                                            'R2_adj': R2_adj,
                                            'ME': statistics['ME'],
                                            'resid_SE': statistics['resid_SE'],
                                            'log-likelihood': results.llf,        
                                            'period(est)': rhythm_param['period'],
                                            'amplitude': rhythm_param['amplitude'],
                                            'acrophase': rhythm_param['acrophase'],
                                            'mesor': rhythm_param['mesor'],
                                            'peaks': rhythm_param['peaks'],
                                            'heights': rhythm_param['heights'],
                                            'troughs': rhythm_param['troughs'],
                                            'heights2': rhythm_param['heights2']
                                            
                                            }, ignore_index=True)
                if n_comps == 0:
                    break
    
    df_results.q = multi.multipletests(df_results.p, method = 'fdr_bh')[1]
    df_results.q_reject = multi.multipletests(df_results.p_reject, method = 'fdr_bh')[1]
    
    
    return df_results

#def population_fit_group(df, n_components = 2, period = 24, lin_comp = False, model_type="lin", alpha = 0, names = [], folder = '', prefix='', plot_measurements = True):
def population_fit_group(df, n_components = 2, period = 24, folder = '', prefix='', names = [], **kwargs):

    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'RSS', 'period(est)', 'amplitude', 'acrophase', 'mesor'], dtype=float)

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    if not any(names):
        names = np.unique(df.test) 

    names = list(map(lambda x:x.split('_rep')[0], names))
    
    for name in set(names):
        for n_comps in n_components:
            for per in period:            
                if n_comps == 0:
                    per = 100000
                    
                    
                df_pop = df[df.test.str.startswith(name)]   

                if folder:                    
                    save_to=os.path.join(folder,prefix+name+'_compnts='+str(n_comps) +'_per=' + str(per))
                    #_, statistics, _, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, save_to = save_to, lin_comp= lin_comp, model_type = model_type, alpha = alpha)
                    _, statistics, _, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, save_to = save_to, **kwargs)
                else:
                    #_, statistics, _, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, lin_comp= lin_comp, model_type = model_type, alpha = alpha, plot_on = True)
                    _, statistics, _, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, **kwargs)
                    
                            
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
    df_best = pd.DataFrame(columns = df_models.columns, dtype=float)
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

    if n_components == 0:
        X_fit = X       
        lin_comp = True
    else:
        for i in np.arange(n_components):
            n = i+1

            A = np.sin((X/(period/n))*np.pi*2)
            B = np.cos((X/(period/n))*np.pi*2)                        

            if not i:
                X_fit = np.column_stack((A, B))            
            else:
                X_fit = np.column_stack((X_fit, np.column_stack((A, B))))                
    if lin_comp and n_components:
        X_fit = np.column_stack((X, X_fit))
    
    X_fit = sm.add_constant(X_fit, has_constant='add')
    
    return X_fit
    
    
#def population_fit(df_pop, n_components = 2, period = 24, lin_comp= False, model_type = 'lin', alpha=0, plot_on = True, plot_measurements=True, plot_individuals=True, plot_margins=True, save_to = '', x_label='', y_label=''):
def population_fit(df_pop, n_components = 2, period = 24, lin_comp= False, model_type = 'lin', plot_on = True, plot_measurements=True, plot_individuals=True, plot_margins=True, save_to = '', x_label='', y_label='', **kwargs):
 
    params = -1

    tests = df_pop.test.unique()
    k = len(tests)
    
    X_test = np.linspace(0, 100, 1000)
    X_fit_eval_params = generate_independents(X_test, n_components = n_components, period = period, lin_comp = lin_comp)
    if lin_comp:
        X_fit_eval_params[:,1] = 0    
    
    min_X = 1000
    max_X = 0
    min_Y = 1000
    max_Y = 0
    min_Y_test = 1000
    max_Y_test = 0
    min_X_test = np.min(X_test)
    
    
    for test in tests:
        x,y = np.array(df_pop[df_pop.test == test].x), np.array(df_pop[df_pop.test == test].y)
        
        min_X = min(min_X, np.min(x))
        max_X = max(max_X, np.max(x))
        
        min_Y = min(min_Y, np.min(y))
        max_Y = max(max_Y, np.max(y))
        
        
        #results, statistics, rhythm_params, X_test, Y_test, model = fit_me(x, y, n_components = n_components, period = period, lin_comp=lin_comp, model_type = model_type, alpha=alpha, plot = False, return_model = True)
        results, statistics, rhythm_params, X_test, _, model = fit_me(x, y, n_components = n_components, period = period, plot = False, return_model = True, **kwargs)
        if type(params) == int:
            params = results.params
        else:
            params = np.vstack([params, results.params])
        if plot_on and plot_individuals:
            Y_eval_params = results.predict(X_fit_eval_params)
            plt.plot(X_test,Y_eval_params,'k', label=test)
            
            min_Y_test = min(min_Y_test, np.min(Y_eval_params))
            max_Y_test = max(max_Y_test, np.max(Y_eval_params))
            
            #plt.plot(x, results.fittedvalues,'k', label=test)
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
    
    
    Y_eval_params = results.predict(X_fit_eval_params)    
    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
        
    if plot_on:
        #plt.plot(x,Y_fit,'r', label="population fit")
        plt.plot(X_test,Y_eval_params,'r', label="population fit")
        plt.legend()
        if x_label:
            plt.xlabel(x_label)    
        else:
            plt.xlabel('time [h]')

        if y_label:
            plt.ylabel(y_label)    
        else:
            plt.ylabel('measurements')
    
        min_Y_test = min(min_Y_test, np.min(Y_eval_params))
        max_Y_test = max(max_Y_test, np.max(Y_eval_params))
    
        

    if plot_on and plot_margins and model_type=='lin':
        _, lower, upper = wls_prediction_std(results, exog=X_fit_eval_params, alpha=0.05)
        plt.fill_between(X_test, lower, upper, color='#888888', alpha=0.1)                   
    
    if plot_measurements:
        if model_type == 'lin':
            plt.axis([min(min_X,0), 1.1*max(max_X,period), 0.9*min(min_Y, min_Y_test), 1.1*max(max_Y, max_Y_test)])
        else:
            plt.axis([min(min_X,0), max_X, 0.9*min(min_Y, min_Y_test), 1.1*max(max_Y, max_Y_test)])
        
    else:
        plt.axis([min_X_test, 50, min_Y_test*0.9, max_Y_test*1.1])
    
    
    if plot_on:
        pop_name = "_".join(test.split("_")[:-1])
        plt.title(pop_name + ', p-value=' + "{0:.5f}".format(statistics['p']))

        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()
    

   

    statistics = calculate_statistics(x, y, Y_fit, n_components, period, lin_comp) 
    statistics_params = {'values': means,
                        'SE': se,
                        'CI': (lower_CI, upper_CI),
                        'p-values': p_values} 


    
    
    return params, statistics, statistics_params, rhythm_params, results


"""
generate random permutations of two populations
"""
def generate_permutations(n_pop1, n_pop2, N):
    n = n_pop1 + n_pop2
    permutations = []
    
    for _ in range(N):
        R = np.random.permutation(n)
        permutations.append((R[:n_pop1], R[n_pop1:]))

    return permutations

"""
generate all possible permutations of two populations. Presumption: populations should be of equal sizes.
"""
def generate_permutations_all(pop1, pop2):
    n_pop1 = len(pop1)
    #n_pop2 = len(pop2)

    permutations = set()

    full = set(pop1 + pop2)

    for i in range(1,n_pop1):
        p1 = itertools.combinations(pop1,i)
        p2 = itertools.combinations(pop2,n_pop1-i)

        #print(list(p1))
        #print(list(p2))

        X = list(itertools.product(p1,p2))

        # flatten
        for i in range(len(X)):
            X[i] = [a for b in X[i] for a in b]

        for x in X:
            x.sort()
            y = list(set(full)-set(x))
            y.sort()
            z = [tuple(x), tuple(y)]
            z.sort()
            permutations.add(tuple(z))


    return(permutations)

"""
Permutation test - does not work as well as it should. The problem: both populations in a permutation need to be rhythmic to assess the differences.
"""
def permutation_test_population(df, pairs, period = 24, n_components = 2, lin_comp = False):#, N=10=, permutations=[]):
    
    
    df_results = pd.DataFrame(columns = ['pair', "d_amp", "p_d_amp", "d_acr", "p_d_acr", "d_mesor", "p_d_mesor"], dtype=float)

    for pair in pairs:

        

        df_pop1 = df[df.test.str.startswith(pair[0])] 
        df_pop2 = df[df.test.str.startswith(pair[1])] 

        _, statistics1, _, rhythm_params1, _ = population_fit(df_pop1, n_components = n_components, period = period, lin_comp= lin_comp, model_type = 'lin', plot_on = False, plot_measurements=False, plot_individuals=False, plot_margins=False)
        _, statistics2, _, rhythm_params2, _ = population_fit(df_pop2, n_components = n_components, period = period, lin_comp= lin_comp, model_type = 'lin', plot_on = False, plot_measurements=False, plot_individuals=False, plot_margins=False)

        p1, amplitude1, acrophase1, mesor1 = statistics1['p'], rhythm_params1['amplitude'], rhythm_params1['acrophase'], rhythm_params1['mesor']
        p2, amplitude2, acrophase2, mesor2 = statistics2['p'], rhythm_params2['amplitude'], rhythm_params2['acrophase'], rhythm_params2['mesor']

        if p1 > 0.05 or p2 > 0.05:
            print(pair, "rhythmicity in one is not significant")
            continue

        d_amp = abs(amplitude1 - amplitude2)
        d_acr = abs(acrophase1 - acrophase2)
        d_mesor = abs(mesor1 - mesor2)
        amps, acrs, mesors = [d_amp], [d_acr], [d_mesor]

        tests1 = list(df_pop1.test.unique())
        tests2 = list(df_pop2.test.unique())
        #n_pop1 = len(tests1)
        #n_pop2 = len(tests2)

        #tests = np.array(tests1 + tests2)

        """
        if not permutations:            
            permutations = []
            permutations_idxs = generate_permutations(n_pop1, n_pop2, N)
            for perm1, perm2 in permutations_idxs:
                permutations.append((tests[perm1], tests[perm2]))
        """
        permutations = generate_permutations_all(tests1, tests2)

        #print(permutations)

        for perm1, perm2 in permutations:
            df_test1 = df[df.test.isin(perm1)]
            df_test2 = df[df.test.isin(perm2)]

            _, statistics_test1, _, rhythm_params_test1, _ = population_fit(df_test1, n_components = n_components, period = period, lin_comp = lin_comp, model_type = 'lin', plot_on = False, plot_measurements=False, plot_individuals=False, plot_margins=False)
            _, statistics_test2, _, rhythm_params_test2, _ = population_fit(df_test2, n_components = n_components, period = period, lin_comp = lin_comp, model_type = 'lin', plot_on = False, plot_measurements=False, plot_individuals=False, plot_margins=False)

            p_test1, amplitude_test1, acrophase_test1, mesor_test1 = statistics_test1['p'], rhythm_params_test1['amplitude'], rhythm_params_test1['acrophase'], rhythm_params_test1['mesor']
            p_test2, amplitude_test2, acrophase_test2, mesor_test2 = statistics_test2['p'], rhythm_params_test2['amplitude'], rhythm_params_test2['acrophase'], rhythm_params_test2['mesor']

            if p_test1 <= 0.05 and p_test2 <= 0.05:
                d_amp_test = abs(amplitude_test1 - amplitude_test2)
                d_acr_test = abs(acrophase_test1 - acrophase_test2)
                d_mesor_test = abs(mesor_test1 - mesor_test2)
            else:
                d_amp_test, d_acr_test, d_mesor_test = 0, 0, 0

           

            amps.append(d_amp_test)
            acrs.append(d_acr_test)
            mesors.append(d_mesor_test)
        

        #print(amps)
        #print(acrs)
        #print(mesors)

       
        
        p_d_amp = 1 - percentileofscore(amps, d_amp, 'rank')/100
        p_d_acr = 1 - percentileofscore(acrs, d_acr, 'rank')/100
        p_d_mesor = 1 - percentileofscore(mesors, d_mesor, 'rank')/100

        #print(p_d_amp, p_d_acr, p_d_mesor)

        d = {"pair": tuple(pair),
             "d_amp": d_amp, 
             "p_d_amp": p_d_amp, 
             "d_acr": d_acr, 
             "p_d_acr": p_d_acr, 
             "d_mesor": d_mesor, 
             "p_d_mesor": p_d_mesor}
        
        df_results = df_results.append(d, ignore_index=True)


    return df_results

                      
                                           



def fit_me(X, Y, n_components = 2, period = 24, lin_comp = False, model_type = 'lin', alpha = 0, name = '', save_to = '', plot=True, plot_residuals=False, plot_measurements=True, plot_margins=True, return_model = False, color = False, plot_phase = True, hold=False, x_label = "", y_label = ""):
    """
    ###
    # prepare the independent variables
    ###
    """
    X_test = np.linspace(0, 100, 1000)

    if n_components == 0:
        X_fit = X
        X_fit_test = X_test
        lin_comp = True
    else:
        for i in np.arange(n_components):
            n = i+1

            A = np.sin((X/(period/n))*np.pi*2)
            B = np.cos((X/(period/n))*np.pi*2)                
            A_test = np.sin((X_test/(period/n))*np.pi*2)
            B_test = np.cos((X_test/(period/n))*np.pi*2)

            if not i:
                X_fit = np.column_stack((A, B))
                X_fit_test = np.column_stack((A_test, B_test))     
            else:
                X_fit = np.column_stack((X_fit, np.column_stack((A, B))))
                X_fit_test = np.column_stack((X_fit_test, np.column_stack((A_test, B_test))))

    
    X_fit_eval_params = X_fit_test
    
    if lin_comp and n_components:
        X_fit = np.column_stack((X, X_fit))
        X_fit_eval_params = np.column_stack((np.zeros(len(X_test)), X_fit_test))
        X_fit_test = np.column_stack((X_test, X_fit_test))                              


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
    elif model_type == 'nb':
        #exposure = np.zeros(len(Y))
        #exposure[:] = np.mean(Y)
        #model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(), exposure = exposure)
        
        # https://towardsdatascience.com/negative-binomial-regression-f99031bb25b4
        # https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/#cameron
        if not alpha:
            
            train_model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
            train_results = train_model.fit()

            df_train = pd.DataFrame()
            df_train['Y'] = Y
            df_train['mu'] = train_results.mu
            df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Y'] - x['mu'])**2 - x['Y']) / x['mu'], axis=1)
            ols_expr = """AUX_OLS_DEP ~ mu - 1"""
            aux_olsr_results = smf.ols(ols_expr, df_train).fit()

            alpha=aux_olsr_results.params[0]
            #print(alpha)

        model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        
        results = model.fit()
    else:
        print("Invalid option")
        return

    
    if model_type =='lin':
        Y_fit = results.fittedvalues
    else:
        Y_fit = results.predict(X_fit)
        
    
    if model_type in ['lin', 'poisson', 'nb']:
        statistics = calculate_statistics(X, Y, Y_fit, n_components, period, lin_comp)
        if model_type in ['poisson', 'nb']:
            statistics['count'] = np.sum(Y)                                
    else:
        RSS = sum((Y - Y_fit)**2)
        p = results.llr_pvalue
        statistics = {'p':p, 'RSS':RSS, 'count': np.sum(Y)}
    
    Y_test = results.predict(X_fit_test)
    Y_eval_params = results.predict(X_fit_eval_params)
    
    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
    
    """
    ###
    # plot
    ###
    """
    if plot:
        if plot_margins:
            if model_type == 'lin':
                _, lower, upper = wls_prediction_std(results, exog=X_fit_test, alpha=0.05)
                if color:
                    plt.fill_between(X_test, lower, upper, color=color, alpha=0.1)
                else:
                    plt.fill_between(X_test, lower, upper, color='#888888', alpha=0.1)
            else:
                res2 = copy.deepcopy(results)
                params = res2.params
                CIs = results.conf_int()
                
                #N = 512
                N = 1024
                
                if n_components == 1:
                    #N2 = 8
                    N2 = 10
                elif n_components == 2:
                    #N2 = 6
                    N2 = 8
                else:                                   
                    #N2 = 8 - n_components 
                    N2 = 10 - n_components 
              
                
                P = np.zeros((len(params), N2))
                
                for i, CI in enumerate(CIs):
                    P[i,:] = np.linspace(CI[0], CI[1], N2)
                    
                amplitude_CI = [rhythm_params['amplitude']]
                mesor_CI = [rhythm_params['mesor']]
                acrophase_CI = [rhythm_params['acrophase']]
                
                param_samples = list(itertools.product(*P))
                N = min(N, len(param_samples))
                
                for i,p in enumerate(sample(param_samples, N)):
                    res2.initialize(results.model, p)            
                    Y_test_CI = res2.predict(X_fit_test)
                    
                    rhythm_params_CI = evaluate_rhythm_params(X_test, Y_test_CI)
                    amplitude_CI.append(rhythm_params_CI['amplitude'])
                    mesor_CI.append(rhythm_params_CI['mesor'])
                    acrophase_CI.append(rhythm_params_CI['acrophase'])
                   
                    
                    """
                    if i == 0:
                        Y_min = Y
                        Y_max = Y
                    else:
                        Y_min = np.min(np.vstack([Y,Y_min]), axis=0)
                        Y_max = np.max(np.vstack([Y,Y_max]), axis=0)
                    """
                    if color and color != '#000000':
                        plt.plot(X_test, Y_test_CI, color=color, alpha=0.05)
                    else:
                        plt.plot(X_test, Y_test_CI, color='#888888', alpha=0.05)
                
                    
                #plt.fill_between(X_test, Y_min, Y_max, color='#888888', alpha=0.1)
        
                #amplitude_CI = (min(amplitude_CI), max(amplitude_CI))
                #mesor_CI = (min(mesor_CI), max(mesor_CI))
                #acrophase_CI = (min(acrophase_CI), max(acrophase_CI))
        
                rhythm_params['amplitude_CI'] = amplitude_CI
                rhythm_params['mesor_CI'] = mesor_CI
                rhythm_params['acrophase_CI'] = acrophase_CI
        
        
        ###
        if not color:
            color = 'black'

        if plot_measurements:        
            if not hold:             
                plt.plot(X,Y, 'ko', markersize=1, label = 'data', color=color)
            else:
                plt.plot(X,Y, 'ko', markersize=1, color=color)
        #plt.plot(X, results.fittedvalues, label = 'fit')
        
        if not hold:
            plt.plot(X_test, Y_test, 'k', label = 'fit', color=color)
        else:
            plt.plot(X_test, Y_test, 'k', label = name, color=color)
        #if color and not plot_margins: 
        #    plt.plot(X_test, Y_test, 'k', label = 'fit', color=color)
        #else:
        #    plt.plot(X_test, Y_test, 'k', label = 'fit')
        
        if plot_measurements:
            X = X % period

            if model_type == 'lin':               
                #plt.axis([min(min(X),0), 1.1*max(max(X),period), 0.9*min(min(Y), min(Y_test)), 1.1*max(max(Y), max(Y_test))])
                plt.axis([min(min(X),0), max(X), 0.9*min(min(Y), min(Y_test)), 1.1*max(max(Y), max(Y_test))])
            else:
                plt.axis([min(min(X),0), max(X), 0.9*min(min(Y), min(Y_test)), 1.1*max(max(Y), max(Y_test))])
        else:
            plt.axis([min(X_test), period, min(Y_test)*0.9, max(Y_test)*1.1])
        #plt.title(name + ', components=' + str(n_components) +' , period=' + str(period) + '\np-value=' + str(statistics['p']) + ', p-value(gof)=' + str(statistics['p_reject']))
        #plt.title(name + ', components=' + str(n_components) +' , period=' + str(period) + '\np-value=' + str(statistics['p']))
        if model_type == 'lin':
            if name: 
                plt.title(name + ', p-value=' + "{0:.5f}".format(statistics['p']))
            else:
                plt.title('p-value=' + "{0:.5f}".format(statistics['p']))
        else:
            if name:
                plt.title(name + ', p-value=' + '{0:.3f}'.format(statistics['p']) + ' (n='+str(statistics['count'])+ ')')            
            else:
                plt.title('p-value=' + '{0:.3f}'.format(statistics['p']) + ' (n='+str(statistics['count'])+ ')')
        if x_label:
            plt.xlabel(x_label)
        else:
            plt.xlabel('Time [h]')
        
        if y_label:
            plt.ylabel(y_label)
        elif model_type == 'lin':
            plt.ylabel('Measurements')
        else:
            plt.ylabel('Count')
        #fig = plt.gcf()
        #fig.set_size_inches(11,8)               
        

        
        if not hold:
            if save_to:
                plt.savefig(save_to+'.png')
                plt.savefig(save_to+'.pdf')
                plt.close()
            else:
                plt.show()
            if plot_residuals:
                resid = results.resid
                sm.qqplot(resid)
                plt.title(name)
                if save_to:
                    plt.savefig(save_to+'_resid.pdf', bbox_inches='tight')
                    plt.savefig(save_to+'_resid.png')                
                    plt.close()
                else:
                    plt.show()
            
            if plot_phase:
                per = rhythm_params['period']
                amp = rhythm_params['amplitude']
                phase = rhythm_params['acrophase']
                if save_to:
                    folder = os.path.join(*os.path.split(save_to)[:-1])                                        
                    plot_phases([phase], [amp], [name], period=per, folder=folder)
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

def plot_phases(acrs, amps, tests, period=24, colors = ("black", "red", "green", "blue"), folder = "", prefix="", legend=True, CI_acrs = [], CI_amps = [], linestyles = [], title = "", labels = []):

    acrs = np.array(acrs, dtype = float)
    amps = np.array(amps, dtype = float)
    
    if colors and len(colors) < len(tests):
        colors += ("black",) * (len(tests)-len(colors))

    x = np.arange(0, 2*np.pi, np.pi/4)
    x_labels = list(map(lambda i: 'CT ' + str(i) + " ", list((x/(2*np.pi) * period).astype(int))))
    x_labels[1::2] = [""]*len(x_labels[1::2])

    ampM = np.max(amps)
    amps /= ampM
    
    acrs = -acrs
    
    ax = plt.subplot(111, projection='polar')        
    ax.set_theta_offset(0.5*np.pi)
    ax.set_theta_direction(-1) 
    lines = []

    for i, (acr, amp, _, color) in enumerate(zip(acrs, amps, tests, colors)):
        
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
        if linestyles:
            #ax.plot([acr, acr], [0, amp], label=test, color=color, linestyle = linestyles[i])
            ax.annotate("", xy=(acr, amp), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, alpha = 0.75, linewidth=2, linestyle = linestyles[i]) )
            lines.append(Line2D([0], [0], color=color, linewidth=2, linestyle=linestyles[i]))
        else:
            #ax.plot([acr, acr], [0, amp], label=test, color=color)
            ax.annotate("", xy=(acr, amp), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, alpha = 0.75, linewidth=2) )
            lines.append(Line2D([0], [0], color=color, linewidth=2))

        #ax.plot([acr, acr], [0, amp], label=test, color=color)
    
        #ax.annotate("", xy=(acr, amp), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, linewidth=2) )
        
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
    #ax.set_title(name, va='bottom')
    if title:
        ax.set_title(title, va='bottom')
    else:
        ax.set_title(name, va='bottom')

    if legend:
        if labels:
            plt.legend(lines, labels, bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0., frameon=False)
        else:
            plt.legend(lines, tests, bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0., frameon=False)
        #ax.legend()
    if folder:
        plt.savefig(os.path.join(folder,prefix+name+"_phase.pdf"))
        plt.savefig(os.path.join(folder,prefix+name+"_phase.png"))
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


    # peaks and heights
    Y = Y[X < 24]
    X = X[X < 24]
    locs, heights = signal.find_peaks(Y, height = MESOR)
    heights = heights['peak_heights'] 

    peaks = X[locs]
    heights = Y[locs]

    Y2 = M - Y    
    locs2, heights2 = signal.find_peaks(Y2, height = MESOR-m)
    heights2 = heights2['peak_heights'] 

    troughs = X[locs2]
    heights2 = Y[locs2]


    return {'period':PERIOD, 'amplitude':AMPLITUDE, 'acrophase':ACROPHASE, 'mesor':MESOR, 'peaks': peaks, 'heights': heights, 'troughs': troughs, 'heights2': heights2}
    
def calculate_statistics(X, Y, Y_fit, n_components, period, lin_comp = False):
    # statistics according to Cornelissen (eqs (8) - (9))
    MSS = sum((Y_fit - Y.mean())**2)
    RSS = sum((Y - Y_fit)**2)

    n_params = n_components * 2 + 1
    if lin_comp:
        n_params += 1            
    N = Y.size

    F = (MSS/(n_params - 1)) / (RSS/(N - n_params)) 
    p = 1 - stats.f.cdf(F, n_params - 1, N - n_params)
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
        try:
            F = (SSLOF/(n_T-1-(2*n_components + 1)))/(SSPE/(N-n_T))
            p_reject = 1 - stats.f.cdf(F, n_T-1-(2*n_components + 1), N-n_T)
        except:
            F = np.nan
            p_reject = np.nan    
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
    p = 1 - stats.f.cdf(F, n_params - 1, N - n_params)
    
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
#def compare_pairs(df, pairs, n_components = 3, period = 24, lin_comp = False, model_type = 'lin', alpha=0, folder = '', prefix = '', plot_measurements=True):
def compare_pairs(df, pairs, n_components = 3, period = 24, folder = "", prefix = "", **kwargs):
    
    df_results = pd.DataFrame()

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
                
    for test1, test2 in pairs: 
        for per in period:
            for n_comps in n_components:                                
                if folder:                                       
                    save_to = os.path.join(folder,prefix + test1 + '-' + test2 + '_per=' + str(per) + '_comps=' + str(n_comps))
                else:
                    save_to = ''
                
                #pvalues, params, results = compare_pair_df_extended(df, test1, test2, n_components = n_comps, period = per, lin_comp = lin_comp, model_type = model_type, alpha=alpha, save_to = save_to, plot_measurements=plot_measurements)
                pvalues, params, _ = compare_pair_df_extended(df, test1, test2, n_components = n_comps, period = per, save_to = save_to, **kwargs)
                
                d = {}
                d['test'] = test1 + ' vs. ' + test2
                d['period'] = per
                d['n_components'] = n_comps
                for i, (param, p) in enumerate(zip(params, pvalues)):
                    d['param' + str(i+1)] = param
                    d['p' + str(i+1)] = p
                
                d['p(F test)'] = pvalues[-1]
                
                df_results = df_results.append(d, ignore_index=True)
  
    
    
    for i, (param, p) in enumerate(zip(params, pvalues)):        
        df_results['q'+str(i+1)] = multi.multipletests(df_results['p'+str(i+1)], method = 'fdr_bh')[1]
    
    df_results['q(F test)'] = multi.multipletests(df_results['p(F test)'], method = 'fdr_bh')[1]


    columns = df_results.columns
    columns = columns.sort_values()
    columns = np.delete(columns, np.where(columns == 'period'))
    columns = np.append(['period'], columns)
    columns = np.append([columns[-1]], columns[:-1])
    
    df_results = df_results.reindex(columns, axis=1)
    
    return df_results

    #return multi.multipletests(P, method = 'fdr_bh')[1]

#def compare_pairs_best_models(df, df_best_models, pairs, lin_comp = False, model_type = 'lin', alpha = 0, folder = '', prefix = '', plot_measurements=True):
def compare_pairs_best_models(df, df_best_models, pairs, folder = "", prefix = "", **kwargs):
    df_results = pd.DataFrame()

    
    for test1, test2 in pairs:
        model1 = df_best_models[df_best_models["test"] == test1].iloc[0]
        model2 = df_best_models[df_best_models["test"] == test2].iloc[0]
    
        n_components1 = model1.n_components
        n_components2 = model2.n_components
    
        period1 = model1.period
        period2 = model2.period


        if n_components1 > n_components2:
            test1, test2 = test2, test1
            n_components1, n_components2 = n_components2, n_components1
            period1, period2 = period2, period1
        
        if folder:            
            save_to = os.path.join(folder, prefix + test1 + '-' + test2 + '_per1=' + str(period1) + '_comps1=' + str(n_components1) + '_per1=' + str(period2) + '_comps1=' + str(n_components2))
        else:
            save_to = ''
        
        #pvalues, params, results = compare_pair_df_extended(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2 = period2, lin_comp = lin_comp, model_type = model_type, alpha=alpha, save_to = save_to, plot_measurements=plot_measurements)
        pvalues, params, _ = compare_pair_df_extended(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2 = period2, save_to = save_to, **kwargs)
        
        d = {}
        d['test'] = test1 + ' vs. ' + test2
        d['period1'] = period1
        d['n_components1'] = n_components1
        d['period2'] = period2
        d['n_components2'] = n_components2
        for i, (param, p) in enumerate(zip(params, pvalues)):
            d['param' + str(i+1)] = param
            d['p' + str(i+1)] = p
        
        d['p(F test)'] = pvalues[-1]
        
        df_results = df_results.append(d, ignore_index=True)
    
        
    
    for i, (param, p) in enumerate(zip(params, pvalues)):        
        df_results['q'+str(i+1)] = multi.multipletests(df_results['p'+str(i+1)], method = 'fdr_bh')[1]
    
    df_results['q(F test)'] = multi.multipletests(df_results['p(F test)'], method = 'fdr_bh')[1]


    columns = df_results.columns
    columns = columns.sort_values()
    #columns = np.delete(columns, np.where(columns == 'period'))
    #columns = np.append(['period'], columns)
    #columns = np.append([columns[-1]], columns[:-1])
    
    df_results = df_results.reindex(columns, axis=1)
    
    return df_results

    #return multi.multipletests(P, method = 'fdr_bh')[1]

def compare_pair_df_extended(df, test1, test2, n_components = 3, period = 24, n_components2 = None, period2 = None, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot_measurements=True, plot_residuals=False, plot_margins=True, x_label = '', y_label = ''):
       
    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1
        
    
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
    X_i = H_i * X 

    for i in np.arange(n_components1):
        n = i+1

        A = np.sin((X/(period1/n))*np.pi*2)        
        B = np.cos((X/(period1/n))*np.pi*2) 
        if not i:
            X_fit = np.column_stack((A, B))        
        else:
            X_fit = np.column_stack((X_fit, np.column_stack((A, B))))
        
    if non_rhythmic:
        X_fit = np.column_stack((X_fit, H_i))
        idx_params = [-1]
    else:
        for i in np.arange(n_components2):
            n = i+1

            A_i = H_i * np.sin((X/(period2/n))*np.pi*2)        
            B_i = H_i * np.cos((X/(period2/n))*np.pi*2) 
        
               
            X_fit = np.column_stack((X_fit, np.column_stack((A_i, B_i))))
        
        X_fit = np.column_stack((X_fit, H_i))
        
        # idx_params = [3, 4] # n = 1
        # idx_params = [5, 6, 7, 8] # n = 2
        # idx_params = [7, 8, 9, 10, 11, 12] # n = 3
        # idx_params = [9, 10, 11, 12, 13, 14, 15, 16] # n = 4
        
        #strt = 2*n_components + 1
        #stp = strt + 2*n_components + 1

        strt = -2
        stp = strt - 2*n_components2 - 1
        idx_params = np.arange(strt, stp, -1)


           
        
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
    elif model_type == 'nb':
        if not alpha:
            train_model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
            train_results = train_model.fit()

            df_train = pd.DataFrame()
            df_train['Y'] = Y
            df_train['mu'] = train_results.mu
            df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Y'] - x['mu'])**2 - x['Y']) / x['mu'], axis=1)
            ols_expr = """AUX_OLS_DEP ~ mu - 1"""
            aux_olsr_results = smf.ols(ols_expr, df_train).fit()

            alpha=aux_olsr_results.params[0]
            #print(alpha)

        model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        
        results = model.fit()
    else:
        print("Invalid option")
        return    
   
    """
    ###
    # plot
    ###
    """
    
    
    ###
    if plot_measurements:
        plt.plot(df_pair[df_pair.test == test1].x, df_pair[df_pair.test == test1].y, 'ko', markersize=1, alpha = 0.75)
        plt.plot(df_pair[df_pair.test == test2].x, df_pair[df_pair.test == test2].y, 'ro', markersize=1, alpha = 0.75)
    #plt.plot(X, results.fittedvalues, label = 'fit')
    
    if model_type =='lin':
        Y_fit = results.fittedvalues        
    else:
        Y_fit = results.predict(X_fit)
        
    
    
    
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

    ### F-test
    # for nested models
    # using extra-sum-of-squares F test
    # in a similar way as described in CYCLOPS
    # https://www.pnas.org/content/114/20/5312#sec-8
    # https://www.pnas.org/content/pnas/suppl/2017/04/20/1619320114.DCSupplemental/pnas.201619320SI.pdf?targetid=nameddest%3DSTXT

    n_params_full = len(results.params)
    n_params_small = n_params_full - len(idx_params) 
    N = len(Y)

    r_small = fit_me(X, Y, n_components, period, lin_comp=lin_comp, model_type=model_type, alpha=alpha, plot=False, x_label = x_label, y_label = y_label)
    RSS_small = r_small[1]['RSS']
    RSS_full = sum((Y - Y_fit)**2)

    DoF_small = N-n_params_small
    DoF_full = N-n_params_full

    """
    print('RSS_small: ', RSS_small)
    print('RSS_full: ', RSS_full)
    print('n_small, dof: ', n_params_small, DoF_small)
    print('n_full, dof: ', n_params_full, DoF_full)
    """
    p_f = compare_models(RSS_small, RSS_full, DoF_small, DoF_full)

    
    
    ### plot with higher density
    
    n_points = 1000
    X_full = np.linspace(min(min(X1),min(X2)), max(max(X1), max(X2)), n_points)
    
    X_fit_full = generate_independents_compare(X_full, X_full, n_components1 = n_components1, period1 = period1, n_components2 = n_components2, period2 = period2, lin_comp= lin_comp)
    
    H_i = X_fit_full[:,-1]
    locs = H_i == 0

    #Y_fit_full = results.predict(X_fit_full)
    #plt.plot(X_full, Y_fit_full[0:n_points], 'k', label = test1)    
    #plt.plot(X_full, Y_fit_full[n_points:], 'r', label = test2)    
    
    Y_fit_full1 = results.predict(X_fit_full[locs])
    Y_fit_full2 = results.predict(X_fit_full[~locs])

    plt.plot(X_full, Y_fit_full1, 'k', label = test1)    
    plt.plot(X_full, Y_fit_full2, 'r', label = test2)    
    
    if model_type == 'lin' and plot_margins:
        _, lower, upper = wls_prediction_std(results, exog=X_fit_full[locs], alpha=0.05)
        plt.fill_between(X_full, lower, upper, color='black', alpha=0.1)   
        _, lower, upper = wls_prediction_std(results, exog=X_fit_full[~locs], alpha=0.05)
        plt.fill_between(X_full, lower, upper, color='red', alpha=0.1)

    
    ### end of plot with higher density
    
    
    #p = min(results.pvalues[idx_params])
    #plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p))
    plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p_f))
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
        sm.qqplot(resid)
        plt.title(test1 + ' vs. ' + test2)
        save_to_resid = save_to.split(".")[0] + '_resid' + save_to.split(".")[1]
        if save_to:
            plt.savefig(save_to_resid)
            plt.close()
        else:
            plt.show()
    

    p_values = list(results.pvalues[idx_params]) + [p_f]
    
    return (p_values, results.params[idx_params], results)

def generate_independents_compare(X1, X2, n_components1 = 3, period1 = 24, n_components2 = 3, period2 = 24, lin_comp = False, non_rhythmic=False):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    
    X = np.concatenate((X1, X2))
    H_i = np.concatenate((H1, H2))
    X_i = H_i * X
   

    for i in np.arange(n_components1):
        n = i+1

        A = np.sin((X/(period1/n))*np.pi*2)        
        B = np.cos((X/(period1/n))*np.pi*2) 
        if not i:
            X_fit = np.column_stack((A, B))        
        else:
            X_fit = np.column_stack((X_fit, np.column_stack((A, B))))
        
    if non_rhythmic:
        X_fit = np.column_stack((X_fit, H_i))                
    else:
        for i in np.arange(n_components2):
            n = i+1

            A_i = H_i * np.sin((X/(period2/n))*np.pi*2)        
            B_i = H_i * np.cos((X/(period2/n))*np.pi*2) 
        
               
            X_fit = np.column_stack((X_fit, np.column_stack((A_i, B_i))))
        
        X_fit = np.column_stack((X_fit, H_i))
        
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))    

    X_fit = sm.add_constant(X_fit, has_constant='add')

    return X_fit



# compare two models according to the F-test
# http://people.reed.edu/~jones/Courses/P24.pdf
# https://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?reg_howtheftestworks.htm  
def get_best_models(df, df_models, n_components = [1,2,3], lin_comp = False, criterium='p', reverse = True):
       
    names = np.unique(df_models.test)   
    df_best = pd.DataFrame(columns = df_models.columns, dtype=float)
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

def plot_df_models(df, df_models, folder ="", **kwargs):
    for row in df_models.iterrows():
        test = row[1].test
        n_components = row[1].n_components
        period = row[1].period
        X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)   
        
        if folder:            
            fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = os.path.join(folder,test+'_compnts='+str(n_components) +'_per=' + str(period)), **kwargs)
        else:
            fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = "", **kwargs)


def plot_tuples_best_models(df, df_best_models, tuples, colors = ['black', 'red'], folder = '', **kwargs):
    
    
    for T in tuples:
        min_x = 1000
        max_x = -1000
        min_y = 1000
        max_y = -1000


        for test, color in zip(T, colors):
            model = df_best_models[df_best_models["test"] == test].iloc[0]
            n_components = model.n_components
            period = model.period
            X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)  

            min_x = min(min(X), min_x)
            max_x = max(max(X % period), max_x)
            min_y = min(min(Y), min_y)
            max_y = max(max(Y), max_y)

            fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = "", plot_residuals = False, hold=True, color = color, **kwargs)
        
        plt.title(" + ".join(T))
        
                
        plt.axis([min(min_x,0), max_x, 0.9*min_y, 1.1*max_y])


        plt.legend()

        if folder:            
            save_to = os.path.join(folder,"+".join(T))   
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
        else:
            plt.show()
        plt.close()



def plot_df_models_population(df, df_models, folder="", model_type="lin"):
    for row in df_models.iterrows():
        pop = row[1].test
        n_components = row[1].n_components
        period = row[1].period
        #X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)  
        df_pop = df[df.test.str.startswith(pop)]
        if folder:            
            save_to = os.path.join(folder, pop+'_pop_compnts='+str(n_components) +'_per=' + str(period))
        else:
            save_to = ""
        population_fit(df_pop, n_components = n_components, period = period, model_type = model_type, save_to = save_to)

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
            save_to = os.path.join(folder, prefix + test1 + '-' + test2)
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
            save_to = os.path.join(folder, prefix + test1 + '-' + test2)
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
    for i in np.arange(len(perr)):
        T0 = popt[i]/perr[i]
        p[i] = 2 * (1 - stats.t.cdf(abs(T0), DoF))
    
    p_dict = {}
    for param, val, p_val in zip(parameters, popt, p):
        p_dict[param] = val
        p_dict["p("+param+")"] = p_val
    
    
    
    plt.plot(X1, Y1, 'ko', markersize=1, label = test1)
    plt.plot(X2, Y2, 'ro', markersize=1, label = test2)
   
    #Y_fit1 = Y_fit[H == 0]
    #Y_fit2 = Y_fit[H == 1]
    
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
        sm.qqplot(resid)
        plt.title(test1 + ' vs. ' + test2)
        save_to_resid = save_to + '_resid' 
        if save_to:
            plt.savefig(save_to_resid)
            plt.close()
        else:
            plt.show()
    
    
    return statistics, p_dict
    
# when the number of samples is large, the 0.05 significance level should be decreased
# calculate_significance_level allows you to define a significance level in such cases
# N: number of samples
# kwargs should include:
# n_params: number of params in a model
# OR
# n_components: number of components in a cosinor model 
# optional: lin_comp (bool): additional linear component
# by default the function returns a significance level for the F-test used in a regression
# if return_T is True, the function returns a significance level for the T-test
# for the explanation of background and references see https://davegiles.blogspot.com/2019/10/everythings-significant-when-you-have.html
def calculate_significance_level(N, **kwargs):
    F = np.log(N)
    
    if 'n_params' in kwargs:
        n_params = kwargs['n_params']
    elif 'n_components' in kwargs:
        n_components = kwargs['n_components']
        n_params = n_components * 2 + 1

        if 'lin_comp' in kwargs and kwargs['lin_comp']:
            n_params += 1
    else:
        print('At least n_params or n_components need to be specified.')
        return


    dof1 = n_params-1

    if 'return_T' in kwargs and kwargs['return_T']:
        alpha_T = 1 - stats.t.cdf(np.sqrt(F), dof1)
        return alpha_T
    else:
        dof2 = N - n_params
        alpha_F = 1 - stats.f.cdf(F, dof1, dof2)
        return alpha_F