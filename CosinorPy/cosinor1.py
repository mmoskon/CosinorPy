from CosinorPy import cosinor
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import ncf, f, norm
import scipy.stats as stats
import statsmodels.stats.multitest as multi
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm


def prepare_df(X1, Y1, X2, Y2):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    
    Y = np.concatenate((Y1, Y2))    
    X = np.concatenate((X1, X2))    
    H = np.concatenate((H1, H2))

    data = pd.DataFrame()
    
    data['x'] = X
    data['y'] = Y
    data['group'] = H
    
    return data

def plot_pair(data, results, test1='1', test2='2', plot_measurements=True, save_to='', plot_dense=True, period=24, plot_margins = True, legend=True):
   
    X = data.x
    Y = data.y 
    H = data.group
    
    color1 = "black"
    """
    if "LDL" in test2:
        color2 = "#FF0000"
    elif "HDL" in test2:
        color2 = "#0000FF"
    elif "CHOL" in test2:
        color2 = "#00FF00"
    else:
        color2 = "#FF0000"
    """
    color2 = "red"

    if plot_dense:
        X_fit = np.linspace(min(X), max(X), 100)
        H1_fit = np.zeros(len(X_fit))
        H2_fit = np.ones(len(X_fit))
        
        X_fit = np.append(X_fit, X_fit)
        H_fit = np.append(H1_fit, H2_fit)
        
        rrr_fit= np.cos(2*np.pi*X_fit/period)
        sss_fit = np.sin(2*np.pi*X_fit/period)
        
        data = pd.DataFrame()
    
        data['x'] = X_fit
        
        
        data['group'] = H_fit
        data['rrr'] = rrr_fit
        data['sss'] = sss_fit
        
        Y_fit = results.predict(data)       
        
        if plot_margins:
            X_fit_M = np.linspace(min(X), max(X), 100)
            rrr_fit_M= np.cos(2*np.pi*X_fit_M/period)
            sss_fit_M = np.sin(2*np.pi*X_fit_M/period)
          
            D1 = np.column_stack((H1_fit, rrr_fit_M, sss_fit_M, H1_fit*rrr_fit_M, H1_fit*sss_fit_M))
            D1 = sm.add_constant(D1, has_constant='add')
            sdev, lower, upper = wls_prediction_std(results, exog=D1, alpha=0.05)
            plt.fill_between(X_fit_M, lower, upper, color=color1, alpha=0.1)   
        
            D2 = np.column_stack((H2_fit, rrr_fit_M, sss_fit_M, H2_fit*rrr_fit_M, H2_fit*sss_fit_M))
            D2 = sm.add_constant(D2, has_constant='add')
            sdev, lower, upper = wls_prediction_std(results, exog=D2, alpha=0.05)
            plt.fill_between(X_fit_M, lower, upper, color=color2, alpha=0.1)   
        
        
    else:
        H_fit = H
        X_fit = X
        Y_fit = results.fittedvalues    
        
    
    X1 = X[H == 0]
    Y1 = Y[H == 0]
    
    X_fit1 = X_fit[H_fit == 0]
    X_fit2 = X_fit[H_fit == 1]
    Y_fit1 = Y_fit[H_fit == 0]
    
    L1 = list(zip(X_fit1,Y_fit1))
    L1.sort()
    X_fit1,Y_fit1 = list(zip(*L1))  
    
    X2 = X[H == 1]
    Y2 = Y[H == 1]
    Y_fit2 = Y_fit[H_fit == 1]
    
    L2 = list(zip(X_fit2,Y_fit2))
    L2.sort()
    X_fit2,Y_fit2 = list(zip(*L2))  
    
    if plot_measurements:
        plt.plot(X1, Y1, 'o', color=color1, markersize=1)
        plt.plot(X2, Y2, 'o', color=color2, markersize=1)
    plt.plot(X_fit1, Y_fit1, color=color1, label = test1)    
    plt.plot(X_fit2, Y_fit2, color=color2, label = test2)
    plt.xlabel('time [h]')    
    plt.ylabel('measurements')
    
    if legend:
        plt.legend()
    
    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()    
   
def plot_single(data, results, test='', plot_measurements=True, save_to='', plot_dense=True, plot_margins = True, period=24):
   
    X = data.x
    Y = data.y 
    
    #if 'control' in test.lower():
    #    color = "black"
    #else:
    #    color = "blue"
    
    color = "black"
        
    if plot_dense:
        X_fit = np.linspace(min(X), 100, 1000)
        
        rrr_fit= np.cos(2*np.pi*X_fit/period)
        sss_fit = np.sin(2*np.pi*X_fit/period)
        
        data = pd.DataFrame()
            
        #data['x'] = X_fit
        data['rrr'] = rrr_fit
        data['sss'] = sss_fit
        #data['Intercept'] = 1
        
        Y_fit = results.predict(data)      
        
        D = np.column_stack((rrr_fit, sss_fit))
        D = sm.add_constant(D, has_constant='add')

        if plot_margins:
            sdev, lower, upper = wls_prediction_std(results, exog=D, alpha=0.05)
            plt.fill_between(X_fit, lower, upper, color=color, alpha=0.1)   
            #plt.fill_between(X_fit, lower, upper, color='#888888', alpha=0.1)   

        
    else:
        X_fit = X
        Y_fit = results.fittedvalues    
        
    
    L1 = list(zip(X_fit,Y_fit))
    L1.sort()
    X_fit,Y_fit = list(zip(*L1))  
    
    if plot_measurements:
        plt.plot(X, Y, 'o', markersize=1, color=color)
    
    plt.plot(X_fit, Y_fit, color=color, label = test)    
    
    if plot_measurements:
        plt.axis([min(min(X),0), 1.1*max(max(X),period), 0.9*min(min(Y), min(Y_fit)), 1.1*max(max(Y), max(Y_fit))])
    else:
        plt.axis([min(X_fit), 50, min(Y_fit)*0.9, max(Y_fit)*1.1])
        
    plt.title(test + ', p-value=' + "{0:.5f}".format(results.f_pvalue))
    
    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()    
   
"""
def fit_cosinor_pair(X1, Y1, X2, Y2, period, test1='1', test2='2'):
   
    data = prepare_df(X1, Y1, X2, Y2)
    fit_results, amp, acr = fit_cosinor_pairs_df(data, period)
    
    plot_pair(data, fit_results, test1=test1, test2=test2, period=period)
    
    return fit_results, amp, acr
"""
"""
def fit_cosinor_pair_df(data, period):
       
    rrr = np.cos(2*np.pi*data.x/period)
    sss = np.sin(2*np.pi*data.x/period)
        
    
    data['rrr'] = rrr
    data['sss'] = sss

    results = smf.ols('y ~ group + rrr + sss + group:rrr + group:sss', data).fit()
    
    beta_s = np.array([results.params['sss'], results.params['group:sss']])
    beta_r = np.array([results.params['rrr'], results.params['group:rrr']])
    
    beta_s[1] += beta_s[0]
    beta_r[1] += beta_r[0]
    
    amp, acr = amp_acr(beta_s, beta_r)
    
    
    #cov = results.cov_params().loc[['rrr', 'group:rrr', 'sss', 'group:sss'],['rrr', 'group:rrr', 'sss', 'group:sss'],]   
    
    #jac = np.array((np.concatenate((np.diag(a_r)[0,:], np.diag(a_s)[0,:])),
    #np.concatenate((np.diag(a_r)[1,:], np.diag(a_s)[1,:])),
    #np.concatenate((np.diag(b_r)[0,:], np.diag(b_s)[0,:])),
    #np.concatenate((np.diag(b_r)[1,:], np.diag(b_s)[1,:]))))    
       
       
    return results, amp, acr
"""

"""
def fit_cosinor_df(data, period):
       
    rrr = np.cos(2*np.pi*data.x/period)
    sss = np.sin(2*np.pi*data.x/period)
        
    data['rrr'] = rrr
    data['sss'] = sss

    results = smf.ols('y ~ rrr + sss', data).fit()
    
    beta_s = results.params['sss']
    beta_r = results.params['rrr']
    
    
    amp, acr = amp_acr(beta_s, beta_r)
           
    return results, amp, acr
"""



def generate_test_data(x, beta_s, beta_r, period, plot_on = True):
    y = beta_r * np.cos(2*np.pi*x/period) + beta_s *  np.sin(2*np.pi*x/period)
    
    if plot_on:
        plt.plot(x,y)
        plt.show()
    
    #amp, acr = amp_acr(beta_s, beta_r, corrected = True)
    #print("rrr (beta)", beta_r, "sss (gamma):", beta_s)
    #print("amp", amp, "acr", acr)
    
    return x,y
    

def amp_acr(beta_s, beta_r, corrected = True):
    amp = (beta_s**2 + beta_r**2)**(1/2)
    
    #print("rrr (beta)", beta_r, "sss (gamma):", beta_s)
    
    if corrected:
        
        if type(beta_s) != np.ndarray:
            beta_s = np.array([beta_s])
            beta_r = np.array([beta_r])
        
        
        acr = np.zeros(len(beta_s))
        # acr corrected according to cosinor2
        for i in range(len(beta_s)):
            rrr = beta_r[i]
            sss = beta_s[i]
            
            if (rrr > 0) and (sss > 0):
                acr[i] = 0 + (-1*np.arctan(np.abs(sss / rrr)))
                #acr[i] = np.arctan(sss / rrr)
            elif (rrr > 0) and (sss < 0):
                acr[i] = 2*(-1)*np.pi + (1*np.arctan(np.abs(sss / rrr)))                
                #acr[i] = (1*np.arctan(sss / rrr))
            elif (rrr < 0) and (sss > 0):
                acr[i] = np.pi*(-1) + (1*np.arctan(np.abs(sss / rrr)))
                #acr[i] = np.pi*(-1) + (1*np.arctan(sss / rrr))
            else:
                acr[i] = np.pi*(-1)+(-1*np.arctan(np.abs(sss / rrr)))
                #acr[i] = np.pi + np.arctan(sss / rrr)
                        
            #acr[i] %= 2*np.pi
            #if acr[i] < 0:                           
            #    acr[i] = acr[i] + 2*np.pi
                #print(acr)
        if type(amp) != np.ndarray:
            acr = acr[0]            
        
        #acr = np.arctan2(beta_s, beta_r)
        #acr = np.arctan2(beta_r, beta_s)
        #acr = np.abs(acr)
                   
           
    else:
        acr = np.arctan(beta_s / beta_r)
        
       
    return amp, acr

def evaluate_cosinor(x, MESOR, amp, acr, period):
    return MESOR + amp*np.cos((2*np.pi*x/period) + acr)

def population_fit_group(df, period = 24, save_folder='', plot_on=True):
    df_cosinor1_fits = pd.DataFrame(columns = ['test', 'p', 'q', 'amplitude', 'LB(amplitude)', 'UB(amplitude)', 'acrophase', 'LB(acrophase)','UB(acrophase)'])
    
    names = df.test.unique()
    names = list(set(map(lambda x:x.split('_rep')[0], names)))
    names.sort()
    
    for name in names:
        df_pop = df[df.test.str.startswith(name)] 
        
        if save_folder:
            save_to = save_folder+"\\pop_"+name+'.pdf'
        else:
            save_to = ""

        res = population_fit_cosinor(df_pop, period = period, save_to = save_to, plot_on = plot_on)


        
        d = {'test': name, 
         'p':res['p_value'], 
         'amplitude': res['means'][-2],
         'LB(amplitude)': res['confint']['amp'][0],  
         'UB(amplitude)': res['confint']['amp'][1],
         'acrophase':res['means'][-1],
         'LB(acrophase)': res['confint']['acr'][0],  
         'UB(acrophase)': res['confint']['acr'][1],
         'acrophase[h]':cosinor.acrophase_to_hours(res['means'][-1],period)}
    
        df_cosinor1_fits = df_cosinor1_fits.append(d, ignore_index=True)
    
    df_cosinor1_fits['q'] = multi.multipletests(df_cosinor1_fits['p'], method = 'fdr_bh')[1]

    return df_cosinor1_fits    

def population_test_cosinor_pairs(df, pairs, period = 24):
    df_res = pd.DataFrame(columns = ['test', 
                                    'amplitude1', 
                                    'amplitude2',
                                    'p(d_amplitude)',
                                    'q(d_amplitude)',                              
                                    'acrophase1',
                                    'acrophase2',
                                    'p(d_acrophase)',
                                    'q(d_acrophase)'])

    for pair in pairs:
        df_pop1 = df[df.test.str.startswith(pair[0])] 
        df_pop2 = df[df.test.str.startswith(pair[1])] 

        res1 = population_fit_cosinor(df_pop1, period = period, plot_on = False)
        res2 = population_fit_cosinor(df_pop2, period = period, plot_on = False)

        res = population_test_cosinor(res1, res2)
        d = {'test': res['test'], 
            'amplitude1':res['amp']['pop1'], 
            'amplitude2':res['amp']['pop2'], 
            'p(d_amplitude)':res['amp']['p_value'],            
            'acrophase1':res['acr']['pop1'],
            'acrophase2':res['acr']['pop2'], 
            'p(d_acrophase)':res['acr']['p_value']}

        df_res = df_res.append(d, ignore_index=True)

    df_res['q(d_amplitude)'] = multi.multipletests(df_res['p(d_amplitude)'], method = 'fdr_bh')[1]
    df_res['q(d_acrophase)'] = multi.multipletests(df_res['p(d_acrophase)'], method = 'fdr_bh')[1]

    return df_res


def population_fit_cosinor(df_pop, period, save_to='', alpha = 0.05, plot_on = True, plot_individuals = True, plot_measurements=True, plot_margins=True):
    params = -1
    tests = df_pop.test.unique()
    k = len(tests)
    param_names = ['Intercept', 'rrr (beta)', 'sss (gamma)', 'amp', 'acr']
    cosinors = []
    
    test_name = tests[0].split('_rep')[0]
    
    for test in tests:
        x,y = np.array(df_pop[df_pop.test == test].x), np.array(df_pop[df_pop.test == test].y)
        fit_results, amp, acr, statistics = fit_cosinor(x, y, period = period, save_to=save_to, plot_on = False)
        if plot_on and plot_individuals:
            X_fit = np.linspace(min(x), max(x), 100)
            rrr_fit= np.cos(2*np.pi*X_fit/period)
            sss_fit = np.sin(2*np.pi*X_fit/period)
        
            data = pd.DataFrame()
            data['rrr'] = rrr_fit
            data['sss'] = sss_fit
            Y_fit = fit_results.predict(data)   
        
            plt.plot(X_fit, Y_fit, 'k', alpha=0.5)
        
            #M = fit_results.params[0]
            #y_fit = evaluate_cosinor(x, M, amp, acr, period)
            #plt.plot(x, y_fit, 'k')
        if plot_on and plot_measurements:
            plt.plot(x, y, 'ko', markersize=1)
        
        if type(params) == int:
            params = np.append(fit_results.params, np.array([amp, acr]))
        else:
            params = np.vstack([params, np.append(fit_results.params, np.array([amp, acr]))])
 
        cosinors.append(fit_results)
 
    if k > 1:
        means = np.mean(params, axis=0)
    else:
        means=params
    MESOR = means[0]
    beta = means[1]
    gamma = means[2]
    
    amp, acr = amp_acr(gamma, beta)
    means[3] = amp
    means[4] = acr
    
    if k > 1:
        sd = np.std(params, axis=0, ddof = 1)
        sdm = sd[0]
        sdb = sd[1]
        sdy = sd[2]
    
        covby = np.cov(params[:,1], params[:,2])[0,1]          
        denom=(amp**2)*k   
        c22=(((sdb**2)*(beta**2))+(2*covby*beta*gamma)+((sdy**2)*(gamma**2)))/denom
        c23=(((-1*((sdb**2)-(sdy**2)))*(beta*gamma))+(covby*((beta**2)-(gamma**2))))/denom
        c33=(((sdb**2)*(gamma**2))-(2*covby*beta*gamma)+((sdy**2)*(beta**2)))/denom
    
        t=abs(stats.t.ppf(alpha/2, df = k-1))
    
        mesoru=MESOR+((t*sdm)/(k**0.5))
        mesorl=MESOR-((t*sdm)/(k**0.5))
  
        ampu=amp+(t*(c22**0.5))
        ampl=amp-(t*(c22**0.5))
        if (ampu > 0 and ampl < 0):
            fiu=np.nan
            fil=np.nan
            print("Warning: Amplitude confidence interval contains zero. Acrophase confidence interval cannot be calculated and was set to NA.")
        else:
            fiu=acr+np.arctan(((c23*(t**2))+((t*np.sqrt(c33))*np.sqrt((amp**2)-(((c22*c33)-(c23**2))*((t**2)/c33)))))/((amp**2)-(c22*(t**2))))
            fil=acr+np.arctan(((c23*(t**2))-((t*np.sqrt(c33))*np.sqrt((amp**2)-(((c22*c33)-(c23**2))*((t**2)/c33)))))/((amp**2)-(c22*(t**2))))
    else:
        mesoru=MESOR
        mesorl=MESOR  
        ampu=amp
        ampl=amp
        fiu = acr
        fil = acr
        
    if plot_on:
        x = np.linspace(min(df_pop.x), max(df_pop.x), 100)
        y = evaluate_cosinor(x, MESOR, amp, acr, period)
        plt.plot(x, y, 'r')


        if plot_margins:

            me = np.linspace(mesoru, mesorl, 25)
            am = np.linspace(ampu, ampl, 25)
            fi = np.linspace(fiu, fil, 25)

            Y = y

            for m in me:
                for a in am:
                    for f in fi:
                        yy = evaluate_cosinor(x, m, a, f, period)
                        Y = np.vstack([Y, yy])
                        #plt.plot(x,yy)


            lower = np.min(Y, axis=0)        
            upper = np.max(Y, axis=0)        

            plt.fill_between(x, lower, upper, color='black', alpha=0.1)  


        if save_to:
            plt.savefig(save_to+'.pdf')
            plt.savefig(save_to+'.png')
            plt.close()
        else:  
            plt.show()
  
  
    #print("acr", acr)
    #print("ACR (h): ", -period * acr/(2*np.pi))
    
    confint = {'amp':(ampl, ampu),
               'acr':(fil, fiu),
               'MESOR':(mesorl, mesoru)}
  
    # calculate the p-value
    if k > 1:
        betas= params[:,1]
        gammas= params[:,2]
        r = np.corrcoef(betas,gammas)[0,1]
        frac1=(k*(k-2))/(2*(k-1))
        frac2=1/(1-r**2)
        frac3=beta**2/sdb**2
        frac4=(beta*gamma)/(sdb*sdy)
        frac5=gamma**2/sdy**2
        brack=frac3-(2*r*frac4)+frac5
        Fvalue=frac1*frac2*brack
        df2 = k-2
        p_value= 1 - stats.f.cdf(Fvalue, 2, df2)
        #print(p_value)
    
    else:
        p_value = np.nan
     
    return {'test': test_name, 'names':param_names, 'values':params, 'means':means, 'confint':confint, 'p_value':p_value}




def population_test_cosinor(pop1, pop2):
    k1 = pop1['values'].shape[0]
    k2 = pop2['values'].shape[0]
    K = k1 + k2
    params1 = pop1['values']
    params2 = pop2['values']
    means1 = pop1['means']
    means2 = pop2['means']
    
    mesors1 = params1[:,0]
    mesor1 = means1[0]
    mesors2 = params2[:,0]
    mesor2 = means2[0]
    betas1 = params1[:,1]
    beta1 = means1[1]
    betas2 = params2[:,1]
    beta2 = means2[1]
    gammas1 = params1[:,2]
    gamma1 = means1[2]
    gammas2 = params2[:,2]
    gamma2 = means2[2]
    amps1 = params1[:,3]
    amp1 = means1[3]
    amps2 = params2[:,3]
    amp2 = means2[3]
    acrs1 = params1[:,4]
    acr1 = means1[4]
    acrs2 = params2[:,4]
    acr2 = means2[4]
    
    M = (k1*mesor1 + k2*mesor2)/K
    A = (k1*amp1 + k2*amp2)/K
    FI = (k1*acr1 + k2*acr2)/K
    BETA = (k1*beta1 + k2*beta2)/K
    GAMMA = (k1*gamma1 + k2*gamma2)/K
    TM = (k1 * (mesor1 - M)**2) + (k2 * (mesor2 - M)**2)

    
    tann=((k1*(amp1**2))*np.sin(2*acr1))+((k2*(amp2**2))*np.sin(2*acr2))
    tand=((k1*(amp1**2))*np.cos(2*acr1))+((k2*(amp2**2))*np.cos(2*acr2))
    if tand > 0:
        twofi = np.arctan(tann/tand)
    else:
        twofi = np.arctan(tann/tand) + np.pi
    
    FITILDE = twofi/2
    varm1 = np.var(mesors1, ddof = 1)
    varm2 = np.var(mesors2, ddof = 1)
    varb1 = np.var(betas1, ddof = 1)
    varb2 = np.var(betas2, ddof = 1)
    vary1 = np.var(gammas1, ddof = 1)
    vary2 = np.var(gammas2, ddof = 1)
    
    covby1 = np.cov(betas1, gammas1)[0,1]          
    covby2 = np.cov(betas2, gammas2)[0,1]          
    
    varm=(((k1-1)*varm1)/(K-2))+(((k2-1)*varm2)/(K-2))
    varb=(((k1-1)*varb1)/(K-2))+(((k2-1)*varb2)/(K-2))
    vary=(((k1-1)*vary1)/(K-2))+(((k2-1)*vary2)/(K-2))
    covby=(((k1-1)*covby1)/(K-2))+(((k2-1)*covby2)/(K-2))
    FM = TM/varm
    acrn=(amp1**2+((np.sin(acr1-FITILDE))**2)) + (amp2**2+((np.sin(acr2-FITILDE))**2))
    acrd1=varb*((np.sin(FITILDE))**2)
    acrd2=2*covby*np.cos(FITILDE)*np.sin(FITILDE)
    acrd3=vary*((np.cos(FITILDE))**2)
    acrd=acrd1-acrd2+acrd3
    FFI=acrn/acrd
    ampn=((amp1-A)**2) + ((amp2-A)**2)
    ampd1=varb*((np.cos(FI))**2)
    ampd2=2*covby*np.cos(FI)*np.sin(FI)
    ampd3=vary*((np.sin(FI))**2)
    ampd=ampd1-ampd2+ampd3
    FA=ampn/ampd
    df1=1
    df2=K-2
    PM = 1 - stats.f.cdf(FM, df1, df2)
    PA = 1 - stats.f.cdf(FA, df1, df2)
    PFI = 1 - stats.f.cdf(FFI, df1, df2)
    
    if PFI < 0.05:
        print("Results of population amplitude difference test are not reliable due to different acrophases.")
    
    return {'test': pop1['test'] + ' vs ' +pop2['test'], 
            'mesor':{'pop1':mesor1, 'pop2':mesor2, 'p_value':PM},
            'amp':{'pop1':amp1, 'pop2':amp2, 'p_value':PA},
            'acr':{'pop1':acr1, 'pop2':acr2, 'p_value':PFI}}
            

def fit_group(df, period = 24, save_folder='', plot_on=True):
    df_cosinor1_fits = pd.DataFrame(columns = ['test', 'p', 'q', 'amplitude', 'p(amplitude)', 'q(amplitude)', 'acrophase', 'p(acrophase)','q(acrophase)'])
    
    if (type(period) == int) or (type(period)==float):
        period = [period] 

    for test in df.test.unique():
        x, y = df[df.test == test].x, df[df.test == test].y
        for per in period:
            if save_folder:
                fit_results, amp, acr, statistics = fit_cosinor(x, y, per, test=test, save_to=save_folder+"\\"+test+"_"+str(per)+".pdf",  plot_on = plot_on)
            else:
                fit_results, amp, acr, statistics = fit_cosinor(x, y, per, test=test, plot_on = plot_on)
            if acr <0:
                acr += 2 * np.pi
        
            d = {'test': test, 
            'p':fit_results.f_pvalue, 
            'amplitude': amp,
            'period': per,
            'p(amplitude)': statistics['p-values'][1], 
            'acrophase':acr,
            'p(acrophase)': statistics['p-values'][2],
            'acrophase[h]':cosinor.acrophase_to_hours(acr,per)}
        
            df_cosinor1_fits = df_cosinor1_fits.append(d, ignore_index=True)
        
    df_cosinor1_fits['q'] = multi.multipletests(df_cosinor1_fits['p'], method = 'fdr_bh')[1]
    df_cosinor1_fits['q(amplitude)'] = multi.multipletests(df_cosinor1_fits['p(amplitude)'], method = 'fdr_bh')[1]
    df_cosinor1_fits['q(acrophase)'] = multi.multipletests(df_cosinor1_fits['p(acrophase)'], method = 'fdr_bh')[1]

    return df_cosinor1_fits    
    

def fit_cosinor(X, Y, period, test='', save_to = '', plot_on = True):
         
    data = pd.DataFrame()
    data['x'] = X
    data['y'] = Y
     
    #fit_results, amp, acr = fit_cosinor_df(data, period)
    
    fit_results, amp, acr, statistics = test_cosinor_single(data, period)
    if plot_on:
        plot_single(data, fit_results, test=test, plot_measurements=True, save_to=save_to, plot_dense=True, period=period)       
    
    return fit_results, amp, acr, statistics

def test_cosinor_pairs(df, pairs, period = 24, folder = '', prefix='', plot_measurements=True, legend=True, df_best_models = -1):
    
    df_results = pd.DataFrame(columns = ['test',
                                         'p', 
                                         'q', 
                                         'amplitude1', 
                                         'p(amplitude1)', 
                                         'q(amplitude1)', 
                                         'amplitude2', 
                                         'p(amplitude2)', 
                                         'q(amplitude2)', 
                                         'd_amplitude', 
                                         'p(d_amplitude)', 
                                         'q(d_amplitude)',
                                         'acrophase1', 
                                         'p(acrophase1)',
                                         'q(acrophase1)',
                                         'acrophase2',
                                         'p(acrophase2)',
                                         'q(acrophase2)',
                                         'd_acrophase',
                                         'p(d_acrophase)',
                                         'q(d_acrophase)'])
    
    for test1, test2 in pairs:    
        if folder:
            save_to = folder +'\\'+prefix+ test1 + '-' + test2
        else:
            save_to = ''
        
        X1, Y1 = np.array(df[df.test == test1].x), np.array(df[df.test == test1].y)
        X2, Y2 = np.array(df[df.test == test2].x), np.array(df[df.test == test2].y)
        df_pair = prepare_df(X1, Y1, X2, Y2)

        idx_amp = 2
        idx_group_amp = 3
        idx_acr = 4
        idx_group_acr = 5

        if type(df_best_models) == int:
            fit_results, statistics_raw, statistics_trans, global_test_amp, ind_test_amp, global_test_acr, ind_test_acr = test_cosinor_pair(df_pair, period)
        else:            
            period = df_best_models[df_best_models.test == test1].period.iloc[0]
            fit_results, statistics_raw, statistics_trans, global_test_amp, ind_test_amp, global_test_acr, ind_test_acr = test_cosinor_pair(df_pair, period)
            #print(period)
        
        acr1, acr2 = statistics_trans['values'][idx_acr], statistics_trans['values'][idx_group_acr]
        amp1, amp2 = statistics_trans['values'][idx_amp], statistics_trans['values'][idx_group_amp]
        
        d = {'test':test1+' vs. '+test2,
            'p': fit_results.f_pvalue,
            'period': period,
            'amplitude1': statistics_trans['values'][idx_amp],
            'p(amplitude1)':statistics_trans['p-values'][idx_amp],            
            'amplitude2': statistics_trans['values'][idx_group_amp],
            'p(amplitude2)':statistics_trans['p-values'][idx_group_amp],
            'd_amplitude': ind_test_amp['value'],
            'p(d_amplitude)': float(ind_test_amp['p_value']),
            'acrophase1': statistics_trans['values'][idx_acr],
            'p(acrophase1)':statistics_trans['p-values'][idx_acr],
            'acrophase2': statistics_trans['values'][idx_group_acr],
            'p(acrophase2)':statistics_trans['p-values'][idx_group_acr],
            'd_acrophase': ind_test_acr['value'],
            'p(d_acrophase)': float(ind_test_acr['p_value'])}

        df_results = df_results.append(d, ignore_index=True)
        plot_pair(df_pair, fit_results, test1=test1, test2=test2, plot_measurements=plot_measurements, save_to=save_to, period=period, legend=legend)
        
        CI_l, CI_u = statistics_trans['CI']   
        CI_acrs = [(CI_l[-2],CI_u[-2]), (CI_l[-1], CI_u[-1])]
        CI_amps = [(CI_l[-4],CI_u[-4]), (CI_l[-3], CI_u[-3])]
        
        cosinor.plot_phases([acr1, acr2], [amp1, amp2], [test1, test2], period=24, folder = folder, prefix="phase_", legend=legend, CI_acrs = CI_acrs, CI_amps = CI_amps)
        
    df_results['q'] = multi.multipletests(df_results['p'], method = 'fdr_bh')[1]
    df_results['q(amplitude1)'] = multi.multipletests(df_results['p(amplitude1)'], method = 'fdr_bh')[1]
    df_results['q(amplitude2)'] = multi.multipletests(df_results['p(amplitude2)'], method = 'fdr_bh')[1]
    df_results['q(acrophase1)'] = multi.multipletests(df_results['p(acrophase1)'], method = 'fdr_bh')[1]
    df_results['q(acrophase2)'] = multi.multipletests(df_results['p(acrophase2)'], method = 'fdr_bh')[1]
    
    df_results['q(d_amplitude)'] = multi.multipletests(df_results['p(d_amplitude)'], method = 'fdr_bh')[1]
    df_results['q(d_acrophase)'] = multi.multipletests(df_results['p(d_acrophase)'], method = 'fdr_bh')[1]



    return df_results

def test_cosinor_single(data, period = 24):
    
    rrr = np.cos(2*np.pi*data.x/period)
    sss = np.sin(2*np.pi*data.x/period)
            
        
    data['rrr'] = rrr
    data['sss'] = sss

    results = smf.ols('y ~ rrr + sss', data).fit()

    beta_s = results.params['sss']
    beta_r = results.params['rrr']
    amp, acr = amp_acr(beta_s, beta_r)
    
    vmat = results.cov_params().loc[['rrr', 'sss'], ['rrr', 'sss']]
    indVmat = vmat
        
    a_r = (beta_r**2 + beta_s**2)**(-0.5) * beta_r
    a_s = (beta_r**2 + beta_s**2)**(-0.5) * beta_s
    b_r = (1 / (1 + (beta_s**2 / beta_r**2))) * (-beta_s / beta_r**2)
    b_s = (1 / (1 + (beta_s**2 / beta_r**2))) * (1 / beta_r)
    
    jac = np.array([[a_r, a_s], [b_r, b_s]]) 
    
    cov_trans = np.dot(np.dot(jac, indVmat), np.transpose(jac))
    se_trans_only =np.sqrt(np.diag(cov_trans))
    zt = abs(norm.ppf((1-0.95)/2))

    trans_names = [results.params.index.values[0]] + ['amp', 'acr']
    coef_trans = np.array([results.params.iloc[0], amp, acr])
    se_trans = np.concatenate((np.sqrt(np.diag(results.cov_params().loc[['Intercept'], ['Intercept']])),se_trans_only))  

    lower_CI_trans = coef_trans - zt * se_trans           
    upper_CI_trans = coef_trans + zt * se_trans
    p_value_trans = 2 * norm.cdf(-np.abs(coef_trans/se_trans))

    statistics= {'parameters': trans_names,
                    'values': coef_trans,
                    'SE': se_trans,
                    'CI': (lower_CI_trans, upper_CI_trans),
                    'p-values': p_value_trans,
                    'F-test': results.f_pvalue}    
       
    return results, amp, acr, statistics
                    
    


def test_cosinor_pair(data, period):
    
    rrr = np.cos(2*np.pi*data.x/period)
    sss = np.sin(2*np.pi*data.x/period)
            
        
    data['rrr'] = rrr
    data['sss'] = sss

    results = smf.ols('y ~ group + rrr + sss + group:rrr + group:sss', data).fit()

    beta_s = np.array([results.params['sss'], results.params['group:sss']])
    beta_r = np.array([results.params['rrr'], results.params['group:rrr']])
        
    groups_s = beta_s.copy()
    groups_s[1] += groups_s[0]
    groups_r = beta_r.copy()
    groups_r[1] += groups_r[0]
        
    amp, acr = amp_acr(groups_s, groups_r)
          
    vmat = results.cov_params().loc[['rrr', 'group:rrr', 'sss', 'group:sss'],['rrr', 'group:rrr', 'sss', 'group:sss'],]   
    indexmat = np.eye(4)
    indexmat[1,0]=1
    indexmat[3,2]=1
    indVmat = np.dot(np.dot(indexmat, vmat), np.transpose(indexmat))

    a_r = (groups_r**2 + groups_s**2)**(-0.5) * groups_r
    a_s = (groups_r**2 + groups_s**2)**(-0.5) * groups_s

    b_r = (1 / (1 + (groups_s**2 / groups_r**2))) * (-groups_s / groups_r**2)
    b_s = (1 / (1 + (groups_s**2 / groups_r**2))) * (1 / groups_r)

    jac = np.array((np.concatenate((np.diag(a_r)[0,:], np.diag(a_s)[0,:])),
    np.concatenate((np.diag(a_r)[1,:], np.diag(a_s)[1,:])),
    np.concatenate((np.diag(b_r)[0,:], np.diag(b_s)[0,:])),
    np.concatenate((np.diag(b_r)[1,:], np.diag(b_s)[1,:]))))

    cov_trans = np.dot(np.dot(jac, indVmat), np.transpose(jac))
    se_trans_only =np.sqrt(np.diag(cov_trans))
    
    
    zt = abs(norm.ppf((1-0.95)/2))
    
    
    coef_raw = results.params
    raw_names = list(coef_raw.index.values)
    coef_raw = coef_raw.values
    se_raw = np.sqrt(np.diag(results.cov_params()))
    lower_CI_raw = coef_raw - zt * se_raw
    upper_CI_raw = coef_raw + zt * se_raw
    p_value_raw = 2 * norm.cdf(-np.abs(coef_raw/se_raw))
    statistics_raw={'parameters': raw_names,
                    'values': coef_raw,
                    'SE': se_raw,
                    'CI': (lower_CI_raw, upper_CI_raw),
                    'p-values': p_value_raw}    
    
    trans_names = list(results.params.index.values[:2]) + ['amp', 'group:amp', 'acr', 'group:acr']
    coef_trans = np.concatenate((np.array(results.params.iloc[0:2]), amp, acr))
    se_trans = np.concatenate((np.sqrt(np.diag(results.cov_params().loc[['Intercept', 'group'], ['Intercept', 'group']])),se_trans_only))             
    lower_CI_trans = coef_trans - zt * se_trans
    upper_CI_trans = coef_trans + zt * se_trans
    p_value_trans = 2 * norm.cdf(-np.abs(coef_trans/se_trans))
    statistics_trans={'parameters': trans_names,
                    'values': coef_trans,
                    'SE': se_trans,
                    'CI': (lower_CI_trans, upper_CI_trans),
                    'p-values': p_value_trans}    
    
    
    
    diff_est_amp = coef_trans[3] - coef_trans[2]
    idx = np.array([-1,1,0,0])
    diff_var_amp = np.dot(np.dot(idx, cov_trans), np.transpose(idx[np.newaxis]))
    glob_chi_amp = diff_est_amp * (1/diff_var_amp) * diff_est_amp
    ind_Z_amp = diff_est_amp/np.sqrt(np.diag(diff_var_amp))
    interval_amp = np.array((diff_est_amp, diff_est_amp - 1.96 * np.sqrt(np.diag(diff_var_amp)), diff_est_amp + 1.96 * np.sqrt(np.diag(diff_var_amp))))
    df_amp = 1
    global_p_value_amp = 1-stats.chi2.cdf(glob_chi_amp, df_amp)
    ind_p_value_amp = 2*norm.cdf(-abs(ind_Z_amp))

    diff_est_acr = coef_trans[5] - coef_trans[4]
    idx = np.array([0,0,-1,1])
    diff_var_acr = np.dot(np.dot(idx, cov_trans), np.transpose(idx[np.newaxis]))
    glob_chi_acr = diff_est_acr * (1/diff_var_acr) * diff_est_acr
    ind_Z_acr = diff_est_acr/np.sqrt(np.diag(diff_var_acr))
    interval_acr = np.array((diff_est_acr, diff_est_acr - 1.96 * np.sqrt(np.diag(diff_var_acr)), diff_est_acr + 1.96 * np.sqrt(np.diag(diff_var_acr))))
    df_acr = 1
    global_p_value_acr = 1-stats.chi2.cdf(glob_chi_acr, df_acr)
    ind_p_value_acr = 2*norm.cdf(-abs(ind_Z_acr))
    
    
    global_test_amp = {'name': 'global test of amplitude change',
                       'statistics': glob_chi_amp,
                       'df': df_amp,
                       'p_value': global_p_value_amp}
    ind_test_amp = {'name': 'individual test of amplitude change',
                    'statistics': ind_Z_amp,
                    'df': np.nan,
                    'value': interval_amp[0],
                    'conf_int': interval_amp[1:],
                    'p_value': ind_p_value_amp}

    global_test_acr = {'name': 'global test of acrophase shift',
                       'statistics': glob_chi_acr, 
                       'df': df_acr,
                       'p_value': global_p_value_acr}
    ind_test_acr = {'name': 'individual test of acrophase shift',
                    'statistics': ind_Z_acr,
                    'df': np.nan,
                    'value': interval_acr[0],
                    'conf_int': interval_acr[1:],
                    'p_value': ind_p_value_acr}
    
    return results, statistics_raw, statistics_trans, global_test_amp, ind_test_amp, global_test_acr, ind_test_acr
                    

# amplitude_detection: determines the minimal number of samples to obtain a statistically significant result for the zero amplitude test
# A ... presumed amplitude
# var ... residual variance
# p ... threshold for statistical significance of the zero amplitude test
# alpha ... type I error probability
def amplitude_detection(A, var, p = 0.95, alpha = 0.05):

    sigma2 = var
    C_2 = A**2/sigma2 

    N = 4
    
    while True:
        lmbda = (N * C_2)/4       
        f2 = f(2, N-3, 0).ppf(1-alpha)
        F = 1-ncf(2,N-3,lmbda).cdf(f2)
        if F >= p:
            break
        
        N += 1
        
    return N

# amplitude_confidence: determines the minimal number of samples to obtain a given length of the confidence interval for the estimated amplitude
# L ... maximal acceptable length of the confidence interval
# var ... residual variance
# alpha ... 1-alpha = confidence level of the cofidence interval
def amplitude_confidence(L, var, alpha = 0.05):
    sigma2 = var
    N = 8 * sigma2 * (norm.ppf(1-alpha/2)**2)/(L**2)
    return int(np.ceil(N))

# acrophase_confidence: determines the minimal number of samples to obtain a given length of the confidence interval for the estimated acrophase
# L ... maximal acceptable length of the confidence interval
# A_0 ... presumed minimal amplitude
# var ... residual variance
# alpha ... 1-alpha = confidence level of the cofidence interval
def acrophase_confidence(L, A_0, var, alpha = 0.05):
    sigma2 = var
    #N = 8 * sigma2 * (norm.ppf(1-alpha/2)**2)/(L**2 * A_0**2) # approximation
    N = (2 * norm.ppf(1-alpha/2)**2 * sigma2)/(A_0**2 * np.sin(L/2)**2) # accurate

    return int(np.ceil(N))

# acrophase_shift_detection: determines the minimal number of samples to detect a specific shift in the acrophase
def acrophase_shift_detection(shift, A_0, var, alpha = 0.05):
    L = 0.5*shift
    return acrophase_confidence(L, A_0, var, alpha)    

    

