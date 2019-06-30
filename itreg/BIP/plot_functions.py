# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:57:00 2019

@author: Björn Müller
"""
import matplotlib.pyplot as plt 
import numpy as np

def plot_lastiter(pdf, exact_solution, exact_data, data):
    plt.plot(pdf.setting.domain.discr.coords, exact_solution.T, label='exact solution')
    plt.plot(pdf.setting.domain.discr.coords, pdf.reco, label='reco')
    plt.title('solution')
    plt.legend()
    plt.show()
    
    plt.plot(pdf.setting.codomain.discr.coords, exact_data, label='exact data')
    plt.plot(pdf.setting.codomain.discr.coords, data, label='data')
    plt.plot(pdf.setting.codomain.discr.coords, pdf.reco_data, label='reco data')
    plt.legend()
    plt.title('data')
    plt.show()
    
def plot_mean(statemanager, exact_solution, n_list=None, n_iter=None, variance=None):
    variance=variance or np.array([3])
#    print(variance[0])
#    if type(variance) is int or float:
#        variance=np.array([variance])
    n_plots=np.size(variance)
    if n_list is None and n_iter is None:
        raise ValueError('Specify the evaluation points')
    if n_list is not None:
        assert n_list.max()<statemanager.N
        a = np.array([s.positions for s in statemanager.states[n_list]])        
    else:
        assert n_iter<statemanager.N
        a = np.array([s.positions for s in statemanager.states[-n_iter:]])
    for i in range(0, n_plots):
        v = a.std(axis=0)*variance[i]
        m = a.mean(axis=0)
        plt.plot(np.array([m+v]).T, label='mean +' +str(variance[i])+ '*variance')
        plt.plot(np.array([m-v]).T, label='mean- '+str(variance[i])+'*variance')
    plt.plot(exact_solution.T, label='exact solution')
    plt.legend()
    plt.show()
    
def plot_verlauf(statemanager, pdf=None, exact_solution=None, plot_real=False):
    arr=[s.log_prob for s in statemanager.states]
    maximum=np.asarray([arr]).max()
    plt.plot(range(0, statemanager.N), arr/maximum, label='iterated log_prob')
    if plot_real is True:
        if pdf is None or exact_solution is None:
            raise ValueError('Specify the log_prob of exact solution')
        plt.plot(range(0, statemanager.N), pdf.log_prob(exact_solution)*np.ones(statemanager.N)/maximum, label='exact solution')
    plt.xlabel('iterations')
    plt.ylabel('log probability')
    plt.legend()
    plt.show()
    
def plot_iter(pdf, statemanager, position):  
    assert type(position)==int
    plt.plot(range(0, statemanager.N), [s.positions[position] for s in statemanager.states])
    plt.xlabel('iterations')
    plt.ylabel('value at x='+str(round(pdf.setting.domain.discr.coords[0, position], 2)))
    plt.show()
