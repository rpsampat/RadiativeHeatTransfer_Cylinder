import numpy as np
import CombustorHeatTransfer as CHT
import os
# from mpi4py import MPI
from multiprocessing import Pool, cpu_count
import itertools
import sys
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
def DOE():
    L = 0.3  # m
    D = 0.10  # m
    D_Casing = 0.1 + 0.05
    liner_wall_thickness = 0.003
    casing_wall_thickness = 0.003
    Pin = 10e5 # Pa
    Tin = 1100 # K
    Uf = 0.6
    beta = 0.0
    EAR = 3
    Power = 1.5e6 #W
    BR = 0.01
    HT_obj = CHT.HeatTransfer(L, D, D_Casing, liner_wall_thickness,casing_wall_thickness, Pin, Tin, Uf, beta, EAR, Power, BR)
    BR_list = np.linspace(0.35,2.0,5)#np.linspace(0.35,2.0,5)
    print(BR_list)
    dict_save={}
    Tin_list = np.arange(800,1300,200)
    Uf_list = np.arange(0.0,0.95,0.3)
    EAR_list = np.arange(2,6,1)

    # Generate all parameter combinations
    param_combinations = list(itertools.product(Tin_list, Uf_list, EAR_list, BR_list))

    return param_combinations, Pin, beta



def run_simulation(params):
    L = 0.3  # m
    D = 0.10  # m
    D_Casing = 0.1 + 0.05
    liner_wall_thickness = 0.003
    casing_wall_thickness = 0.003
    Pin = 10e5  # Pa
    Power = 1.5e6  # W

    Tin, Uf, EAR, BR, beta = params
    HT_obj = CHT.HeatTransfer(L, D, D_Casing, liner_wall_thickness, casing_wall_thickness, Pin, Tin, Uf, beta, EAR,
                              Power, BR)
    # HT_obj.Tin = Tin
    # HT_obj.Uf = Uf
    # HT_obj.EAR = EAR
    # HT_obj.BR = BR

    HT_obj.initialize()
    ht_dict = HT_obj.solver()
    caseid = f'Tin_{Tin}_Uf_{Uf:.2f}_EAR_{EAR}_BR_{BR}_Beta_{beta}'
    return caseid, ht_dict

if __name__ == '__main__':
    Pin = 10e5 # Pa
    Tin_list = np.arange(800, 1300, 200)
    Uf_list = np.arange(0.0, 0.95, 0.3)
    EAR_list = np.arange(2, 6, 1)
    BR_list = np.linspace(0.3, 2.0, 5)#np.linspace(0.01,0.1,5)#
    Beta_list = [0.0]#np.linspace(0.0,0.5,5)
    # Generate all parameter combinations
    param_combinations = list(itertools.product(Tin_list, Uf_list, EAR_list, BR_list,Beta_list))

    # Use all available cores
    num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        results = pool.map(run_simulation, param_combinations)

    # Collect results into a dictionary
    dict_save = dict(results)
    dict_save['Tin'] = Tin_list
    dict_save['Uf'] = Uf_list
    dict_save['EAR'] = EAR_list
    dict_save['BR'] = BR_list
    dict_save['Beta'] = Beta_list
    with open(parentdir+'/data/' + 'dictsave_DOE_Pin_'+str(Pin) +'_highBR' + '.pkl', 'wb') as file:
        pickle.dump(dict_save, file, pickle.HIGHEST_PROTOCOL)
