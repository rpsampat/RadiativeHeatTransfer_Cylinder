import numpy as np
import CombustorHeatTransfer as CHT
import os
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
    BR_list = np.linspace(0.01,0.1,10)
    print(BR_list)
    dict_save={}
    for BR in BR_list:
        print(BR)
        HT_obj.BR = BR
        HT_obj.initialize()
        ht_dict = HT_obj.solver()
        caseid = 'BR' + str(BR)
        dict_save[caseid]=ht_dict
    with open(parentdir+'/data/' + 'dictsave_DOE_BR' + '.pkl', 'wb') as file:
        pickle.dump(dict_save, file, pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    DOE()