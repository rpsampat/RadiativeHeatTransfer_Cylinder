import matplotlib.pyplot as plt
import openpyxl
import pickle
import math
#import GasComp as GC
# import ExhaustOrifice as EO
import numpy as np
from pathlib import Path
import os
class WSGG_model:
    def __init__(self):

        self.path_dir = os.path.split(os.getcwd())[0]
        self.P = 5 # atm
        self.num_graygas = 4
        self.emissivity_poly_order=5
        self.emissivity_poly_order_co2 = 5
        self.read_coeff()


    def read_sheet(self,sheet):
        data = {}
        columns = [cell.value for cell in sheet[1]]
        for row in sheet.iter_rows(min_row=2, values_only=True):
            data_list = dict(zip(columns, row))
            data[data_list['j']]=data_list
        return data

    def read_coeff(self):
        # Specify the Excel file path
        excel_file_path = self.path_dir+'\data'+'/2014_Cassol_WSGGcoeff.xlsx'

        # Open the Excel workbook
        workbook = openpyxl.load_workbook(excel_file_path, read_only=True)

        # Read data from the "h2o" sheet
        h2o_sheet = workbook['h2o']
        self.h2o_data = self.read_sheet(h2o_sheet)

        # Read data from the "co2" sheet
        co2_sheet = workbook['co2']
        self.co2_data = self.read_sheet(co2_sheet)

        # Close the workbook
        workbook.close()

    def temperature_coeff(self,T):
        """
        a_xj = sum^{Kx}_{k=1} b_xjk T^(k-1)
        :return:
        """
        a_h2o = {}
        a_co2 = {}
        K_h2o = {}
        K_co2 = {}
        for j in range(self.num_graygas):
            val=0
            val2 = 0
            K_h2o[j + 1] = self.h2o_data[j + 1]['K_j']
            K_co2[j + 1] = self.co2_data[j + 1]['K_j']
            for k in range(self.emissivity_poly_order):
                val += self.h2o_data[j + 1]['b'+str(k + 1)]*(T**(k))
                val2 += self.co2_data[j + 1]['b' + str(k + 1)] * (T ** (k))
            a_h2o[j + 1] = val
            a_co2[j + 1] = val2

        return a_h2o, a_co2, K_h2o, K_co2

    def emissivity(self,P,X_h2o,X_co2,T,L):
        a_h2o, a_co2, K_h2o, K_co2 = self.temperature_coeff(T)
        K_eff= {}
        a_eff = {}
        epsilon=0
        epsilon_co2 = 0
        epsilon_h2o = 0
        count = 0
        a_h2o_sum = np.sum([a_h2o[i] for i in list(a_h2o.keys())])
        a_co2_sum = np.sum([a_co2[i] for i in list(a_co2.keys())])
        # print("Temp=",T)
        # print("Sum_ a_h2o=",a_h2o_sum)
        # print(a_co2_sum)
        kappa_eff={}

        for j in range(self.num_graygas+1):
            if j==0:
                # transparency window
                a_h2o_eff = 1-np.sum([a_h2o[i] for i in list(a_h2o.keys())])
                K_h2o_eff = 0
            else:
                a_h2o_eff = a_h2o[j]
                K_h2o_eff = K_h2o[j]
            for j1 in range(self.num_graygas+1):
                if j1==0:
                    # transparency window
                    a_co2_eff = 1-np.sum([a_co2[i] for i in list(a_co2.keys())])
                    K_co2_eff = 0
                else:
                    a_co2_eff = a_co2[j1]
                    K_co2_eff = K_co2[j1]
                K_eff[count] = (P*X_h2o*K_h2o_eff + P*X_co2*K_co2_eff)/(P*(X_h2o+X_co2))
                a_eff[count] = a_h2o_eff*a_co2_eff
                #if a_eff[count] < 0:
                  #  print("Aneg=", a_eff[count])

                if abs(K_eff[count])>0:
                    epsilon+=a_eff[count]*(1-math.e**(-K_eff[count]*P*(X_h2o+X_co2)*L))
                kappa_eff[count] = K_eff[count] * P * (X_h2o + X_co2)
                count+=1

        a_sum = np.sum([a_eff[i] for i in list(a_eff.keys())])
        # print("a_sum=",a_sum)


        for j in range(self.num_graygas+1):
            if j==0:
                K_co2_eff = 0
                K_h2o_eff = 0
                a_co2_eff = 1-a_co2_sum
                a_h2o_eff = 1-a_h2o_sum
            else:
                K_co2_eff = K_co2[j]
                K_h2o_eff = K_h2o[j]
                a_co2_eff = a_co2[j]
                a_h2o_eff = a_h2o[j]

            epsilon_co2+=a_co2_eff*(1-math.e**(-K_co2_eff*P*(X_co2)*L))
            epsilon_h2o += a_h2o_eff * (1 - math.e ** (-K_h2o_eff * P * (X_h2o) * L))

        # print("Epsi_h2o=",epsilon_h2o)

        """print(a_eff)
        print(K_eff)
        print(epsilon_co2)
        print(epsilon_h2o)"""
        #epsilon = (epsilon_co2)+(epsilon_h2o)
        return epsilon,epsilon_h2o,epsilon_co2, kappa_eff, a_eff

    def flamelet_data(self):
        H2_perc = [0,10,50,80,100]#[0, 50, 80, 100]  # [0,80]#
        phi_list = [0.3, 0.35,0.5,0.6,0.7, 0.8,0.9, 1.0]#[0.3, 0.6, 0.8, 1.0]  # [0.6,1.0]#
        # reference condition
        power_ref = 70  # kW
        phi_ref = 1.0
        T_heater_ref = 600  # K
        vdot_N2 = 0
        H2_perc_ref = 0.0
        settings = GC.Settings('CH4-H2', power_ref, phi_ref, T_heater_ref, vdot_N2, H2_perc_ref)
        species_names = np.array(settings.species_names)
        settings.solve_reactor="on"
        ind_h2o = np.where(species_names == 'H2O')[0][0]
        ind_co2 = np.where(species_names == 'CO2')[0][0]
        flamelet_data={}
        for j in range(len(H2_perc)):
            epsi_plot = []
            flamelet_data[H2_perc[j]]={}
            for phi in phi_list:
                settings.phi = phi
                settings.H2_perc = H2_perc[j]
                settings.main()
                X=settings.exhaust_gas
                flamelet_data[H2_perc[j]][phi]={'H2O':X[ind_h2o], 'CO2':X[ind_co2], 'T':settings.T_exhaust, 'X':X}
        with open("Flamelet_data",'wb') as f:
            pickle.dump(flamelet_data, f, pickle.HIGHEST_PROTOCOL)

    def main(self):
        L= 0.100 # atm
        P = 1 # atm
        X_h2o = 0.5
        X_co2 = 0.1
        T = 1400 #K
        self.read_coeff()
        epsi = self.emissivity(P,X_h2o,X_co2,T,L)
        print("Emissivity=",epsi)

    def main_doe(self):
        H2_perc = [0, 50, 80, 100]  # [0,80]#
        phi_list = [0.3, 0.6, 0.8, 1.0]  # [0.6,1.0]#
        try:
            with open("Flamelet_data", 'rb') as f:
                FD = pickle.load(f)
        except:
            self.flamelet_data()
            with open("Flamelet_data", 'rb') as f:
                FD = pickle.load(f)
        self.read_coeff()
        L = 0.100  # m
        P = 1  # atm
        T = 2000  # K

        # reference condition
        power_ref = 70  # kW
        phi_ref = 1.0
        T_heater_ref = 600  # K
        vdot_N2 = 0
        H2_perc_ref = 0.0
        settings = GC.Settings('CH4-H2', power_ref, phi_ref, T_heater_ref, vdot_N2, H2_perc_ref)

        fig,ax=plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        for j in range(len(H2_perc)):
            epsi_plot = []
            Mr_list = [] # ratio of mole fraction of h2o to co2
            nu_list = []
            for phi in phi_list:
                X_tot = FD[H2_perc[j]][phi]['H2O']+FD[H2_perc[j]][phi]['CO2']
                X_co2 = FD[H2_perc[j]][phi]['CO2']/X_tot
                X_h2o = FD[H2_perc[j]][phi]['H2O']/X_tot
                print(P*(FD[H2_perc[j]][phi]['H2O']+FD[H2_perc[j]][phi]['CO2'])*L)
                Mr_list.append(X_h2o/X_co2)
                epsi = self.emissivity(P, FD[H2_perc[j]][phi]['H2O'], FD[H2_perc[j]][phi]['CO2'], 2000, L)
                Tmax= FD[H2_perc[j]][phi]['T']
                epsi_plot.append(epsi)
                settings.phi = phi
                settings.H2_perc = H2_perc[j]
                settings.main()
                nu_list.append(settings.nu_in)
            ax.plot(phi_list,epsi_plot)
            ax1.plot(phi_list,Mr_list)
            ax2.plot(phi_list,nu_list)
        ax.set_ylabel("Total emittance")
        ax.set_xlabel("Equivalence ratio")
        ax1.set_ylabel("Molar ratio")
        ax1.set_xlabel("Equivalence ratio")
        ax2.set_ylabel("Kinematic viscosity")
        ax2.set_xlabel("Equivalence ratio")
        plt.show()

    def main_emissivity_kero_h2(self):
        EO_obj = EO.Exhaust_Orifice_Design()
        power=100#kW
        alpha = 0.9
        phi = 0.6
        self.P = 10 #atm
        alpha_list = [0,1]#[0,0.3,0.5,0.8,0.9,0.95,1.0]#np.arange(0.0,1.05,0.1)
        phi_list = np.arange(0.3,1.1,0.1)
        T_list = [500,1000,1500,2000,2500]
        fig,ax = plt.subplots()
        for alpha in alpha_list:
            epsi_list=[]
            for phi in phi_list:
                #for T in T_list:
                mdot_H2 = alpha * power / (EO_obj.LHV_H2 * 1000)
                mdot_CxHy = (1 - alpha) * power / (EO_obj.LHV_CxHy * 1000)
                n_CxHy = (mdot_CxHy / EO_obj.MW_CxHy)
                n_H2 = mdot_H2 / EO_obj.MW_H2
                n_H2, n_air, n_CO2, n_H2O, n_O2_out, n_N2_out =EO_obj.dual_fuel_combustion_mols(phi,n_CxHy,n_H2)
                N_sum  = n_CO2+n_H2O+n_O2_out+n_N2_out
                X_CO2 = n_CO2/N_sum
                X_H2O = n_H2O/N_sum
                #print('CO2=',X_CO2)
                #print('H2O=',X_H2O)
                mdot_mix, T_mix, rho_mix, gamma_mix, mdot_tot, T_comb, rho_comb, gamma_comb = EO_obj.massflows(alpha, power, phi,
                                                                                                             0.2)
                T_ref = T_comb
                #if T_ref>2500:
                  #  T_ref = 2500
                epsi,epsi_h2o,epsi_co2 = self.emissivity(self.P, X_H2O, X_CO2, T=T_ref, L=0.1)
                print(epsi)
                epsi_list.append(epsi)

            ax.plot(phi_list,epsi_list)


        #print(epsi)
        alpha_str = [str(f'{a:.2f}') for a in alpha_list]  # Convert list to comma-separated string
        ax.legend(alpha_str, title=r'$\alpha$')
        ax.set_xlabel('Equivalence Ratio')
        ax.set_ylabel(r'$\epsilon$')
        plt.show()
        fig.savefig('Epsilon_WSGG_kero_h2.png', bbox_inches='tight', dpi=600)

if __name__=="__main__":
    WSGG = WSGG_model()
    WSGG.main()
    #WSGG.flamelet_data()
    # WSGG.main_emissivity_kero_h2()
    #WSGG.main_doe()
