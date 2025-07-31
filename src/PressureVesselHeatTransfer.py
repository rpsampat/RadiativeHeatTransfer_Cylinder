import numpy as np
import math
import cantera as ct
import pickle
import FilmCoolingLiner
import mesh
import RadiationCode
import WallHeatTransfer
import FluidAnnulusHeatTransfer
import CombustorCore
import WSGG_model
import scipy
import effusion_cooling_holes as ECH
import os
import sys
import inspect
path_PV_design = 'C:/Users/rishikeshsampa/OneDrive - Delft University of Technology/Documents/PostDoc Research/Pressurized Combustor Design/Design/Codes/src/'
sys.path.insert(0, path_PV_design)
import ExhaustOrifice


class HeatTransfer:
    def __init__(self, L, D, D_Casing, dia_orifice, liner_wall_thickness, casing_wall_thickness, Pin, Tin, alpha, Power, phi):
        self.L = L
        self.D = D
        self.D_Casing = D_Casing
        self.dia_orifice = dia_orifice
        self.liner_wall_thickness = liner_wall_thickness
        self.casing_wall_thickness = casing_wall_thickness
        # Combustor inlet state
        self.P_in = Pin
        self.T_in = Tin
        self.power = Power
        self.alpha = alpha # H2 mix
        self.phi = phi


        # Properties
        self.LHV_H2_mass = 120  # MJ/kg
        self.LHV_H2_mole = self.LHV_H2_mass * 2  # MJ/kmol
        self.sigma = 5.67e-8
        self.chem_mech = 'H2_11_20.yaml'
        self.verbose = 'n'  # y/n

        # Mesh params
        self.axial_elem = 10  # 30
        self.azimuthal_elem = 4  # 10
        self.radial_elem_hotgas = 6  # 20
        self.thickness_liner_elem = 5
        self.thickness_casing_elem = 5

    def wall_temperature(self, mesh_surf, Twall):
        # Twall = 1500 # K
        T_surf = Twall * np.ones(mesh_surf.N_surf)

        return T_surf

    def vol_temperature(self, mesh_vol):
        Tmax = 2000  # K
        z0 = max(mesh_vol.z) / 3
        Tmin = 1100
        alpha = 100.0
        T_vol = Tmax * np.ones(mesh_vol.N_vol)
        for i in range(len(mesh_vol.z)):
            z = mesh_vol.z[i]
            T_vol[i] = Tmin + (Tmax - Tmin) / ((1 + (math.e ** (-alpha * (z - z0)))))

        return T_vol

    def inlet_composition_v2(self, U_f, EAR, Beta):
        """

        :param U_f: fuel utilization in SOFC
        :param EAR: Excess air ratio (lambda) for overall SOFC+GT
        :param Beta: hydrogen power fraction given directly to combustor Beta=n_h2_direct*LHV_H2/Power
        :return: Dictionary of net composition entering the combustor
        """
        delta = 0.5 * ((EAR / 0.21) - ((1 - Beta) * U_f) + 2)  # denominator factor for total moles
        X_h2 = (1 - U_f + (Beta * U_f)) / delta
        X_h2o = (1 - Beta) * U_f / delta
        X_o2 = (0.5 * (EAR - ((1 - Beta) * U_f))) / delta
        X_n2 = (0.5 * EAR * 0.79 / 0.21) / delta
        X_dict = {'O2': X_o2, 'N2': X_n2, 'H2': X_h2, 'H2O': X_h2o}

        return X_dict

    def initialize(self):

        # Core composition and mass flow
        EO = ExhaustOrifice.Exhaust_Orifice_Design()
        WSGG = WSGG_model.WSGG_model()
        WSGG.P = self.P_in  # atm
        EO.P = self.P_in
        EO.dia_orifice = self.dia_orifice  # m
        mdot_H2 = self.alpha * self.power / (EO.LHV_H2 * 1000)
        mdot_CxHy = (1 - self.alpha) * self.power / (EO.LHV_CxHy * 1000)
        n_CxHy = (mdot_CxHy / EO.MW_CxHy)
        n_H2 = mdot_H2 / EO.MW_H2
        n_H2, n_air, n_CO2, n_H2O, n_O2_out, n_N2_out = EO.dual_fuel_combustion_mols(self.phi, n_CxHy, n_H2)
        N_sum = n_CO2 + n_H2O + n_O2_out + n_N2_out
        X_CO2 = n_CO2 / N_sum
        X_H2O = n_H2O / N_sum
        root = scipy.optimize.root(EO.exhaust_massflow, 0.3, args=(self.alpha, self.power, self.phi))
        mdot_pr_air = root.x[0]
        mdot_mix, T_mix, rho_mix, gamma, mdot_comb, T_comb, rho_comb, gamma_comb = EO.massflows(self.alpha, self.power,
                                                                                                self.phi,
                                                                                                mdot_pr_air)

        mdot_pr_air = 6000 * 1.225 / 60000
        print('T_comb=', T_comb)
        print('mdot_pr=', mdot_pr_air)
        print('mdot_core=', mdot_comb * 60000 / 1.225)
        # emissivity of gas
        epsi, epsi_h2o, epsi_co2, kappa_eff, a_eff = WSGG.emissivity(WSGG.P, X_H2O, X_CO2, T=T_comb, L=0.1)
        print('epsi_g=', epsi)

        self.mdot_comb_in = mdot_comb

        # Combustor liner interior mesh (quartz tube)
        self.mesh_surf_obj = mesh.Surf_Cyl_Mesh()
        self.mesh_surf_obj.num_axial = self.axial_elem
        self.mesh_surf_obj.num_azimuthal = self.azimuthal_elem
        self.mesh_surf_obj.L = self.L  #
        self.mesh_surf_obj.D = self.D
        self.mesh_surf_obj.initialize()

        self.mesh_vol_obj = mesh.Vol_Cyl_Mesh()
        self.mesh_vol_obj.num_axial = self.axial_elem
        self.mesh_vol_obj.num_azimuthal = self.azimuthal_elem
        self.mesh_vol_obj.num_radial = self.radial_elem_hotgas
        self.mesh_vol_obj.L = self.L
        self.mesh_vol_obj.D = self.D
        self.mesh_vol_obj.initialize()

        # Combustor Liner wall mesh (quartz tube)
        self.mesh_wall_liner = mesh.Surf_Cyl_Wall_Mesh()
        self.mesh_wall_liner.num_axial = self.axial_elem
        self.mesh_wall_liner.num_thickness = self.thickness_liner_elem
        self.mesh_wall_liner.thickness = self.liner_wall_thickness
        self.mesh_wall_liner.L = self.L  #
        self.mesh_wall_liner.D = self.D
        self.mesh_wall_liner.initialize()
        self.mdot_comb = self.mdot_comb_in * np.ones(self.axial_elem)

        # Combustor Casing interior mesh (Pressure Vessel)
        self.mesh_surf_obj2 = mesh.Surf_Cyl_Mesh()
        self.mesh_surf_obj2.num_axial = self.axial_elem
        self.mesh_surf_obj2.num_azimuthal = self.azimuthal_elem
        self.mesh_surf_obj2.L = self.L  #
        self.mesh_surf_obj2.D = self.D_Casing
        self.mesh_surf_obj2.initialize()

        # self.mesh_vol_obj2 = mesh.Vol_Cyl_Mesh()
        # self.mesh_vol_obj2.num_axial = self.axial_elem
        # self.mesh_vol_obj2.num_azimuthal = self.azimuthal_elem
        # self.mesh_vol_obj2.num_radial = self.radial_elem_hotgas
        # self.mesh_vol_obj2.L = self.L
        # self.mesh_vol_obj2.D = self.D_Casing
        # self.mesh_vol_obj2.D0 = self.D
        # self.mesh_vol_obj2.initialize()
        # self.mesh_vol_obj2.kappa = np.zeros(self.mesh_vol_obj2.N_vol)

        # Combustor Casing wall mesh (Pressure Vessel)
        self.mesh_wall_casing = mesh.Surf_Cyl_Wall_Mesh()
        self.mesh_wall_casing.num_axial = self.axial_elem
        self.mesh_wall_casing.num_thickness = self.thickness_casing_elem
        self.mesh_wall_casing.L = self.L  #
        self.mesh_wall_casing.D = self.D_Casing
        self.mesh_wall_casing.initialize()

        # Combustor Core
        self.core_obj = CombustorCore.CombustorCore(self.axial_elem, self.L, self.D, self.m_cooling)
        self.core_obj.hole_axial_loc = self.hole_axial_loc
        self.core_obj.m_inj_frac = self.m_inj_frac
        self.core_obj.chem_mech = self.chem_mech
        self.core_obj.m_inj_calc()

        # Heat Transfer objects

        # Convection in annulus
        self.Conv_HT_obj = FluidAnnulusHeatTransfer.FluidHeatTransfer()
        self.Conv_HT_obj.T_in = float(self.T_in)
        self.Conv_HT_obj.P_in = float(self.P_in)
        self.Conv_HT_obj.X_dict_cooling = X_dict_cooling
        self.Conv_HT_obj.L = self.L
        self.Conv_HT_obj.m_in_annulus = self.m_cooling  # kg/s
        self.Conv_HT_obj.num_axial = self.axial_elem
        self.Conv_HT_obj.m_inj_frac = np.zeros(len(self.Conv_HT_obj.hole_axial_loc))  # self.m_inj_frac

        self.Wall_HT_obj = WallHeatTransfer.WallHeatTransfer()

    def solver(self):
        # Initialisation values
        T_liner = 1500  # K
        T_casing = 1500  # K
        T_surf = self.wall_temperature(self.mesh_surf_obj, T_liner)
        T_surf3 = np.array(T_surf)
        T_w2 = np.reshape(T_surf, (self.mesh_surf_obj.num_azimuthal, self.mesh_surf_obj.num_axial), order='F')
        T_w2 = T_w2[3,
               :]  # np.ones((mesh_wall_liner.num_thickness,1)) @ np.reshape(T_w2[3,:], (1,mesh_wall_liner.num_axial))

        T_surf2 = self.wall_temperature(self.mesh_surf_obj2, T_casing)
        T_w1 = T_casing * np.ones(self.mesh_wall_casing.num_axial)
        T_liner_th = T_liner * np.ones((self.mesh_wall_liner.num_thickness, self.mesh_wall_liner.num_axial))
        T_casing_th = T_casing * np.ones((self.mesh_wall_casing.num_thickness, self.mesh_wall_casing.num_axial))

        T_vol = self.vol_temperature(self.mesh_vol_obj)
        T_g = np.reshape(T_vol,
                         (self.mesh_vol_obj.num_radial, self.mesh_vol_obj.num_azimuthal, self.mesh_vol_obj.num_axial),
                         order='F')
        T_g = T_g[self.mesh_vol_obj.num_radial - 1, 2, :]

        T_annulus = self.T_in * np.ones(self.mesh_wall_liner.num_axial)
        P_annulus = self.P_in * np.ones(self.mesh_wall_liner.num_axial)
        rho_annulus = P_annulus * (28.8 / 8314) / T_annulus
        T_in = self.T_in
        P_in = self.P_in
        # X_in = {'H2': 0.2, 'O2': 0.21, 'N2': 0.79}
        # T_g,P_g,X_g = core_obj.crn(T_annulus,P_annulus,T_in,P_in,X_dict,mdot_comb_in)
        gas_comb = ct.Solution(self.chem_mech)

        # Radiation precalc
        RC_obj = RadiationCode.ZonalMethod()
        RC_obj2 = RadiationCode.ZonalMethod()
        WSGG = WSGG_model.WSGG_model()
        T_g, P_g, X_g = self.core_obj.crn(T_annulus, P_annulus, T_in, P_in, self.X_core, self.mdot_comb_in)
        h2o_ind = gas_comb.species_names.index('H2O')
        X_H2O = X_g[-1][h2o_ind]
        epsi, epsi_h2o, epsi_co2, kappa_dict, a_eff_dict = WSGG.emissivity(P_g[-1], X_H2O, X_co2=0, T=T_g[-1], L=0.1)
        # mesh_vol_obj.kappa = kappa[0]*np.ones(mesh_vol_obj.N_vol)
        DEF_obj = RadiationCode.DirectExchangeFactor(self.mesh_surf_obj2, self.mesh_surf_obj2, self.mesh_vol_obj, path_absorb='on')
        RC_obj.precalc_WSGG(kappa_dict, DEF_obj)
        # SiSj, SiGj, GiSj, GiGj = RC_obj.matrix_main(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj)


        alpha = 0.1
        for iter in range(100000):
            if self.verbose == 'y':
                print('Iteration=', iter)
            if iter % 100 == 0:
                T_g, P_g, X_g = self.core_obj.crn(T_annulus, P_annulus, self.T_in, self.P_in, self.X_core,
                                                  self.mdot_comb_in)
                # if iter>0:
                #     X_H2O = X_g[-1][h2o_ind]
                #     epsi, epsi_h2o, epsi_co2, kappa_dict = WSGG.emissivity(P_g[-1], X_H2O, X_co2=0, T=T_g[-1], L=0.1)
                #     RC_obj.precalc_WSGG(kappa_dict, DEF_obj)

            # Radiative heat flux
            if self.verbose == 'y':
                print('Radiation Solver Active')

            Q_s, Q_g, Q_si_wo_SiGj = RC_obj.solver_WSGG(kappa_dict, a_eff_dict, DEF_obj, T_surf2, T_surf2, T_vol)


            Q_s_plot = np.reshape(Q_s, (self.mesh_surf_obj2.num_azimuthal, self.mesh_surf_obj2.num_axial), order='F')
            area_surf = np.reshape(self.mesh_surf_obj2.Area,
                                   (self.mesh_surf_obj2.num_azimuthal, self.mesh_surf_obj2.num_axial),
                                   order='F')
            q_rad_inner = Q_s_plot[3, :] / area_surf[3, :]


            # Convective heat flux annulus
            if self.verbose == 'y':
                print('Annulus Convective Heat Flux Solver Active')
            h_conv_casing, h_conv_liner_outer, T_annulus, P_annulus, rho_annulus = self.Conv_HT_obj.solver(
                self.mesh_wall_casing,
                self.mesh_wall_liner,
                T_annulus,
                P_annulus, rho_annulus,
                T_w1, T_w2)
            # Convective heat transfer inner surface of liner
            if self.verbose == 'y':
                print('Liner Inner Convective Heat Flux Solver Active')
            x = self.mesh_wall_liner.z[[(i) * self.mesh_wall_liner.num_thickness for i in range(len(T_g))]]
            dz = self.mesh_wall_liner.d_y[[(i) * self.mesh_wall_liner.num_thickness for i in range(len(T_g))]]
            k_w_inner = self.mesh_wall_liner.k_mat[[(i) * self.mesh_wall_liner.num_thickness for i in range(len(T_g))]]
            x = x + x[1]
            h_conv1 = FilmCoolingLiner.convection_liner(self.mdot_comb, self.D, x, T_g, P_g, X_g, gas_comb)

            conv_coeff_P_bottom = -h_conv1
            conv_B_vect_bottom = -h_conv1*T_g

            # Wall Heat Transfer
            if self.verbose == 'y':
                print('Wall Heat Transfer Solver Active')
            # print(q_conv_liner)
            # print(q_rad_inner)
            if self.verbose == 'y':
                print("Tin=", self.T_in)
            # Liner
            Tw = self.Wall_HT_obj.solver(self.mesh_wall_liner, -q_rad_inner*0, -q_rad_inner*0, h_conv_liner_outer,
                                         T_annulus, T_g, self.T_in, conv_coeff_P_bottom, conv_B_vect_bottom)
            Tw_plot = np.reshape(Tw, (self.mesh_wall_liner.num_thickness, self.mesh_wall_liner.num_axial), order='F')
            Tw_plot = np.array(T_liner_th + (Tw_plot - T_liner_th) * alpha)
            # Casing
            h_conv_casing_outer = np.zeros(len(h_conv_liner_outer))
            conv_coeff_P_bottom_casing = -h_conv_casing
            conv_B_vect_bottom_casing = -h_conv_casing * T_annulus
            Tw_casing = self.Wall_HT_obj.solver(self.mesh_wall_casing, -q_rad_inner, -q_rad_inner * 0.0,
                                                h_conv_casing_outer, T_annulus, T_annulus,
                                                self.T_in, conv_coeff_P_bottom_casing, conv_B_vect_bottom_casing)
            Tw1_plot = np.reshape(Tw_casing, (self.mesh_wall_casing.num_thickness, self.mesh_wall_casing.num_axial),
                                  order='F')
            Tw1_plot = np.array(T_casing_th + (Tw1_plot - T_casing_th) * alpha)

            T_surf = np.ones((self.mesh_surf_obj.num_azimuthal, 1)) @ np.reshape(Tw_plot[0, :],
                                                                                 (1, self.mesh_wall_liner.num_axial))
            T_surf3 = np.ones((self.mesh_surf_obj.num_azimuthal, 1)) @ np.reshape(
                Tw_plot[self.mesh_wall_liner.num_thickness - 1, :], (1, self.mesh_wall_liner.num_axial))
            T_surf = np.ndarray.flatten(T_surf, order='F')
            T_surf3 = np.ndarray.flatten(T_surf3, order='F')
            T_w2_0 = np.array(T_w2)
            T_w2 = np.array(Tw_plot[self.mesh_wall_liner.num_thickness - 1, :])
            T_w1 = np.array(Tw1_plot[self.mesh_wall_casing.num_thickness - 1, :])
            T_surf2 = np.ones((self.mesh_surf_obj2.num_azimuthal, 1)) @ np.reshape(Tw1_plot[0, :],
                                                                                   (1, self.mesh_wall_casing.num_axial))
            T_surf2 = np.ndarray.flatten(T_surf2, order='F')
            epsi_Tw2 = np.max(np.abs(T_w2 - T_w2_0) / (T_w2_0))
            T_liner_th = np.array(Tw_plot)
            T_casing_th = np.array(Tw1_plot)
            # print(T_w2)
            if self.verbose == 'y':
                print('Epsi_tw2=', epsi_Tw2)
            if epsi_Tw2 < 1e-12:  # and neg_exist==0:
                break

        dict_save = {'Q_s': Q_s, 'Q_g': Q_g,
                     'Surf_Liner_wall': self.mesh_surf_obj,
                     'Surf_Casing_wall': self.mesh_surf_obj2, 'Liner_Gas_mesh': self.mesh_vol_obj,
                     'Annulus_Gas_mesh': self.mesh_vol_obj2, 'h_conv_liner_gas': h_conv1,
                     'h_conv_liner_coolant': h_conv_liner_outer,
                     'h_conv_casing_coolant': h_conv_casing,
                     'Mesh_wall_liner': self.mesh_wall_liner, 'T_wall_liner': T_liner_th, 'T_annulus': T_annulus,
                     'T_wall_casing': T_casing_th,'P_annulus': P_annulus,
                     'rho_annulus': rho_annulus, 'Annulus_flow_obj': self.Conv_HT_obj, 'Core_Temp': T_g, 'Core_Pr': P_g,
                     'Core_comp': X_g}

        return dict_save
        # caseid='axial'+str(self.axial_elem)+'_linerthelem'+str(self.thickness_liner_elem)+'_mdotcoolfrac'+str(Xi)+"_nofilmcooling"
        # with open(parentdir+'/data/' + 'dictsave_' + caseid + '.pkl', 'wb') as file:
        #     pickle.dump(dict_save, file, pickle.HIGHEST_PROTOCOL)