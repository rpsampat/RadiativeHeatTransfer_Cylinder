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
import effusion_cooling_holes as ECH


class HeatTransfer:
    def __init__(self, L, D, D_Casing, liner_wall_thickness,casing_wall_thickness, Pin, Tin, Uf, beta, EAR, Power, BR):
        self.L = L
        self.D = D
        self.D_Casing = D_Casing
        self.liner_wall_thickness = liner_wall_thickness
        self.casing_wall_thickness = casing_wall_thickness
        # Combustor inlet state
        self.P_in = Pin
        self.T_in = Tin
        self.U_f = Uf
        self.Beta = beta
        self.EAR = EAR
        self.Power = Power

        # Liner cooling
        self.BR = BR

        # Properties
        self.LHV_H2_mass = 120  # MJ/kg
        self.LHV_H2_mole = self.LHV_H2_mass * 2  # MJ/kmol
        self.sigma = 5.67e-8
        self.chem_mech = 'H2_11_20.yaml'
        self.verbose= 'n'#y/n

        # Mesh params
        self.axial_elem = 10  # 30
        self.azimuthal_elem = 4  # 10
        self.radial_elem_hotgas = 6  # 20
        self.thickness_liner_elem = 5
        self.thickness_casing_elem = 5

    def wall_temperature(self,mesh_surf,Twall):
        #Twall = 1500 # K
        T_surf = Twall*np.ones(mesh_surf.N_surf)

        return T_surf

    def vol_temperature(self, mesh_vol):
        Tmax = 2000  # K
        z0 = max(mesh_vol.z)/3
        Tmin = 1100
        alpha = 100.0
        T_vol = Tmax*np.ones(mesh_vol.N_vol)
        for i in range(len(mesh_vol.z)):
            z = mesh_vol.z[i]
            T_vol[i] = Tmin+(Tmax-Tmin)/((1+(math.e**(-alpha*(z-z0)))))

        return T_vol

    def inlet_composition_v2(self,U_f, EAR, Beta):
        """

        :param U_f: fuel utilization in SOFC
        :param EAR: Excess air ratio (lambda) for overall SOFC+GT
        :param Beta: hydrogen power fraction given directly to combustor Beta=n_h2_direct*LHV_H2/Power
        :return: Dictionary of net composition entering the combustor
        """
        delta = 0.5*((EAR/0.21)-((1-Beta)*U_f)+2) # denominator factor for total moles
        X_h2 = (1-U_f+(Beta*U_f))/delta
        X_h2o = (1-Beta)*U_f/delta
        X_o2 = (0.5*(EAR-((1-Beta)*U_f)))/delta
        X_n2 = (0.5*EAR*0.79/0.21)/delta
        X_dict={'O2':X_o2,'N2':X_n2,'H2':X_h2,'H2O':X_h2o}

        return X_dict

    def initialize(self):


        ndot_H2_in_tot = self.Power / (self.LHV_H2_mole * 1e6)  # kmol/s

        ndot_H2_anode = (1 - self.Beta) * ndot_H2_in_tot
        ndot_cathode_off = (ndot_H2_anode / 2) * ((self.EAR / 0.21) - self.U_f)
        ndot_H2_comb_in = ndot_H2_in_tot * (1 - self.U_f + (self.Beta * self.U_f))
        X_dict = self.inlet_composition_v2(self.U_f, self.EAR, self.Beta)
        X_dict_cooling = {'O2': X_dict['O2'], 'N2': X_dict['N2']}
        gas_cooling = ct.Solution('air.yaml')
        gas_cooling.TPX = self.T_in, ct.one_atm, X_dict_cooling

        # Effusion cooling
        self.ECH_obj = ECH.effusion_cooling()
        self.ECH_obj.BR = self.BR
        Xi, m_inj_frac, inj_loc, num_holes_axial, hole_dia_array = self.ECH_obj.hole_def(L_start=0, L_end=self.L, L_liner=self.L,
                                                                                    dia_cyl=self.D)
        m_cooling = ndot_cathode_off * Xi * gas_cooling.mean_molecular_weight  # kg/s
        self.hole_axial_loc = inj_loc  # distance from burner head as a fraction of liner length
        # mass flow as a fraction of cooling flow at the inlet of the annulus near the burner head.
        # Injection flows correspond to location specified in hole_axial_loc
        self.m_inj_frac = m_inj_frac  # np.array([0.1, 0.20, 0.1])  #np.array([0.1, 0.20, 0.1])  #np.array([0.0, 0.0, 0.0])  #np.array([0.1, 0.20, 0.1])  #
        print('Xi=', Xi)
        
        # Core composition and mass flow
        self.X_core = dict(X_dict)
        self.X_core['O2'] = self.X_core['O2'] * (1 - Xi)
        self.X_core['N2'] = self.X_core['N2'] * (1 - Xi)
        MW_core = (self.X_core['O2'] * 32) + (self.X_core['N2'] * 28) + (self.X_core['H2'] * 2) + (self.X_core['H2O'] * 18)
        self.mdot_comb_in = MW_core * ndot_H2_comb_in / self.X_core['H2']

        # Combustor liner interior mesh
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

        # Combustor Liner wall mesh
        self.mesh_wall_liner = mesh.Surf_Cyl_Wall_Mesh()
        self.mesh_wall_liner.num_axial = self.axial_elem
        self.mesh_wall_liner.num_thickness = self.thickness_liner_elem
        self.mesh_wall_liner.thickness = self.liner_wall_thickness
        self.mesh_wall_liner.L = self.L  #
        self.mesh_wall_liner.D = self.D
        self.mesh_wall_liner.initialize()
        self.mdot_comb = self.mdot_comb_in * np.ones(self.axial_elem)

        # Combustor Casing interior mesh
        self.mesh_surf_obj2 = mesh.Surf_Cyl_Mesh()
        self.mesh_surf_obj2.num_axial = self.axial_elem
        self.mesh_surf_obj2.num_azimuthal = self.azimuthal_elem
        self.mesh_surf_obj2.L = self.L  #
        self.mesh_surf_obj2.D = self.D_Casing
        self.mesh_surf_obj2.initialize()

        self.mesh_vol_obj2 = mesh.Vol_Cyl_Mesh()
        self.mesh_vol_obj2.num_axial = self.axial_elem
        self.mesh_vol_obj2.num_azimuthal = self.azimuthal_elem
        self.mesh_vol_obj2.num_radial = self.radial_elem_hotgas
        self.mesh_vol_obj2.L = self.L
        self.mesh_vol_obj2.D = self.D_Casing
        self.mesh_vol_obj2.D0 = self.D
        self.mesh_vol_obj2.initialize()
        self.mesh_vol_obj2.kappa = np.zeros(self.mesh_vol_obj2.N_vol)

        # Combustor Casing wall mesh
        self.mesh_wall_casing = mesh.Surf_Cyl_Wall_Mesh()
        self.mesh_wall_casing.num_axial = self.axial_elem
        self.mesh_wall_casing.num_thickness = self.thickness_casing_elem
        self.mesh_wall_casing.L = self.L  #
        self.mesh_wall_casing.D = self.D_Casing
        self.mesh_wall_casing.initialize()

        # Combustor Core
        self.core_obj = CombustorCore.CombustorCore(self.axial_elem, self.L, self.D, m_cooling)
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
        self.Conv_HT_obj.m_in_annulus = m_cooling  # kg/s
        self.Conv_HT_obj.num_axial = self.axial_elem
        self.Conv_HT_obj.hole_axial_loc = inj_loc  # self.hole_axial_loc
        self.Conv_HT_obj.m_inj_frac = m_inj_frac  # self.m_inj_frac
        self.Conv_HT_obj.num_holes_axial = num_holes_axial
        self.Conv_HT_obj.hole_dia = hole_dia_array

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
        DEF_obj = RadiationCode.DirectExchangeFactor(self.mesh_surf_obj, self.mesh_surf_obj, self.mesh_vol_obj)
        RC_obj.precalc_WSGG(kappa_dict, DEF_obj)
        # SiSj, SiGj, GiSj, GiGj = RC_obj.matrix_main(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj)
        DEF_obj2 = RadiationCode.DirectExchangeFactor(self.mesh_surf_obj, self.mesh_surf_obj2, self.mesh_vol_obj)
        DEF_obj2.eval()
        SiSj2, SiGj2, GiSj2, GiGj2 = RC_obj2.matrix_main(self.mesh_surf_obj, self.mesh_surf_obj2,
                                                              self.mesh_vol_obj2)

        alpha = 0.1
        for iter in range(100000):
            if self.verbose=='y':
                print('Iteration=',iter)
            if iter%100==0:

                T_g, P_g,X_g = self.core_obj.crn(T_annulus, P_annulus, self.T_in, self.P_in, self.X_core, self.mdot_comb_in)
                # if iter>0:
                #     X_H2O = X_g[-1][h2o_ind]
                #     epsi, epsi_h2o, epsi_co2, kappa_dict = WSGG.emissivity(P_g[-1], X_H2O, X_co2=0, T=T_g[-1], L=0.1)
                #     RC_obj.precalc_WSGG(kappa_dict, DEF_obj)


            # Radiative heat flux
            if self.verbose == 'y':
                print('Radiation Solver Active')

            Q_s, Q_g, Q_si_wo_SiGj = RC_obj.solver_WSGG(kappa_dict, a_eff_dict, DEF_obj, T_surf, T_surf, T_vol)

            # Q_s, Q_g, Q_si_wo_SiGj = RC_obj.solver(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj, T_surf, T_surf, T_vol,
            #                                        SiSj,
            #                                        SiGj, GiSj, GiGj)
            Q_s2, Q_g2, Q_si_wo_SiGj2 = RC_obj.solver(self.mesh_surf_obj, self.mesh_surf_obj2, self.mesh_vol_obj2, T_surf3, T_surf2,
                                                      T_vol,
                                                      SiSj2, SiGj2, GiSj2, GiGj2)
            Q_s2_surf2, Q_g2_surf2, Q_si_wo_SiGj2_surf2 = RC_obj.solver_surf2(self.mesh_surf_obj, self.mesh_surf_obj2, self.mesh_vol_obj2, T_surf3, T_surf2,
                                                      T_vol,
                                                      SiSj2, SiGj2, GiSj2, GiGj2)
            Q_s_plot = np.reshape(Q_s, (self.mesh_surf_obj.num_azimuthal, self.mesh_surf_obj.num_axial), order='F')
            area_surf = np.reshape(self.mesh_surf_obj.Area, (self.mesh_surf_obj.num_azimuthal, self.mesh_surf_obj.num_axial),
                                   order='F')
            q_rad_inner = Q_s_plot[3, :] / area_surf[3, :]

            Q_s_plot2 = np.reshape(Q_si_wo_SiGj2, (self.mesh_surf_obj.num_azimuthal, self.mesh_surf_obj.num_axial), order='F')
            Q_s_plot2_surf2 = np.reshape(Q_si_wo_SiGj2_surf2, (self.mesh_surf_obj2.num_azimuthal, self.mesh_surf_obj2.num_axial), order='F')
            area_surf2 = np.reshape(self.mesh_surf_obj2.Area, (self.mesh_surf_obj2.num_azimuthal, self.mesh_surf_obj2.num_axial),
                                    order='F')
            q_rad_outer = Q_s_plot2[3, :] / area_surf[3, :]
            # Need to find heat flux on casing
            q_rad_surf2_inner = Q_s_plot2_surf2[3, :] / area_surf2[3, :]

            # Convective heat flux annulus
            if self.verbose == 'y':
                print('Annulus Convective Heat Flux Solver Active')
            h_conv_casing, h_conv_liner_outer, T_annulus, P_annulus, rho_annulus = self.Conv_HT_obj.solver(self.mesh_wall_casing,
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
            x =x +x[1]
            h_conv1 = FilmCoolingLiner.convection_liner(self.mdot_comb,self.D,x,T_g,P_g,X_g,gas_comb)
            eta = 0.73 * np.ones(len(T_annulus))
            Tw_in = T_g - eta * (T_g-T_annulus)
            # h_conv1 = h_conv1[5]*np.ones(len(x))
            conv_coeff_P_bottom = -2*11.4/(dz)
            conv_B_vect_bottom = -2*11.4 * Tw_in/(dz)
            # conv_coeff_P_bottom = -h_conv1
            # conv_B_vect_bottom = -h_conv1*T_g

            # Wall Heat Transfer
            if self.verbose == 'y':
                print('Wall Heat Transfer Solver Active')
            # print(q_conv_liner)
            # print(q_rad_inner)
            if self.verbose == 'y':
                print("Tin=",self.T_in)
            # Liner
            Tw = self.Wall_HT_obj.solver(self.mesh_wall_liner,-q_rad_inner,-q_rad_outer, h_conv_liner_outer, T_annulus, T_g, self.T_in,conv_coeff_P_bottom,conv_B_vect_bottom)
            Tw_plot = np.reshape(Tw, (self.mesh_wall_liner.num_thickness, self.mesh_wall_liner.num_axial), order='F')
            Tw_plot = np.array(T_liner_th + (Tw_plot - T_liner_th) * alpha)
            # Casing
            h_conv_casing_outer = np.zeros(len(h_conv_liner_outer))
            conv_coeff_P_bottom_casing = -h_conv1
            conv_B_vect_bottom_casing = -h_conv1*T_annulus
            Tw_casing = self.Wall_HT_obj.solver(self.mesh_wall_casing, -q_rad_surf2_inner, -q_rad_outer*0.0, h_conv_casing_outer, T_annulus, T_annulus,
                                    self.T_in, conv_coeff_P_bottom_casing, conv_B_vect_bottom_casing)
            Tw1_plot = np.reshape(Tw_casing, (self.mesh_wall_casing.num_thickness, self.mesh_wall_casing.num_axial), order='F')
            Tw1_plot = np.array(T_casing_th + (Tw1_plot - T_casing_th) * alpha)




            T_surf = np.ones((self.mesh_surf_obj.num_azimuthal,1)) @ np.reshape(Tw_plot[0,:], (1,self.mesh_wall_liner.num_axial))
            T_surf3 = np.ones((self.mesh_surf_obj.num_azimuthal,1)) @ np.reshape(Tw_plot[self.mesh_wall_liner.num_thickness-1,:], (1,self.mesh_wall_liner.num_axial))
            T_surf = np.ndarray.flatten(T_surf,order='F')
            T_surf3 = np.ndarray.flatten(T_surf3, order='F')
            T_w2_0 = np.array(T_w2)
            T_w2 = np.array(Tw_plot[self.mesh_wall_liner.num_thickness-1,:])
            T_w1 = np.array(Tw1_plot[self.mesh_wall_casing.num_thickness-1,:])
            T_surf2 = np.ones((self.mesh_surf_obj2.num_azimuthal, 1)) @ np.reshape(Tw1_plot[0, :],
                                                                            (1, self.mesh_wall_casing.num_axial))
            T_surf2 = np.ndarray.flatten(T_surf2, order='F')
            epsi_Tw2 = np.max(np.abs(T_w2-T_w2_0)/(T_w2_0))
            T_liner_th = np.array(Tw_plot)
            T_casing_th = np.array(Tw1_plot)
            # print(T_w2)
            if self.verbose == 'y':
                print('Epsi_tw2=',epsi_Tw2)
            if epsi_Tw2<1e-12:# and neg_exist==0:
                break

        dict_save={'Q_s':Q_s,'Q_g':Q_g,'Q_s2':Q_s2,'Q_g2':Q_g2,'Q_si_wo_SiGj2':Q_si_wo_SiGj2,'Surf_Liner_wall':self.mesh_surf_obj,
                   'Surf_Casing_wall':self.mesh_surf_obj2,'Liner_Gas_mesh':self.mesh_vol_obj,'Annulus_Gas_mesh':self.mesh_vol_obj2,'h_conv_liner_gas':h_conv1,'h_conv_liner_coolant':h_conv_liner_outer,
                   'Mesh_wall_liner':self.mesh_wall_liner,'T_wall_liner':T_liner_th,'T_annulus':T_annulus,'P_annulus':P_annulus,
                   'rho_annulus':rho_annulus,'Annulus_flow_obj':self.Conv_HT_obj,'Core_Temp':T_g,'Core_Pr':P_g,'Core_comp':X_g}

        return dict_save
        # caseid='axial'+str(self.axial_elem)+'_linerthelem'+str(self.thickness_liner_elem)+'_mdotcoolfrac'+str(Xi)+"_nofilmcooling"
        # with open(parentdir+'/data/' + 'dictsave_' + caseid + '.pkl', 'wb') as file:
        #     pickle.dump(dict_save, file, pickle.HIGHEST_PROTOCOL)