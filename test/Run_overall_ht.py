import os
import sys
import inspect
import numpy as np
import math
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir+'/src/')
import FilmCoolingLiner

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir+'/src/')
import mesh
import RadiationCode
import WallHeatTransfer
import FluidAnnulusHeatTransfer

class CombustorHeatTransfer:
    def __init__(self):
        a=0
        self.sigma = 5.67e-8

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

    def main(self):
        L =0.3 #m
        D = 0.10 #m
        D_Casing = 0.1+0.05
        axial_elem = 30
        azimuthal_elem = 10
        radial_elem_hotgas = 10
        thickness_liner_elem = 5
        thickness_casing_elem = 5
        mdot_comb_in = 0.5 # kg/s
        m_cooling = 0.02 # kg/s
        T_liner = 800  # K
        T_casing = 1500  # K
        self.T_in = 1100 # K , used to initialize values for annulus flow
        self.P_in = 10e5 # K , used to initialize values for annulus flow
        self.hole_axial_loc = np.array([0.2, 0.5, 0.75])  # distance from burner head as a fraction of liner length
        # mass flow as a fraction of cooling flow at the inlet of the annulus near the burner head.
        # Injection flows correspond to location specified in hole_axial_loc
        self.m_inj_frac = np.array([0.1, 0.20, 0.1])  # np.array([0.0, 0.0, 0.0])  #
        # Combustor liner interior mesh
        mesh_surf_obj = mesh.Surf_Cyl_Mesh()
        mesh_surf_obj.num_axial = axial_elem
        mesh_surf_obj.num_azimuthal = azimuthal_elem
        mesh_surf_obj.L = L  #
        mesh_surf_obj.D = D
        mesh_surf_obj.initialize()

        mesh_vol_obj = mesh.Vol_Cyl_Mesh()
        mesh_vol_obj.num_axial = axial_elem
        mesh_vol_obj.num_azimuthal = azimuthal_elem
        mesh_vol_obj.num_radial = radial_elem_hotgas
        mesh_vol_obj.L = L
        mesh_vol_obj.D = D
        mesh_vol_obj.initialize()

        # Combustor Liner wall mesh
        mesh_wall_liner = mesh.Surf_Cyl_Wall_Mesh()
        mesh_wall_liner.num_axial = axial_elem
        mesh_wall_liner.num_thickness = thickness_liner_elem
        mesh_wall_liner.thickness = 0.003
        mesh_wall_liner.L = L  #
        mesh_wall_liner.D = D
        mesh_wall_liner.initialize()
        mdot_comb = mdot_comb_in*np.ones(axial_elem)

        # Combustor Casing interior mesh
        mesh_surf_obj2 = mesh.Surf_Cyl_Mesh()
        mesh_surf_obj2.num_axial = axial_elem
        mesh_surf_obj2.num_azimuthal = azimuthal_elem
        mesh_surf_obj2.L = L  #
        mesh_surf_obj2.D = D_Casing
        mesh_surf_obj2.initialize()

        # Combustor Casing wall mesh
        mesh_wall_casing = mesh.Surf_Cyl_Wall_Mesh()
        mesh_wall_casing.num_axial = axial_elem
        mesh_wall_casing.num_thickness = thickness_casing_elem
        mesh_wall_casing.L = L  #
        mesh_wall_casing.D = D_Casing
        mesh_wall_casing.initialize()


        T_surf = self.wall_temperature(mesh_surf_obj,T_liner)
        T_surf3 = np.array(T_surf)
        T_w2 = np.reshape(T_surf, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
        T_w2 = T_w2[3,
               :]  # np.ones((mesh_wall_liner.num_thickness,1)) @ np.reshape(T_w2[3,:], (1,mesh_wall_liner.num_axial))

        T_surf2 = self.wall_temperature(mesh_surf_obj2, T_casing)
        T_w1 = T_casing * np.ones(mesh_wall_casing.num_axial)
        T_liner_th = T_liner*np.ones((mesh_wall_liner.num_thickness,mesh_wall_liner.num_axial))

        T_vol = self.vol_temperature(mesh_vol_obj)
        T_g =np.reshape(T_vol, (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial), order='F')
        T_g = T_g[mesh_vol_obj.num_radial-1, 2,:]

        #Initialisation values
        T_annulus = self.T_in * np.ones(mesh_wall_liner.num_axial)
        P_annulus = self.P_in * np.ones(mesh_wall_liner.num_axial)
        rho_annulus = P_annulus * (28.8 / 8314) / T_annulus



        # Heat Transfer objects
        RC_obj = RadiationCode.ZonalMethod()
        RC_obj2 = RadiationCode.ZonalMethod()

        Conv_HT_obj = FluidAnnulusHeatTransfer.FluidHeatTransfer()
        Conv_HT_obj.T_in = float(self.T_in)
        Conv_HT_obj.P_in = float(self.P_in)
        Conv_HT_obj.L = L
        Conv_HT_obj.m_in_annulus = m_cooling # kg/s
        Conv_HT_obj.num_axial = axial_elem
        Conv_HT_obj.hole_axial_loc = self.hole_axial_loc
        Conv_HT_obj.m_inj_frac = self.m_inj_frac

        Wall_HT_obj = WallHeatTransfer.WallHeatTransfer()

        alpha = 0.1
        SiSj, SiGj, GiSj, GiGj = RC_obj.matrix_main(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj)
        SiSj2, SiGj2, GiSj2, GiGj2 = RC_obj2.matrix_main(mesh_surf_obj, mesh_surf_obj2, mesh_vol_obj)





        for iter in range(100000):
            print('Iteration=',iter)

            # Radiative heat flux
            print('Radiation Solver Active')

            Q_s, Q_g, Q_si_wo_SiGj = RC_obj.solver(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj, T_surf, T_surf, T_vol,
                                                   SiSj,
                                                   SiGj, GiSj, GiGj)
            Q_s2, Q_g2, Q_si_wo_SiGj2 = RC_obj.solver(mesh_surf_obj, mesh_surf_obj2, mesh_vol_obj, T_surf3, T_surf2,
                                                      T_vol,
                                                      SiSj2, SiGj2, GiSj2, GiGj2)
            Q_s_plot = np.reshape(Q_s, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
            area_surf = np.reshape(mesh_surf_obj.Area, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial),
                                   order='F')
            q_rad_inner = Q_s_plot[3, :] / area_surf[3, :]

            Q_s_plot2 = np.reshape(Q_si_wo_SiGj2, (mesh_surf_obj2.num_azimuthal, mesh_surf_obj2.num_axial), order='F')
            area_surf2 = np.reshape(mesh_surf_obj2.Area, (mesh_surf_obj2.num_azimuthal, mesh_surf_obj2.num_axial),
                                    order='F')
            q_rad_outer = Q_s_plot2[3, :] / area_surf2[3, :]

            # Convective heat flux annulus
            print('Convective Heat Flux Solver Active')
            h_conv_casing, h_conv_liner, T_annulus, P_annulus, rho_annulus = Conv_HT_obj.solver(mesh_wall_casing,
                                                                                                mesh_wall_liner,
                                                                                                T_annulus,
                                                                                                P_annulus, rho_annulus,
                                                                                                T_w1, T_w2)
            # Convective heat transfer inner surface of liner
            mu = 3.26*1e-5
            Pr = 0.7
            dia = D
            k = 0.026
            x = mesh_wall_liner.z[[(i) * mesh_wall_liner.num_thickness for i in range(len(T_g))]]
            x =x +x[1]
            Nu_x = FilmCoolingLiner.convection_liner(mdot_comb,mu,Pr, dia,x)
            h_conv1 = (Nu_x*k/dia)
            # h_conv1 = h_conv1[5]*np.ones(len(x))

            # Wall Heat Transfer
            print('Wall Heat Transfer Solver Active')
            # print(q_conv_liner)
            # print(q_rad_inner)
            print("Tin=",self.T_in)
            Tw = Wall_HT_obj.solver(mesh_wall_liner,-q_rad_inner,-q_rad_outer,h_conv1, h_conv_liner, T_annulus, T_g, self.T_in,T_w2)
            # fig, ax = plt.subplots()
            # ax.plot(Tw)
            # plt.show()
            # ind_negative = np.where(Tw<0)[0]
            # if len(ind_negative)>0:
            #     neg_exist= 1
            # else:
            #     neg_exist = 0
            # Tw[ind_negative] = float(self.T_in) # Handling negative temperatures during iterations
            Tw_plot = np.reshape(Tw, (mesh_wall_liner.num_thickness, mesh_wall_liner.num_axial), order='F')
            Tw_plot = np.array(T_liner_th+(Tw_plot-T_liner_th)*alpha)




            T_surf = np.ones((mesh_surf_obj.num_azimuthal,1)) @ np.reshape(Tw_plot[0,:], (1,mesh_wall_liner.num_axial))
            T_surf3 = np.ones((mesh_surf_obj.num_azimuthal,1)) @ np.reshape(Tw_plot[mesh_wall_liner.num_thickness-1,:], (1,mesh_wall_liner.num_axial))
            T_surf = np.ndarray.flatten(T_surf,order='F')
            T_surf3 = np.ndarray.flatten(T_surf3, order='F')
            T_w2_0 = np.array(T_w2)
            T_w2 = np.array(Tw_plot[mesh_wall_liner.num_thickness-1,:])
            epsi_Tw2 = np.max(np.abs(T_w2-T_w2_0)/(T_w2_0))
            T_liner_th = np.array(Tw_plot)
            # print(T_w2)
            print('Epsi_tw2=',epsi_Tw2)
            if epsi_Tw2<1e-12:# and neg_exist==0:
                break

        fig, ax = plt.subplots()
        ax.plot(q_rad_inner)

        fig, ax = plt.subplots()
        ax.plot(q_rad_outer)

        fig, ax = plt.subplots()
        ax.plot(h_conv_liner)

        fig, ax = plt.subplots()
        ax.plot(T_g)
        ax.set_ylabel('T gas (K)')


        fig, ax = plt.subplots()
        ax.plot(T_annulus)
        ax.set_ylabel('T fluid annulus (K)')

        fig, ax = plt.subplots()
        ax.plot(P_annulus)
        ax.set_ylabel('PT fluid annulus (Pa)')

        fig, ax = plt.subplots()
        ax.plot(rho_annulus)
        ax.set_ylabel('Density fluid annulus (kg/m3)')

        fig, ax = plt.subplots()
        img = ax.imshow(Tw_plot)
        ax.invert_yaxis()
        fig.colorbar(img)
        plt.show()





if __name__=='__main__':
    CHT_obj = CombustorHeatTransfer()
    CHT_obj.main()

