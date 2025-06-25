import os
import sys
import inspect
import numpy as np
import math
import matplotlib.pyplot as plt

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

    def wall_temperature(self,mesh_surf):
        Twall = 800 # K
        T_surf = Twall*np.ones(mesh_surf.N_surf)

        return T_surf

    def vol_temperature(self, mesh_vol):
        Tmax = 2000  # K
        z0 = max(mesh_vol.z)/3
        Tmin = 1100
        alpha = 100.0
        T_vol = 2000*np.ones(mesh_vol.N_vol)
        # for i in range(len(mesh_vol.z)):
        #     z = mesh_vol.z[i]
        #     T_vol[i] = Tmin+(Tmax-Tmin)/((1+(math.e**(-alpha*(z-z0)))))

        return T_vol

    def main(self):
        L =0.3 #m
        D = 0.10 #m
        axial_elem = 30
        azimuthal_elem = 10
        radial_elem_hotgas = 10
        thickness_liner_elem = 10
        thickness_casing_elem = 10
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
        mesh_wall_liner.L = L  #
        mesh_wall_liner.D = D
        mesh_wall_liner.initialize()

        # Combustor Liner wall mesh
        mesh_wall_casing = mesh.Surf_Cyl_Wall_Mesh()
        mesh_wall_casing.num_axial = axial_elem
        mesh_wall_casing.num_thickness = thickness_casing_elem
        mesh_wall_casing.L = L  #
        mesh_wall_casing.D = D + 0.050
        mesh_wall_casing.initialize()

        T_surf = self.wall_temperature(mesh_surf_obj)
        T_vol = self.vol_temperature(mesh_vol_obj)
        T_g =np.reshape(T_vol, (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial), order='F')
        T_g = T_g[mesh_vol_obj.num_radial-1, 2,:]

        #Initialisation values
        self.T_in = 1100
        self.P_in = 10e5
        T_annulus = self.T_in * np.ones(mesh_wall_liner.num_axial)
        P_annulus = self.P_in * np.ones(mesh_wall_liner.num_axial)
        rho_annulus = P_annulus * (28.8 / 8314) / T_annulus
        T_w1 = 700 * np.ones(mesh_wall_casing.num_axial)
        T_w2 = np.reshape(T_surf, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
        T_w2 = T_w2[3,:]#np.ones((mesh_wall_liner.num_thickness,1)) @ np.reshape(T_w2[3,:], (1,mesh_wall_liner.num_axial))

        # Heat Transfer objects
        RC_obj = RadiationCode.ZonalMethod()

        Conv_HT_obj = FluidAnnulusHeatTransfer.FluidHeatTransfer()
        Conv_HT_obj.T_in = self.T_in
        Conv_HT_obj.P_in = self.P_in

        Wall_HT_obj = WallHeatTransfer.WallHeatTransfer()

        for iter in range(5):
            print('Iteration=',iter)
            # Radiative heat flux
            print('Radiation Solver Active')
            Q_s,Q_g = RC_obj.solver(mesh_surf_obj,mesh_vol_obj,T_surf,T_vol)
            Q_s_plot = np.reshape(Q_s, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
            area_surf = np.reshape(mesh_surf_obj.Area, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
            q_rad_inner = Q_s_plot[3,:]/area_surf[3,:]
            q_rad_outer = q_rad_inner*0


            # Convective heat flux annulus
            print('Convective Heat Flux Solver Active')
            q_conv_casing, q_conv_liner, T_annulus,P_annulus, rho_annulus = Conv_HT_obj.solver(mesh_wall_casing, mesh_wall_liner, T_annulus, P_annulus, rho_annulus, T_w1, T_w2)
            q_conv1 = q_conv_liner*0

            # Wall Heat Transfer
            print('Wall Heat Transfer Solver Active')
            Tw = Wall_HT_obj.solver(mesh_wall_liner,q_rad_inner,q_rad_outer,-q_conv1,-q_conv_liner, T_annulus, T_g)
            Tw_plot = np.reshape(Tw, (mesh_wall_liner.num_thickness, mesh_wall_liner.num_axial), order='F')



            T_surf = np.ones((mesh_surf_obj.num_azimuthal,1)) @ np.reshape(Tw_plot[0,:], (1,mesh_wall_liner.num_axial))
            T_surf = np.ndarray.flatten(T_surf,order='F')
            T_w2 = Tw_plot[mesh_wall_liner.num_thickness-1,:]

        fig, ax = plt.subplots()
        ax.plot(q_rad_inner)

        fig, ax = plt.subplots()
        ax.plot(q_conv_liner)

        fig, ax = plt.subplots()
        ax.plot(T_annulus)
        ax.set_ylabel('T fluid annulus (K)')

        fig, ax = plt.subplots()
        img = ax.imshow(Tw_plot)
        ax.invert_yaxis()
        fig.colorbar(img)
        plt.show()





if __name__=='__main__':
    CHT_obj = CombustorHeatTransfer()
    CHT_obj.main()

