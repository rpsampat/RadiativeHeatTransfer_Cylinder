import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir+'/src/')
import mesh
import RadiationCode

class SimplCylinder:
    def __init__(self):
        a=0
        self.sigma = 5.67e-8

    def wall_temperature(self,mesh_surf):
        Twall = 800 # K
        T_surf = Twall*np.ones(mesh_surf.N_surf)

        return T_surf

    def vol_temperature(self, mesh_vol):
        T = 2000  # K
        T_vol = T * np.ones(mesh_vol.N_vol)

        return T_vol

    def main(self):
        L =0.3 #m
        D = 0.10 #m
        mesh_surf_obj = mesh.Surf_Cyl_Mesh()
        mesh_vol_obj = mesh.Vol_Cyl_Mesh()
        mesh_surf_obj.L = L#
        mesh_vol_obj.L = L
        mesh_surf_obj.D = D
        mesh_vol_obj.D = D
        mesh_surf_obj.cyl_coords()
        mesh_vol_obj.cyl_coords()

        epsilon_surf = 0.9 * np.ones(mesh_surf_obj.N_surf)
        kappa = mesh_vol_obj.kappa

        RC_obj = RadiationCode.ZonalMethod()
        SiSj, SiGj, GiSj, GiGj = RC_obj.main(mesh_surf_obj,mesh_vol_obj)

        T_surf = self.wall_temperature(mesh_surf_obj)
        T_vol = self.vol_temperature(mesh_vol_obj)

        Eb_s = self.sigma*(T_surf**4)
        Eb_g = self.sigma*(T_vol**4)

        Q_s = (epsilon_surf*mesh_surf_obj.Area*Eb_s) - (SiSj @ Eb_s) - (SiGj @ Eb_g)
        Q_g = (4*kappa*mesh_vol_obj.Volume*Eb_g) - (GiSj @ Eb_s) - (GiGj @ Eb_g)

        fig,ax = plt.subplots(ncols =1, nrows=2)
        x_s_plot = np.reshape(mesh_surf_obj.x, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
        Q_s_plot = np.reshape(Q_s,(mesh_surf_obj.num_azimuthal,mesh_surf_obj.num_axial),order='F')
        img = ax[0].imshow(Q_s_plot,cmap='jet')
        img1 = ax[1].imshow(x_s_plot,cmap='jet')
        fig.colorbar(img)
        fig.colorbar(img1)

        fig1, ax1 = plt.subplots(ncols =1, nrows=2)
        Q_g_plot = np.reshape(Q_g, (mesh_vol_obj.num_radial,mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial),order='F')
        img1 = ax1[0].imshow(Q_g_plot[:,:,2], cmap='jet')
        img2 = ax1[1].imshow(Q_g_plot[:, 9, :], cmap='jet')
        fig1.colorbar(img1)
        fig1.colorbar(img2)

        # fig2, ax2 = plt.subplots(ncols=1, nrows=2)
        # x_g_plot = np.reshape(mesh_vol_obj.x, (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial),
        #                       order='F')
        # img1 = ax2[0].imshow(x_g_plot[:, :, 1], cmap='jet')
        # img2 = ax2[1].imshow(x_g_plot[:, 0, :], cmap='jet')
        # fig2.colorbar(img1)
        # fig2.colorbar(img2)

        plt.show()



if __name__=='__main__':
    obj = SimplCylinder()
    obj.main()