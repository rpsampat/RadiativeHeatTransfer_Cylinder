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

class AxialTempProfile_Cylinder:
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
        T_vol = np.ones(mesh_vol.N_vol)
        for i in range(len(mesh_vol.z)):
            z = mesh_vol.z[i]
            T_vol[i] = Tmin+(Tmax-Tmin)/((1+(math.e**(-alpha*(z-z0)))))


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
        mesh_surf_obj.initialize()
        mesh_vol_obj.initialize()

        epsilon_surf = 0.9 * np.ones(mesh_surf_obj.N_surf)
        kappa = mesh_vol_obj.kappa

        T_surf = self.wall_temperature(mesh_surf_obj)
        T_vol = self.vol_temperature(mesh_vol_obj)

        DEF_obj = RadiationCode.DirectExchangeFactor(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj)
        DEF_obj.eval()
        RC_obj = RadiationCode.ZonalMethod()
        SiSj, SiGj, GiSj, GiGj = RC_obj.matrix_main(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj)

        Q_s, Q_g,Q_si_wo_SiGj = RC_obj.solver(mesh_surf_obj, mesh_surf_obj, mesh_vol_obj, T_surf, T_surf, T_vol, SiSj, SiGj, GiSj,
                                 GiGj)

        fig, ax = plt.subplots()
        x_s_plot = np.reshape(mesh_surf_obj.x, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
        Q_s_plot = np.reshape(Q_s, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
        area_surf = np.reshape(mesh_surf_obj.Area, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial),
                               order='F')
        img = ax.imshow(Q_s_plot / area_surf, cmap='YlOrRd_r')
        ax.invert_yaxis()
        ax.set_xlabel('Z (m)')
        ax.set_ylabel('Azimuthal angle')
        azimuthal_vals = np.linspace(0, 360, mesh_surf_obj.num_azimuthal) * (2 / 360)  # Azimuthal in degrees
        axial_vals = np.linspace(0, mesh_surf_obj.L, mesh_surf_obj.num_axial)
        xticks = range(mesh_surf_obj.num_axial)
        # Choose tick indices (e.g., every 10th value)
        azim_tick_idx = np.linspace(0, mesh_surf_obj.num_azimuthal - 1, mesh_surf_obj.num_azimuthal, dtype=int)
        axial_tick_idx = np.linspace(0, mesh_surf_obj.num_axial - 1, mesh_surf_obj.num_axial, dtype=int)
        ax.set_xticks(xticks[::3])

        ax.set_yticks(range(mesh_surf_obj.num_azimuthal))
        ax.set_xticklabels([f"{axial_vals[i]:.2f}" for i in axial_tick_idx[::3]])  # adjust number of ticks as needed
        ax.set_yticklabels(
            [f"{(azimuthal_vals[i]):.1f}" + "$\pi$" if i > 0 else f"{(azimuthal_vals[i]):.0f}" for i in azim_tick_idx])
        cbar = fig.colorbar(img, ax=ax, orientation='horizontal', pad=0.2)
        # Set the colorbar label (acts as a title under it)
        cbar.set_label('Radiative Heat Flux (W/m$^2$)', labelpad=10)

        fig01, ax01 = plt.subplots()
        img1 = ax01.imshow(x_s_plot, cmap='jet')
        ax01.invert_yaxis()

        fig01.colorbar(img1)

        # Create polar mesh
        fig1, ax1 = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(6, 6))
        ax1.set_yticklabels([])  # optional: remove radial ticks
        Q_g_plot = np.reshape(Q_g, (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial),
                              order='F')
        vol_gas = np.reshape(mesh_vol_obj.Volume,
                             (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial), order='F')
        Q_g_flux_plot = Q_g_plot[:, :, :] / vol_gas[:, :, :]
        vmin = -np.max(Q_g_flux_plot[:, :,:])#np.min(Q_g_flux_plot[:, :])
        vmax = np.max(Q_g_flux_plot[:, :,:])
        # Plot in polar coordinates
        r = np.linspace(0, mesh_vol_obj.D / 2, mesh_vol_obj.num_radial)  # ring radii
        theta = np.linspace(0, 2 * np.pi, mesh_vol_obj.num_azimuthal)  # angular divisions
        # Plot using pcolormesh
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        c = ax1.pcolormesh(Theta, R, Q_g_flux_plot[:,:,10], cmap='YlOrRd', shading='auto', snap=True, vmin=vmin, vmax=vmax)
        # img1 = ax1.plot(Theta,Q_g_plot[:,:,2])
        # Add colorbar
        plt.colorbar(c, ax=ax1, pad=0.1, orientation='horizontal').set_label('Volumetric Radiative Heat Flux (W/m$^2$)')

        fig2, ax2 = plt.subplots()

        img2 = ax2.imshow(Q_g_flux_plot[:,9,:], cmap='YlOrRd', vmin=vmin, vmax=vmax)
        ax2.invert_yaxis()
        ax2.set_xlabel('Z (m)')
        ax2.set_ylabel('R (m)')

        radial_vals = np.linspace(0, mesh_vol_obj.D / 2, mesh_vol_obj.num_radial)  # Azimuthal in degrees
        axial_vals = np.linspace(0, mesh_vol_obj.L, mesh_vol_obj.num_axial)
        xticks = range(mesh_vol_obj.num_axial)
        # Choose tick indices (e.g., every 10th value)
        radial_tick_idx = np.linspace(0, mesh_vol_obj.num_radial - 1, mesh_vol_obj.num_radial, dtype=int)
        axial_tick_idx = np.linspace(0, mesh_vol_obj.num_axial - 1, mesh_vol_obj.num_axial, dtype=int)
        ax2.set_xticks(xticks[::3])

        ax2.set_yticks(range(mesh_vol_obj.num_azimuthal))
        ax2.set_xticklabels([f"{axial_vals[i]:.2f}" for i in axial_tick_idx[::3]])  # adjust number of ticks as needed
        ax2.set_yticklabels([f"{(radial_vals[i]):.2f}" for i in radial_tick_idx])
        cbar = fig2.colorbar(img2, ax=ax2, orientation='horizontal', pad=0.2)
        # Set the colorbar label (acts as a title under it)
        cbar.set_label('Volumetric Radiative Heat Flux (W/m$^3$)', labelpad=10)

        fig3, ax3 = plt.subplots()
        T_g_plot = np.reshape(T_vol, (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial),
                              order='F')
        z_g_plot = np.reshape(mesh_vol_obj.z,
                              (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial),
                              order='F')
        ax3.plot(z_g_plot[3,1,:],T_g_plot[3, 1, :])
        ax3.set_ylabel('Temperature (K)')
        ax3.set_xlabel('Z (m)')

        pathsave = parentdir + '/data/'
        figname = "RadiativeHeatFlux_InnerSurface_SigmoidalGasTemp"
        fig.savefig(pathsave + figname + ".png", dpi=600, bbox_inches="tight")

        figname1 = "RadiativeHeatFlux_HotGas_crosssection_SigmoidalGasTemp"
        fig1.savefig(pathsave + figname1 + ".png", dpi=600, bbox_inches="tight")

        figname2 = "RadiativeHeatFlux_HotGas_axialsection_SigmoidalGasTemp"
        fig2.savefig(pathsave + figname2 + ".png", dpi=600, bbox_inches="tight")

        figname3 = "TemperatureProfile_SigmoidalGasTemp"
        fig3.savefig(pathsave + figname3 + ".png", dpi=600, bbox_inches="tight")

        plt.show()


if __name__=='__main__':
    obj = AxialTempProfile_Cylinder()
    obj.main()