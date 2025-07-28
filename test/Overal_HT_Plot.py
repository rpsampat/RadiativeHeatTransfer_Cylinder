import os
import sys
import inspect
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL.TiffTags import TAGS_V2_GROUPS

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir+'/src/')
pathsave = parentdir +'/data/OverallHT/'
# import mesh
def heat_transfer_plot(dictdat):
    Q_s = dictdat["Q_s"]
    Q_si_wo_SiGj2 = dictdat["Q_si_wo_SiGj2"]
    Q_g = dictdat["Q_g"]
    Q_g2 = dictdat['Q_g2']
    Q_s2 = dictdat["Q_s2"]


    mesh_surf_obj = dictdat['Surf_Liner_wall']
    mesh_surf_obj2 = dictdat['Surf_Casing_wall']
    mesh_vol_obj = dictdat['Liner_Gas_mesh']
    Q_si_wo_SiGj2 = dictdat['Q_si_wo_SiGj2']
    h_conv_liner_gas = dictdat['h_conv_liner_gas']
    h_conv_liner_coolant = dictdat['h_conv_liner_coolant']
    Mesh_wall_liner = dictdat['Mesh_wall_liner']
    T_wall_liner = dictdat['T_wall_liner']
    T_annulus = dictdat['T_annulus']
    P_annulus = dictdat['P_annulus']
    rho_annulus = dictdat['rho_annulus']
    Annulus_flow_obj = dictdat['Annulus_flow_obj']
    T_g = dictdat['Core_Temp']
    P_g = dictdat['Core_Pr']
    X_g = dictdat['Core_comp']

    Q_s_plot = np.reshape(Q_s, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
    area_surf = np.reshape(mesh_surf_obj.Area, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial),
                           order='F')
    q_rad_inner = Q_s_plot[3, :] / area_surf[3, :]

    Q_s_plot2 = np.reshape(Q_si_wo_SiGj2, (mesh_surf_obj2.num_azimuthal, mesh_surf_obj2.num_axial), order='F')
    area_surf2 = np.reshape(mesh_surf_obj2.Area, (mesh_surf_obj2.num_azimuthal, mesh_surf_obj2.num_axial),
                            order='F')
    q_rad_outer = Q_s_plot2[3, :] / area_surf2[3, :]

    z = [Mesh_wall_liner.z[i * Mesh_wall_liner.num_thickness] for i in range(Mesh_wall_liner.num_axial)]
    
    fig, ax = plt.subplots()
    ax.plot(z,q_rad_inner)
    ax.set_xlabel('Z(m)')
    ax.set_ylabel('$\dot{Q}_{rad}$ liner inner wall (W/m$^2$)')
    figname = 'fig0_q_rad_inner.png'

    fig1, ax1 = plt.subplots()
    ax1.plot(z,q_rad_outer)
    ax1.set_xlabel('Z(m)')
    ax1.set_ylabel('$\dot{Q}_{rad}$ liner outer wall (W/m$^2$)')
    figname1 = 'fig1_q_rad_outer.png'

    fig2, ax2 = plt.subplots()
    ax2.plot(z,h_conv_liner_gas)
    ax2.set_xlabel('Z(m)')
    ax2.set_ylabel('h$_{conv}$ liner gas-side (W/m$^2$-K)')
    figname2 = 'fig2_h_conv_liner_gas.png'

    fig3, ax3 = plt.subplots()
    ax3.plot(z,h_conv_liner_coolant)
    ax3.set_xlabel('Z(m)')
    ax3.set_ylabel('h$_{conv}$ liner coolant-side (W/m$^2$-K)')
    figname3 = 'fig3_h_conv_liner_coolant.png'

    fig4, ax4 = plt.subplots()
    ax4.plot(z,T_g)
    ax4.set_xlabel('Z(m)')
    ax4.set_ylabel('T gas (K)')
    figname4 = 'fig4_T_g.png'

    fig5, ax5 = plt.subplots()
    ax5.plot(z,T_annulus)
    ax5.set_xlabel('Z(m)')
    ax5.set_ylabel('T fluid annulus (K)')
    figname5 = 'fig5_T_annulus.png'

    fig6, ax6 = plt.subplots()
    ax6.plot(z,P_annulus)
    ax6.set_xlabel('Z(m)')
    ax6.set_ylabel('PT fluid annulus (Pa)')
    figname6 = 'fig6_P_annulus.png'

    fig7, ax7 = plt.subplots()
    ax7.plot(z,rho_annulus)
    ax7.set_xlabel('Z(m)')
    ax7.set_ylabel('Density fluid annulus (kg/m3)')
    figname7 = 'fig7_rho_annulus.png'

    fig8, ax8 = plt.subplots()
    img = ax8.imshow(T_wall_liner)
    ax8.set_ylabel('Thickness (m)')
    ax8.set_xlabel('Z (m)')
    ax8.invert_yaxis()
    fig8.colorbar(img)
    figname8 = 'fig8_T_wall_liner.png'

    fig9, ax9 = plt.subplots()
    ax9.plot(z,T_wall_liner[-1,:], label='Outer')
    ax9.plot(z,T_wall_liner[0,:], label = 'Inner')
    ax9.set_ylabel('Temperature (K)')
    ax9.set_xlabel('Z (m)')
    ax9.legend(loc='upper left')
    figname9 = 'fig9_T_wall_liner_lineplot.png'

    fig10, ax10 = plt.subplots()
    ax10.plot(z, P_g)
    ax10.set_xlabel('Z (m)')
    ax10.set_ylabel('P gas (Pa)')
    figname10 = 'fig10_P_g.png'

    # Radiative heat flux on wall
    fig11, ax11 = plt.subplots()
    x_s_plot = np.reshape(mesh_surf_obj.x, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
    Q_s_plot = np.reshape(Q_s, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial), order='F')
    area_surf = np.reshape(mesh_surf_obj.Area, (mesh_surf_obj.num_azimuthal, mesh_surf_obj.num_axial),
                           order='F')
    img11 = ax11.imshow(Q_s_plot / area_surf, cmap='YlOrRd_r')
    ax11.invert_yaxis()
    ax11.set_xlabel('Z (m)')
    ax11.set_ylabel('Azimuthal angle')
    azimuthal_vals = np.linspace(0, 360, mesh_surf_obj.num_azimuthal) * (2 / 360)  # Azimuthal in degrees
    axial_vals = np.linspace(0, mesh_surf_obj.L, mesh_surf_obj.num_axial)
    xticks = range(mesh_surf_obj.num_axial)
    # Choose tick indices (e.g., every 10th value)
    azim_tick_idx = np.linspace(0, mesh_surf_obj.num_azimuthal - 1, mesh_surf_obj.num_azimuthal, dtype=int)
    axial_tick_idx = np.linspace(0, mesh_surf_obj.num_axial - 1, mesh_surf_obj.num_axial, dtype=int)
    ax11.set_xticks(xticks[::3])

    ax11.set_yticks(range(mesh_surf_obj.num_azimuthal))
    ax11.set_xticklabels([f"{axial_vals[i]:.2f}" for i in axial_tick_idx[::3]])  # adjust number of ticks as needed
    ax11.set_yticklabels(
        [f"{(azimuthal_vals[i]):.1f}" + "$\pi$" if i > 0 else f"{(azimuthal_vals[i]):.0f}" for i in azim_tick_idx])
    cbar = fig11.colorbar(img11, ax=ax11, orientation='horizontal', pad=0.2)
    # Set the colorbar label (acts as a title under it)
    cbar.set_label('Radiative Heat Flux (W/m$^2$)', labelpad=10)
    figname11 = "RadiativeHeatFlux_Liner_InnerSurface.png"


    #Radiative heat flux from gas
    fig12, ax12 = plt.subplots()
    Q_g_plot = np.reshape(Q_g, (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial),
                          order='F')
    vol_gas = np.reshape(mesh_vol_obj.Volume,
                         (mesh_vol_obj.num_radial, mesh_vol_obj.num_azimuthal, mesh_vol_obj.num_axial), order='F')
    Q_g_flux_plot = Q_g_plot[:, :, :] / vol_gas[:, :, :]
    vmin = -np.max(Q_g_flux_plot[:, :, :])  # np.min(Q_g_flux_plot[:, :])
    vmax = np.max(Q_g_flux_plot[:, :, :])

    img12 = ax12.imshow(Q_g_flux_plot[:, 9, :], cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax12.invert_yaxis()
    ax12.set_xlabel('Z (m)')
    ax12.set_ylabel('R (m)')

    radial_vals = np.linspace(0, mesh_vol_obj.D / 2, mesh_vol_obj.num_radial)  # Azimuthal in degrees
    axial_vals = np.linspace(0, mesh_vol_obj.L, mesh_vol_obj.num_axial)
    xticks = range(mesh_vol_obj.num_axial)
    # Choose tick indices (e.g., every 10th value)
    radial_tick_idx = np.linspace(0, mesh_vol_obj.num_radial - 1, mesh_vol_obj.num_radial, dtype=int)
    axial_tick_idx = np.linspace(0, mesh_vol_obj.num_axial - 1, mesh_vol_obj.num_axial, dtype=int)
    ax12.set_xticks(xticks[::3])

    yticks = range(mesh_vol_obj.num_radial)
    ax12.set_yticks(yticks[::3])
    ax12.set_xticklabels([f"{axial_vals[i]:.2f}" for i in axial_tick_idx[::3]])  # adjust number of ticks as needed
    ax12.set_yticklabels([f"{(radial_vals[i]):.2f}" for i in radial_tick_idx[::3]])
    cbar = fig12.colorbar(img12, ax=ax12, orientation='horizontal', pad=0.2)
    # Set the colorbar label (acts as a title under it)
    cbar.set_label('Volumetric Radiative Heat Flux (W/m$^3$)', labelpad=10)
    figname12 = "RadiativeHeatFlux_HotGas_axialsection.png"

    # === Save all figures at the end ===

    figures = [fig, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10,fig11,fig12]
    fignames = [figname, figname1, figname2, figname3, figname4,
                figname5, figname6, figname7, figname8, figname9, figname10,figname11,figname12]

    for f, name in zip(figures, fignames):
        f.savefig(os.path.join(pathsave, name), dpi=500, bbox_inches='tight')

    # plt.show()


def open_file():
    axial_elem = 30
    azimuthal_elem = 10
    radial_elem_hotgas = 20
    thickness_liner_elem = 5
    thickness_casing_elem = 5
    Xi = 0.3  # cooling fract
    caseid = 'axial' + str(axial_elem) + '_linerthelem' + str(thickness_liner_elem) + '_mdotcoolfrac' + str(Xi)+"_nofilmcooling"
    with open(parentdir + '/data/' + 'dictsave_' + caseid + '.pkl', 'rb') as file:
        dictdat = pickle.load(file, encoding='latin-1')

    return dictdat

def main():
    data = open_file()
    heat_transfer_plot(data)


if __name__=="__main__":
    main()
