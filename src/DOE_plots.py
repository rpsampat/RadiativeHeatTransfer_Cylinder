import os
import inspect
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import FormatStrFormatter,LogFormatter
from matplotlib.colors import LogNorm
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)


def two_abscissa_plot(val_stk, y_list, x_list, ylabel, xlabel, zlabel, figname, show, save, z_norm, cmap, clim1,clim2):
    fig, ax = plt.subplots()

    if z_norm == "log":
        imgplot = ax.imshow(val_stk, cmap=cmap, origin="lower", norm=LogNorm())
    else:
        imgplot = ax.imshow(val_stk, cmap=cmap, origin="lower")
    imgplot.cmap.set_bad(color='white')

    if len(x_list) > len(y_list):
        if z_norm == "linear":
            cbar = fig.colorbar(imgplot,orientation='horizontal', format=tick.FormatStrFormatter('%.2g'))
            cbar.ax.set_xlabel(zlabel)
        else:
            cbar = fig.colorbar(imgplot,orientation='horizontal')
            cbar.ax.set_xlabel(zlabel)
        # cbar.ax.set_ylabel(zlabel)
    else:
        if z_norm == "linear":
            cbar = fig.colorbar(imgplot,orientation='vertical', format=tick.FormatStrFormatter('%.2g'))
            cbar.ax.set_ylabel(zlabel)
        else:
            cbar = fig.colorbar(imgplot,orientation='vertical')
            cbar.ax.set_ylabel(zlabel)


    # cbar.set_ticks(np.arange(np.min(val_stk), np.max(val_stk) + 1,100))
    ax.set_yticks(range(len(y_list)), y_list)
    ax.set_xticks(range(len(x_list)), x_list)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    imgplot.set_clim(clim1,clim2)
    if save == 'y':
        fig.savefig(figname + '.pdf')
        fig.savefig(figname + '.png')
    if show == 'y':
        plt.show()


def DOE_plots():
    Pin = 10e5  # Pa
    beta = 0.0

    with open(parentdir+'/data/' + 'dictsave_DOE_Pin_'+str(Pin) +'_lowBR' + '.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    BR_list = data_dict['BR']
    Tin_list = data_dict['Tin']
    Uf_list = data_dict['Uf']
    EAR_list = data_dict['EAR']
    EAR =EAR_list[1]
    BR = BR_list[1]
    print(BR)
    beta_list = data_dict['Beta']
    beta = beta_list[-1]
    # # Contour Twall max on Tin vs Uf
    # for Tin in Tin_list:
    #     val_arr=[]
    #     for Uf in Uf_list:
    #         caseid = f'Tin_{Tin}_Uf_{Uf:.2f}_EAR_{EAR}_BR_{BR}_Beta_{beta}'
    #         ht_dict = data_dict[caseid]
    #         Tmax = np.max(ht_dict['T_wall_liner'])
    #         val_arr.append(Tmax)
    #     try:
    #         val_stk = np.vstack((val_stk, val_arr))
    #     except:
    #         val_stk = np.array(val_arr)
    #
    # two_abscissa_plot(val_stk, Tin_list, Uf_list, ylabel='Temperature (K)', xlabel= 'Fuel Utilization factor',
    #                   zlabel='T$_{wall}$', figname='MaxWallTemperature', show='y', save='n', z_norm='linear', cmap='jet',clim1=800,clim2=1500)

    # # Contour Twall max on Tin vs BR
    # Uf = Uf_list[3]
    # print(Uf)
    # for Tin in Tin_list:
    #     val_arr = []
    #     for BR in BR_list:
    #         caseid = f'Tin_{Tin}_Uf_{Uf:.2f}_EAR_{EAR}_BR_{BR}_Beta_{beta}'
    #         ht_dict = data_dict[caseid]
    #         Tmax = np.max(ht_dict['T_wall_liner'])
    #         val_arr.append(Tmax)
    #     try:
    #         val_stk = np.vstack((val_stk, val_arr))
    #     except:
    #         val_stk = np.array(val_arr)
    #
    # two_abscissa_plot(val_stk, Tin_list, BR_list, ylabel='Temperature (K)', xlabel='Blowing Ratio',
    #                   zlabel='T$_{wall}$', figname='MaxWallTemperature', show='y', save='n', z_norm='linear',
    #                   cmap='jet', clim1=800, clim2=1500)

    # Contour Twall max on EAR vs BR
    Uf = Uf_list[1]
    print(Uf)
    Tin = Tin_list[-1]
    for EAR in EAR_list:
        val_arr = []
        for BR in BR_list:
            caseid = f'Tin_{Tin}_Uf_{Uf:.2f}_EAR_{EAR}_BR_{BR}_Beta_{beta}'
            ht_dict = data_dict[caseid]
            Tmax = np.max(ht_dict['T_wall_liner'])
            T_g = ht_dict['Core_Temp']
            val_arr.append(Tmax)
        try:
            val_stk = np.vstack((val_stk, val_arr))
        except:
            val_stk = np.array(val_arr)

    two_abscissa_plot(val_stk, EAR_list, BR_list, ylabel='Excess Air Ratio', xlabel='Blowing Ratio',
                      zlabel='T$_{wall}$', figname='MaxWallTemperature', show='n', save='n', z_norm='linear',
                      cmap='jet', clim1=1100, clim2=2100)
    val_stk = 0

    # Contour Tg on EAR vs BR
    Uf = Uf_list[1]
    print(Uf)
    Tin = Tin_list[-1]
    for EAR in EAR_list:
        val_arr = []
        for BR in BR_list:
            caseid = f'Tin_{Tin}_Uf_{Uf:.2f}_EAR_{EAR}_BR_{BR}_Beta_{beta}'
            ht_dict = data_dict[caseid]
            Tmax = np.max(ht_dict['T_wall_liner'])
            T_g = ht_dict['Core_Temp'][-1]
            val_arr.append(T_g)
        try:
            val_stk = np.vstack((val_stk, val_arr))
        except:
            val_stk = np.array(val_arr)

    two_abscissa_plot(val_stk, EAR_list, BR_list, ylabel='Excess Air Ratio', xlabel='Blowing Ratio',
                      zlabel='T$_{gas}$', figname='GasTemperature', show='y', save='n', z_norm='linear',
                      cmap='jet', clim1=1700, clim2=2900)

if __name__=='__main__':
    DOE_plots()
