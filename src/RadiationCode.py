import math
import  numpy as np
import pickle
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
datadir = parentdir+'/data/'
class ZonalMethod:
    def __init__(self):
        self.L=1.0 #m
        self.sigma = 5.67e-8

    def sisj_eval(self):
        file_path = datadir+'DirectExchangeFactor_ss'
        with open(file_path, 'rb') as file:
            sisj_data = pickle.load(file)

        return sisj_data

    def gisj_eval(self):
        file_path = datadir+'DirectExchangeFactor_gs'
        with open(file_path, 'rb') as file:
            gisj_data = pickle.load(file)

        return gisj_data

    def gigj_eval(self):
        file_path = datadir+'DirectExchangeFactor_gg'
        with open(file_path, 'rb') as file:
            gigj_data = pickle.load(file)

        return gigj_data

    def T_matrix_eval(self,epsilon,rho,Area,sisj):
        t_fact = -rho/(epsilon*Area)
        diag_fact = 1/epsilon
        T = (sisj @ np.diag(t_fact))+np.diag(diag_fact)

        return T

    def S_matrix_eval(self,epsilon,sisj):
        S=sisj@np.diag(epsilon)

        return S

    def R_matrix_eval(self,gisj,epsilon):
        R = gisj @ np.diag(epsilon)

        return R

    def Q_matrix_eval(self, epsilon, rho, Area, gisj):
        q_fact = rho / (epsilon * Area)
        Q = (gisj @ np.diag(q_fact))

        return Q

    def matrix_main(self,mesh_surf,mesh_vol):
        DEF_obj = DirectExchangeFactor(mesh_surf,mesh_vol)
        sisj =self.sisj_eval()
        gisj = self.gisj_eval()
        gigj = self.gigj_eval()
        epsilon_surf = 0.9*np.ones(mesh_surf.N_surf) # emissivity
        rho_surf = (1-0.9)*np.ones(mesh_surf.N_surf) # reflectivity
        T = self.T_matrix_eval(epsilon_surf,rho_surf,mesh_surf.Area,sisj)
        S = self.S_matrix_eval(epsilon_surf,sisj)
        R = self.R_matrix_eval(gisj, epsilon_surf)
        Q = self.Q_matrix_eval(epsilon_surf, rho_surf, mesh_surf.Area, gisj)
        SiSj = np.linalg.inv(T) @ S
        GiSj = R + Q @ SiSj
        SiGj = np.transpose(GiSj)
        GiGj = gigj + Q @ SiGj

        return SiSj, SiGj, GiSj, GiGj

    def solver(self, mesh_surf_obj,mesh_vol_obj,T_surf,T_vol):
        epsilon_surf = 0.9 * np.ones(mesh_surf_obj.N_surf)
        kappa = mesh_vol_obj.kappa
        SiSj, SiGj, GiSj, GiGj = self.matrix_main(mesh_surf_obj, mesh_vol_obj)

        Eb_s = self.sigma * (T_surf ** 4)
        Eb_g = self.sigma * (T_vol ** 4)

        Q_s = (epsilon_surf * mesh_surf_obj.Area * Eb_s) - (SiSj @ Eb_s) - (SiGj @ Eb_g)
        Q_g = (4 * kappa * mesh_vol_obj.Volume * Eb_g) - (GiSj @ Eb_s) - (GiGj @ Eb_g)

        return Q_s,Q_g





class DirectExchangeFactor:
    def __init__(self,mesh_surf1,mesh_vol1):
        self.surf1 = mesh_surf1
        self.vol1 = mesh_vol1
        self.main_surf()
        self.main_vol()


    def sisj(self,i,j):
        s_x = self.surf1.x[i]-self.surf1.x[j]
        s_y = self.surf1.y[i] - self.surf1.y[j]
        s_z = self.surf1.z[i] - self.surf1.z[j]
        S2 = (s_x**2)+(s_y**2)+(s_z**2)
        cos_theta_i_S2 = ((self.surf1.n_x[i]*-s_x)+(self.surf1.n_y[i]*-s_y)+(self.surf1.n_z[i]*-s_z))/math.sqrt(S2)
        cos_theta_j_S2 = ((self.surf1.n_x[j] * s_x) + (self.surf1.n_y[j] * s_y) + (self.surf1.n_z[j] * s_z))/math.sqrt(S2)
        sisj = cos_theta_i_S2*cos_theta_j_S2*self.surf1.Area[i]*self.surf1.Area[j]/(math.pi*(S2))

        return sisj

    def gisj(self,i,j):
        s_x = self.vol1.x[i] - self.surf1.x[j]
        s_y = self.vol1.y[i] - self.surf1.y[j]
        s_z = self.vol1.z[i] - self.surf1.z[j]
        S2 = (s_x ** 2) + (s_y ** 2) + (s_z ** 2)
        cos_theta_j_S2 = ((self.surf1.n_x[j] * s_x) + (self.surf1.n_y[j] * s_y) + (self.surf1.n_z[j] * s_z))/math.sqrt(S2)
        gisj = cos_theta_j_S2 * self.surf1.Area[j] * self.vol1.Volume[i] / (math.pi * (S2))

        return gisj

    def path_absorption(self,i,j):
        s_x = self.vol1.x[i] - self.vol1.x[j]
        s_y = self.vol1.y[i] - self.vol1.y[j]
        s_z = self.vol1.z[i] - self.vol1.z[j]
        S = math.sqrt((s_x ** 2) + (s_y ** 2) + (s_z ** 2))
        #Determining net kappa along absorption path
        if S==0:
            kappa_avg = self.vol1.kappa[i]
            kappa_S_avg = 0
        else:
            ds = math.pow((self.vol1.Volume_total/self.vol1.N_vol),0.333)
            num_pts = int(S/(2*ds))+1
            ds = S/num_pts
            kappa_sum=0
            kappa_S_avg =0
            for p in range(num_pts):
                x2 = self.vol1.x[j] + ((p * ds) + ds / 2) * s_x / S
                y2 = self.vol1.y[j] + ((p * ds) + ds / 2) * s_y / S
                z2 = self.vol1.z[j] + ((p * ds) + ds / 2) * s_z / S
                ind_x = np.where(np.abs(self.vol1.x-x2)<ds/2)[0]
                ind_y = np.where(np.abs(self.vol1.y - y2) < ds/2)[0]
                ind_z = np.where(np.abs(self.vol1.z - z2) < ds/2)[0]
                ind_intersect = np.intersect1d(ind_x,ind_y)
                ind_intersect = np.intersect1d(ind_intersect, ind_y)
                kappa_local=0
                if len(ind_intersect)==0:
                    continue
                else:
                    for q in range(len(ind_intersect)):
                        kappa_local += self.vol1.kappa[q]
                    kappa_local = kappa_local/len(ind_intersect)
                    kappa_sum += kappa_local

            kappa_avg = kappa_sum*ds/S
            kappa_S_avg = kappa_sum*ds
        
        return kappa_avg, kappa_S_avg
    
    def gigj(self, i, j):
        s_x = self.vol1.x[i] - self.vol1.x[j]
        s_y = self.vol1.y[i] - self.vol1.y[j]
        s_z = self.vol1.z[i] - self.vol1.z[j]
        S2 = (s_x ** 2) + (s_y ** 2) + (s_z ** 2)
        # kappa, kappa_S = self.path_absorption(i,j)
        kappa = self.vol1.kappa[i]
        kappa_S = kappa*math.sqrt(S2)
        gigj = math.exp(-kappa_S)*(kappa**2)*self.vol1.Volume[i] * self.vol1.Volume[j] / (math.pi * (S2))

        return gigj
            




    def main_surf(self):
        N1 = self.surf1.N_surf
        S=np.zeros((N1,N1))
        for i in range(N1):
            for j in range(i+1,N1):
                S[i,j] = self.sisj(i,j)
                S[j,i] = S[i,j]
        file_path = datadir+'DirectExchangeFactor_ss'
        with open(file_path, 'wb') as file:
            pickle.dump(S, file, pickle.HIGHEST_PROTOCOL)

    def main_vol(self):
        N1 = self.vol1.N_vol
        N2 = self.surf1.N_surf
        GS = np.zeros((N1, N2))

        for i in range(N1):
            for j in range(N2):
                GS[i, j] = self.gisj(i, j)
        file_path = datadir+'DirectExchangeFactor_gs'
        with open(file_path, 'wb') as file:
            pickle.dump(GS, file, pickle.HIGHEST_PROTOCOL)

        GG = np.zeros((N1,N1))
        for i in range(N1):
            for j in range(i+1,N1):
                GG[i, j] = self.gigj(i, j)
                GG[j, i] = GG[i, j]
        file_path = datadir+'DirectExchangeFactor_gg'
        with open(file_path, 'wb') as file:
            pickle.dump(GG, file, pickle.HIGHEST_PROTOCOL)


