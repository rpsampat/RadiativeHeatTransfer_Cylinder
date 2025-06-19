import math
import numpy as np
class Surf_Cyl_Mesh:
    def __init__(self):
        self.num_azimuthal= 10
        self.num_axial = 30
        self.N_surf = self.num_axial*self.num_azimuthal
        self.x=np.zeros(self.N_surf)
        self.y=np.zeros(self.N_surf)
        self.z = np.zeros(self.N_surf)
        self.Area = np.zeros(self.N_surf)
        self.n_x = np.zeros(self.N_surf)
        self.n_y=np.zeros(self.N_surf)
        self.n_z =np.zeros(self.N_surf)
        self.L = 0.3 #m
        self.D = 0.10 #m
        # self.cyl_coords()

    def cyl_coords(self):
        count=0
        for i in range(self.num_axial):
            for j in range(self.num_azimuthal):
                d_theta = 2*math.pi/self.num_azimuthal
                theta = (j+0.5)* d_theta
                self.x[count] = (self.D/2)*math.cos(theta)
                self.y[count] = (self.D/2)*math.sin(theta)
                self.z[count] = (i+1)*self.L/self.num_axial
                self.Area[count] = (self.D/2)*d_theta*(self.L/self.num_axial)
                self.n_x[count] = -(self.D/2)*math.cos(theta)
                self.n_y[count] = -(self.D/2)*math.sin(theta)
                self.n_z[count] = 0

                count+=1

class Vol_Cyl_Mesh:
    def __init__(self):
        self.num_azimuthal= 10
        self.num_axial = 30
        self.num_radial  = 10
        self.N_vol = self.num_axial*self.num_azimuthal*self.num_radial

        self.x=np.zeros(self.N_vol)
        self.y=np.zeros(self.N_vol)
        self.z = np.zeros(self.N_vol)
        self.Volume = np.zeros(self.N_vol)
        self.n_x = np.zeros(self.N_vol)
        self.n_y=np.zeros(self.N_vol)
        self.n_z =np.zeros(self.N_vol)
        self.kappa = 10*np.ones(self.N_vol)
        self.L = 0.3 #m
        self.D = 0.10 #m
        # self.cyl_coords()

    def cyl_coords(self):
        count=0
        for i in range(self.num_axial):
            for j in range(self.num_azimuthal):
                d_theta = 2 * math.pi / self.num_azimuthal
                theta = (j+0.5) * d_theta
                for k in range(self.num_radial):
                    dR = (self.D/2)/self.num_radial
                    R = (k*dR) +(dR/2)
                    self.x[count] = R*math.cos(theta)
                    self.y[count] = R*math.sin(theta)
                    self.z[count] = ((i)+0.5)*self.L/self.num_axial
                    self.Volume[count] = R*R*d_theta*(self.L/self.num_axial)
                    self.n_x[count] = -R*math.cos(theta)
                    self.n_y[count] = -R*math.sin(theta)
                    self.n_z[count] = 0
                    count+=1
        self.Volume_total = (math.pi*(self.D**2)/4)*self.L

