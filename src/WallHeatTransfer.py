import numpy as np
import scipy
import math


class WallHeatTransfer:
    """
    Wall heat transfer of a coaxial cylinder setup to emulate a gas turbine combustor and its casing. The inner wall is
    the combustor liner and the outer wall is the pressure vessel casing. The solid wall is modelled as a finite
    difference problem in 2D and the convection in the liner and annulus are solved as 1D problems.
    """
    def __init__(self):
        self.L=1
        self.sigma = 5.67e-8

    def wall_matrix_setup_fdm(self,mesh,q_rad_inner,q_rad_outer,q_conv2,T_f_cool,T_g,T_in,conv_coeff_P_bottom,conv_B_vect_bottom):
        """
        Heat transfer in solid wall. Finite Difference Discretization
        :param mesh: solid wall mesh resolved along length and thickness
        :param q_rad_inner:
        :param q_rad_outer:
        :param q_conv1:
        :param q_conv2:
        :return:
        """
        A_mat = np.zeros((mesh.N_surf,mesh.N_surf))
        B_vect = np.zeros(mesh.N_surf)
        for i in range(mesh.num_axial):
            for j in range(mesh.num_thickness):
                ind = (i*mesh.num_thickness)+j
                k = mesh.k_mat[ind]
                dy = mesh.d_y[ind]
                dz = mesh.d_z[ind]
                ind_z0 = ((i - 1) * mesh.num_thickness) + j
                ind_z1 = ((i + 1) * mesh.num_thickness) + j
                ind_y0 = (i * mesh.num_thickness) + j - 1
                ind_y1 = (i * mesh.num_thickness) + j + 1

                A_mat[ind, ind] += - 2*(k * dz / (dy**2)) - 2*(k * dy / (dz**2))

                if i<mesh.num_axial-1 and i>0 and j<mesh.num_thickness-1 and j>0:
                    A_mat[ind, ind_z1] += (k * dy / (dz**2))
                    A_mat[ind, ind_z0] += (k * dy / (dz**2))
                    A_mat[ind, ind_y1] += (k * dz / (dy**2))
                    A_mat[ind, ind_y0] += (k * dz / (dy**2))

                elif i==0 and j<mesh.num_thickness-1 and j>0:
                    # left boundary
                    B_vect[ind] += -(k * dy * T_f_cool[i]/ (dz**2))
                    A_mat[ind, ind_z1] += (k * dy / (dz**2))
                    A_mat[ind, ind_y1] += (k * dz / (dy**2))
                    A_mat[ind, ind_y0] += (k * dz / (dy**2))

                elif i==mesh.num_axial-1 and j<mesh.num_thickness-1 and j>0:
                    # right boundary
                    # B_vect[ind] += -(2*k * dy * T_f_cool[i] / (dz ** 2))
                    A_mat[ind, ind_z0] += (k * dy / (dz**2))#-(k * dy / (dz**2))
                    A_mat[ind, ind_y1] += (k * dz / (dy**2))
                    A_mat[ind, ind_y0] += (k * dz / (dy**2))
                    A_mat[ind, ind] += (k * dy / (dz**2))

                elif j==mesh.num_thickness-1:
                    # top end boundary condition of wall
                    B_vect[ind] += -(q_conv2[i]*(dz/dy)*T_f_cool[i]) - (q_rad_outer[i] * dz) #+ (0.9*self.sigma * dz *((Tw_upper[i]**4)-(1000**4))) #-(k*T_f_cool[i]*dz/(dy**2))
                    A_mat[ind, ind] += (k * dz / (dy**2))-q_conv2[i] *(dz/dy)
                    A_mat[ind, ind_y0] += (k * dz / (dy**2))
                    if i<mesh.num_axial-1 and i>0:
                        A_mat[ind, ind_z1] += (k * dy / (dz**2))
                        A_mat[ind, ind_z0] += (k * dy / (dz**2))
                    elif i==0:
                        # left boundary
                        B_vect[ind] += -(k * dy * T_f_cool[i] / (dz**2))
                        A_mat[ind, ind_z1] += (k * dy / (dz ** 2))
                        # A_mat[ind, ind] += -(k * dy / (dz ** 2))
                    elif i==mesh.num_axial-1:
                        # right boundary
                        # B_vect[ind] += -(2*k * dy * T_f_cool[i] / (dz ** 2))
                        A_mat[ind, ind] += (k * dy / (dz ** 2))
                        # A_mat[ind, ind] += (k * dy / (dz**2))
                        A_mat[ind, ind_z0] += (k * dy / (dz ** 2)) #- (k * dy / (dz ** 2))
                        # A_mat[ind, ind] += (k * dy / (dz ** 2))
                    else:
                        pass


                elif j==0:
                    # bottom end boundary condition of wall
                    B_vect[ind] += (conv_B_vect_bottom[i]*(dz/dy)) - (q_rad_inner[i] * dz)#  -(k*T_g[i]*dz/(dy**2))
                    A_mat[ind, ind] += (k * dz / (dy**2)) + conv_coeff_P_bottom[i] * (dz/dy)
                    A_mat[ind, ind_y1] += (k * dz / (dy**2))
                    if i<mesh.num_axial-1 and i>0:
                        A_mat[ind, ind_z1] += (k * dy / (dz**2))
                        A_mat[ind, ind_z0] += (k * dy / (dz**2))
                    elif i==0:
                        # left boundary
                        B_vect[ind] += -(k * dy * T_f_cool[i] / (dz**2))
                        A_mat[ind, ind_z1] += (k * dy / (dz ** 2))
                        # A_mat[ind, ind] += (k * dy / (dz ** 2))
                    elif i==mesh.num_axial-1:
                        # right boundary
                        # B_vect[ind] += -(2*k * dy * T_f_cool[i] / (dz ** 2))
                        A_mat[ind, ind] += (k * dy / (dz**2))
                        A_mat[ind, ind_z0] += (k * dy / (dz ** 2))
                        # A_mat[ind, ind_z0] += (k * dy / (dz ** 2)) - (k * dy / (dz ** 2))
                        # A_mat[ind, ind] += (k * dy / (dz ** 2))
                    else:
                        pass


                else:
                    pass


        return A_mat,B_vect

    def wall_matrix_setup_fvm(self,mesh,q_rad_inner,q_rad_outer,q_conv1,q_conv2,T_f_cool,T_g,T_in,Tw_upper):
        """
        Heat transfer in solid wall. Finite Difference Discretization
        :param mesh: solid wall mesh resolved along length and thickness
        :param q_rad_inner:
        :param q_rad_outer:
        :param q_conv1:
        :param q_conv2:
        :return:
        """
        A_mat = np.zeros((mesh.N_surf,mesh.N_surf))
        B_vect = np.zeros(mesh.N_surf)
        for i in range(mesh.num_axial):
            for j in range(mesh.num_thickness):
                ind = (i*mesh.num_thickness)+j
                k = mesh.k_mat[ind]
                dy = mesh.d_y[ind]
                dz = mesh.d_z[ind]
                ind_z0 = ((i - 1) * mesh.num_thickness) + j
                ind_z1 = ((i + 1) * mesh.num_thickness) + j
                ind_y0 = (i * mesh.num_thickness) + j - 1
                ind_y1 = (i * mesh.num_thickness) + j + 1

                A_mat[ind, ind] += - 2*(k * dz / (dy)) - 2*(k * dy / (dz))
                if i<mesh.num_axial-1 and i>0 and j<mesh.num_thickness-1 and j>0:
                    A_mat[ind, ind_z1] += (k * dy / (dz))
                    A_mat[ind, ind_z0] += (k * dy / (dz))
                    A_mat[ind, ind_y1] += (k * dz / (dy))
                    A_mat[ind, ind_y0] += (k * dz / (dy))

                elif i==0 and j<mesh.num_thickness-1 and j>0:
                    # left boundary
                    B_vect[ind] += -(2*k * dy * T_in/ (dz))
                    A_mat[ind, ind_z1] += (k * dy / (dz))
                    A_mat[ind, ind_y1] += (k * dz / (dy))
                    A_mat[ind, ind_y0] += (k * dz / (dy))
                    A_mat[ind, ind] += -(k * dy / (dz))

                elif i==mesh.num_axial-1 and j<mesh.num_thickness-1 and j>0:
                    # right boundary
                    # B_vect[ind] += -(2*k * dy * T_f_cool[i] / (dz ** 2))
                    A_mat[ind, ind_z0] += (k * dy / (dz))#-(k * dy / (dz**2))
                    A_mat[ind, ind_y1] += (k * dz / (dy))
                    A_mat[ind, ind_y0] += (k * dz / (dy))
                    A_mat[ind, ind] += (k * dy / (dz))

                elif j==mesh.num_thickness-1:
                    # top end boundary condition of wall
                    B_vect[ind] += -(q_conv2[i]*(dz)*T_f_cool[i]) - (q_rad_outer[i] * dz) #+ (0.9*self.sigma * dz *((Tw_upper[i]**4)-(1000**4))) #-(k*T_f_cool[i]*dz/(dy**2))
                    A_mat[ind, ind] += (k * dz / (dy))-q_conv2[i] *(dz)
                    A_mat[ind, ind_y0] += (k * dz / (dy))
                    if i<mesh.num_axial-1 and i>0:
                        A_mat[ind, ind_z1] += (k * dy / (dz))
                        A_mat[ind, ind_z0] += (k * dy / (dz))
                    elif i==0:
                        # left boundary
                        B_vect[ind] += -(2*k * dy * T_in / (dz))
                        A_mat[ind, ind_z1] += (k * dy / (dz))
                        A_mat[ind, ind] += -(k * dy / (dz))
                    elif i==mesh.num_axial-1:
                        # right boundary
                        # B_vect[ind] += -(2*k * dy * T_f_cool[i] / (dz ** 2))
                        A_mat[ind, ind] += (k * dy / (dz))
                        # A_mat[ind, ind] += (k * dy / (dz**2))
                        A_mat[ind, ind_z0] += (k * dy / (dz)) #- (k * dy / (dz ** 2))
                        # A_mat[ind, ind] += (k * dy / (dz ** 2))
                    else:
                        pass


                elif j==0:
                    # bottom end boundary condition of wall
                    B_vect[ind] += +(q_conv1[i]*(dz)*T_g[i]) - (q_rad_inner[i] * dz)#  -(k*T_g[i]*dz/(dy**2))
                    A_mat[ind, ind] += (k * dz / (dy)) + q_conv1[1] * (dz)
                    A_mat[ind, ind_y1] += (k * dz / (dy))
                    if i<mesh.num_axial-1 and i>0:
                        A_mat[ind, ind_z1] += (2*k * dy / (dz))
                        A_mat[ind, ind_z0] += (k * dy / (dz))
                    elif i==0:
                        # left boundary
                        B_vect[ind] += -(2*k * dy * T_in / (dz))
                        A_mat[ind, ind_z1] += (k * dy / (dz))
                        A_mat[ind, ind] += -(k * dy / (dz))
                    elif i==mesh.num_axial-1:
                        # right boundary
                        # B_vect[ind] += -(2*k * dy * T_f_cool[i] / (dz ** 2))
                        A_mat[ind, ind] += (k * dy / (dz))
                        A_mat[ind, ind_z0] += (k * dy / (dz))
                        # A_mat[ind, ind_z0] += (k * dy / (dz ** 2)) - (k * dy / (dz ** 2))
                        # A_mat[ind, ind] += (k * dy / (dz ** 2))
                    else:
                        pass


                else:
                    pass


        return A_mat,B_vect

    def solver(self,mesh,q_rad_inner,q_rad_outer,q_conv2,T_f_cool,T_g,T_in,conv_coeff_P_bottom,conv_B_vect_bottom):
        A_mat, B_vect = self.wall_matrix_setup_fdm(mesh,q_rad_inner,q_rad_outer,q_conv2,T_f_cool,T_g,T_in,conv_coeff_P_bottom,conv_B_vect_bottom)
        Tw = scipy.linalg.solve(A_mat,B_vect)

        return Tw










