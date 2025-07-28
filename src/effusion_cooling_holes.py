import math
import numpy as np

class effusion_cooling:
    def __init__(self):
        self.hole_dia = 0.0005 #m
        self.hole_spacing_axial = 3 * self.hole_dia #m
        self.hole_spacing_tangential = 2*self.hole_dia
        self.BR = 0.1 # blowing ratio

    def hole_def(self, L_start,L_end,L_liner, dia_cyl):
        L = L_end-L_start
        num_rows = int(L/self.hole_spacing_axial) # number of rows
        num_circum = int(math.pi*dia_cyl/self.hole_spacing_tangential) # number of holes around circumference per row
        area_hole = (math.pi/4)*(self.hole_dia**2)
        area_core = (math.pi/4)*(dia_cyl**2)
        num_holes = num_rows * num_circum
        num_holes_axial = num_circum*np.ones(num_rows)
        m_inj_frac = (0.99/num_rows)*np.ones(num_rows)
        inj_loc = np.linspace(L_start,L_end,num_rows)/L_liner
        m_c_by_m_h = self.BR*num_holes*area_hole/area_core
        Xi = m_c_by_m_h/(1+m_c_by_m_h)
        hole_dia_array = self.hole_dia*np.ones(num_rows)


        return Xi, m_inj_frac, inj_loc, num_holes_axial, hole_dia_array
