import os
import sys
import math
import qutip
import cmath
import sympy
import scipy
import mpmath
import pickle
import random
import crayons
import inspect
import operator
import threading
import itertools
import functools
import qutip as qt
import numpy as np
import vpython as vp

##################################################################################################################

def im():
    return complex(0, 1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

##################################################################################################################

def c_xyz(c):
    if c == float('inf'):
        return [0,0,1]
    x = c.real
    y = c.imag
    return [-1*(2*x)/(1.+(x**2)+(y**2)),\
            (2*y)/(1.+(x**2)+(y**2)),\
            (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]

def xyz_c(xyz):
    x, y, z = -1*xyz[0], xyz[1], xyz[2]
    if z == 1:
        return float('inf') 
    else:
        return complex(x/(1-z), y/(1-z))

def polynomial_v(polynomial):
    coordinates = [polynomial[i]/(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) for i in range(len(polynomial))]
    return np.array(coordinates)

def combos(a,b):
    f = math.factorial
    return f(a) / f(b) / f(a-b)

def v_polynomial(v):
    polynomial = v.tolist()
    return [(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) * polynomial[i] for i in range(len(polynomial))] 

def C_polynomial(roots):
    s = sympy.symbols("s")
    polynomial = sympy.Poly(functools.reduce(lambda a, b: a*b, [s-np.conjugate(root) for root in roots]), domain="CC")
    return [complex(c) for c in polynomial.coeffs()]

def polynomial_C(polynomial):
    try:
        roots = [np.conjugate(complex(root)) for root in mpmath.polyroots(polynomial)]
    except:
        return [float('Inf') for i in range(len(polynomial)-1)]
    return roots

def C_v(roots):
    return polynomial_v(C_polynomial(roots))

def v_C(v):
    return polynomial_C(v_polynomial(v))

def v_SurfaceXYZ(v):
    return [c_xyz(c) for c in v_C(v)]

def SurfaceXYZ_v(XYZ):
    return C_v([xyz_c(xyz) for xyz in XYZ])

def q_SurfaceXYZ(q):
    return v_SurfaceXYZ(q.full().T[0])

def SurfaceXYZ_q(XYZ):
    return Qobj(C_v([xyz_c(xyz) for xyz in XYZ]))

##################################################################################################################

class MajoranaSphere:
    def __init__(self, state,\
                       center=vp.vector(0,0,0),\
                       radius=1,\
                       tag=None,\
                       sphere_color=vp.color.blue,\
                       star_color=vp.color.white,\
                       arrow_color=None,\
                       show_sphere=True,\
                       show_stars=True,\
                       show_arrow=True,\
                       show_tag=True,\
                       show_tag_on_stars=False):
        self.n = state.shape[0]
        self.state = state
        self.center = center
        self.radius = radius
        self.tag = tag
        self.sphere_color = sphere_color
        self.star_color = star_color
        self.arrow_color = self.sphere_color if arrow_color == None else arrow_color
        self.show_sphere = show_sphere
        self.show_stars = show_stars
        self.show_arrow = show_arrow
        self.show_tag = show_tag
        self.show_tag_on_stars= show_tag_on_stars

        self.vsphere = vp.sphere(visible=False) 
        self.varrow = vp.arrow(visible=False)
        self.vstars = [vp.sphere(visible=False) for i in range(self.n-1)] 
        self.vtag = vp.text(text="e", visible=False) 
        self.vtags = [vp.text(text="e", visible=False) for i in range(self.n-1)]
    
    def display_visuals(self):
        if self.show_sphere:
            self.vsphere.pos = self.center
            self.vsphere.radius = self.radius
            self.vsphere.color = self.sphere_color
            self.vsphere.opacity = 0.3
            self.vsphere.visible = True
        else:
            self.vsphere.visible = False
        self.state.dims = [[self.n],[1]]
        stars_xyz = q_SurfaceXYZ(self.state) if (self.show_stars or (self.tag and self.show_tag_on_stars)) else None
        if self.show_stars:
            for i, vstar in enumerate(self.vstars):
                vstar.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vstar.radius = 0.1*self.radius
                vstar.color = self.star_color
                vstar.opacity = 0.5
                vstar.emissive = True
                vstar.visible = True
        else:
            for vstar in self.vstars:
                vstar.visible = False
        spin_axis = self.spin_axis() if (self.show_arrow or (self.tag and self.show_tag)) else None
        if self.show_arrow:
            self.varrow.pos = self.center
            self.varrow.axis = self.radius*vp.vector(*spin_axis)
            self.varrow.color = self.arrow_color
            self.varrow.shaftwidth = 0.05
            self.varrow.emissive = True
            self.varrow.visible = True
        else:
            self.varrow.visible = False
        if self.tag and self.show_tag:
            self.vtag.text = self.tag
            self.vtag.pos = self.radius*vp.vector(*spin_axis) + self.center
            self.vtag.align = "center"
            self.vtag.color = self.arrow_color
            self.vtag.height = 0.2*self.radius
            self.vtag.visible = True
        else:
            self.vtag.visible = False
        if self.tag and self.show_tag_on_stars:
            for i, vtag in enumerate(self.vtags):
                vtag.text = self.tag
                vtag.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vtag.align = "center"
                vtag.color = self.arrow_color
                vtag.height = 0.2*self.radius
                vtag.visible = True
        else:
            for vtag in self.vtags:
                vtag.visible = False

    def spin_operators(self):
        spin = (self.n-1.)/2.
        return {"X": qutip.jmat(spin, "x"),\
                "Y": qutip.jmat(spin, "y"),\
                "Z": qutip.jmat(spin, "z"),\
                "+": qutip.jmat(spin, "+"),\
                "-": qutip.jmat(spin, "-")}

    def spin_axis(self):
        spin_ops = self.spin_operators()
        self.state.dims = [[self.n], [1]]
        spin_axis = np.array([[qt.expect(spin_ops["X"], self.state)],\
                              [qt.expect(spin_ops["Y"], self.state)],\
                              [qt.expect(spin_ops["Z"], self.state)],\
                              [qt.expect(qt.identity(self.n), self.state)]])
        return normalize(spin_axis[:-1])

    def evolve(self, operator, inverse=False, dt=0.01):
        unitary = (-2*math.pi*im()*dt*operator).expm()
        if inverse:
            unitary = unitary.dag()
        self.state.dims = [[self.n],[1]]
        self.state = unitary*self.state

sphere = MajoranaSphere(qt.rand_ket(4), sphere_color=vp.vector(*np.random.rand(3)),\
                            star_color=vp.vector(*np.random.rand(3)),\
                            tag="hello")

vp.rate(100)
while True:
    sphere.display_visuals()