
import os
import sys
import time
import math
import qutip
import cmath
import sympy
import scipy
import ephem
import ephem.stars
import mpmath
#import pyglet
import pickle
import random
import crayons
import inspect
import geopy
import datetime
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

def str_to_bool(s):
    if s == "True" or s == "true":
        return True
    elif s == "False" or s == "false":
        return False
    else:
        return None

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def totalset(iterable):
    ps = list(powerset(iterable))
    total = []
    for p in ps:
        total.extend(list(itertools.permutations(p)))
    return total

def relations(iterable):
    rel = []
    for x in totalset(iterable):
        if len(x) > 1:
            rel.append(list(x))
    return rel

def symmeterize(pieces, labels=None):
    n = len(pieces)
    if labels == None:
        labels = list(range(n))
    unique_labels = list(set(labels))
    label_counts = [0 for i in range(len(unique_labels))]
    label_permutations = itertools.permutations(labels, n)
    for permutation in label_permutations:
        for i in range(len(unique_labels)):
            label_counts[i] += list(permutation).count(unique_labels[i])
    normalization = 1./math.sqrt(functools.reduce(operator.mul, [math.factorial(count) for count in label_counts], 1)/math.factorial(n))    
    permutations = list(itertools.permutations(pieces, n))
    perm_states = []
    for permutation in permutations:
        perm_state = permutation[0]
        for state in permutation[1:]:
            perm_state = qt.tensor(perm_state, state)
        perm_state.dims = [[perm_state.shape[0]],[1]]
        perm_states.append(perm_state)
    tensor_sum = sum(perm_states)
    return normalization*tensor_sum

def perm_parity(lst):
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity    

def antisymmeterize(pieces):
    n = len(pieces)
    normalization = 1./math.sqrt(math.factorial(n))
    permutations = list(itertools.permutations(pieces, n))
    int_permutations = list(itertools.permutations(list(range(n)), n))
    perm_states = []
    for i in range(len(permutations)):
        permutation = permutations[i]
        perm_state = permutation[0]
        for state in permutation[1:]:
            perm_state = qt.tensor(perm_state, state)
        perm_state = perm_state*perm_parity(list(int_permutations[i]))
        perm_state.dims = [[perm_state.shape[0]],[1]]
        perm_states.append(perm_state)
    tensor_sum = sum(perm_states)
    return normalization*tensor_sum

def direct_sum(pieces):
    return qt.Qobj(normalize(np.concatenate([piece.full().T[0] for piece in pieces])))

##################################################################################################################

def spin_axis(state):
    n = state.shape[0]
    spin = (n-1.)/2.
    if state.isket:
        state.dims = [[n], [1]]
    else:
        state.dims = [[n],[n]]
    spin_axis = np.array([[qt.expect(qt.jmat(spin,"x"), state)],\
                          [qt.expect(qt.jmat(spin,"y"), state)],\
                          [qt.expect(qt.jmat(spin,"z"), state)],\
                          [qt.expect(qt.identity(n), state)]])
    return normalize(spin_axis[:-1])

##################################################################################################################

def qubit_to_vector(qubit):
    return np.array([[qutip.expect(qutip.sigmax(), qubit)],\
                     [qutip.expect(qutip.sigmay(), qubit)],\
                     [qutip.expect(qutip.sigmaz(), qubit)],\
                     [qutip.expect(qutip.identity(2), qubit)]])

def vector_to_qubit(vector):
    x, y, z, t = vector.T[0]
    return (1./2)*(x*qutip.sigmax() + y*qutip.sigmay() + z*qutip.sigmaz() + t*qutip.identity(2))

# n dimensions to n-1-sphere
def vector_to_angles(xyzs):
    coordinates = [c.real for c in xyzs.T[0].tolist()[::-1]]
    n = len(coordinates)-1
    r = math.sqrt(sum([x**2 for x in coordinates]))
    angles = []
    for i in range(n):
        if i != n-1:
            angle = coordinates[i]
            divisor = math.sqrt(sum([coordinates[j]**2 for j in range(i,n+1)]))
            if divisor == 0:
                angles.append(0)
            else: 
                angles.append(math.acos(angle/divisor))
        else:
            angle = None
            if math.sqrt(coordinates[-1]**2 + coordinates[-2]**2) != 0:
                angle = coordinates[-2]/math.sqrt(coordinates[-1]**2 + coordinates[-2]**2)
                angle = math.acos(angle)
                if coordinates[-1] < 0:
                    angle = 2*math.pi - angle
            else:
                angle = 0
            angles.append(angle)
    return angles, r

# n-sphere to n+1 dimensions
# note: last angle ranges to 2pi, all previous to pi
def angles_to_vector(angles, r=1):
    n = len(angles)
    coordinates = []
    for i in range(n+1):
        coordinate = r
        coordinate *= functools.reduce(operator.mul,\
                        [math.sin(angles[j]) for j in range(i)], 1)
        if i != n:
            coordinate *= math.cos(angles[i])
        coordinates.append([coordinate])
    return np.array(coordinates[::-1])

# n-sphere in n+1 dimensions to hyperplane in n dimensions plus infinity
# projection from [1, 0, 0, 0, ...]
def stereographic_projection(xyz):
    coordinates = xyz.T[0]
    if coordinates[-1] == 1:
        return len(coordinates)
    else:
        return np.array([[coordinate/(1-coordinates[-1])] for coordinate in coordinates[:-1]])

# from hyperplane in n dimensions plus infinity to n-sphere in n+1 dimensions
def inverse_stereographic_projectionAll(xyz):
    if isinstance(xyz, int):
        n = xyz
        return np.array([[1]]+[[0]]*(n-1))
    coordinates = xyz.T[0][::-1]
    s = sum([coordinate**2 for coordinate in coordinates])
    sphere = [[(s - 1)/(s + 1)]]
    for coordinate in coordinates:
        sphere.append([(2*coordinate)/(s + 1)])
    return np.array(sphere[::-1])

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
    return qt.Qobj(C_v([xyz_c(xyz) for xyz in XYZ]))

##################################################################################################################

class MajoranaSphere:
    def __init__(self, state,\
                       center=None,\
                       radius=1,\
                       sphere_color=None,\
                       star_color=None,\
                       arrow_color=None,\
                       show_sphere=True,\
                       show_stars=True,\
                       show_arrow=True,\
                       show_tag=False,\
                       show_tag_on_stars=False,\
                       star_radius=0.1,\
                       tag="e"):
        self.star_radius=star_radius
        self.n = state.shape[0]
        self.state = state
        self.energy = None
        self.evolving = False
        self.center = center if center else vp.vector(0,0,0)
        self.radius = radius
        self.sphere_color = sphere_color if sphere_color != None else vp.color.blue
        self.star_color = star_color if star_color != None else vp.color.white
        self.arrow_color = self.sphere_color if arrow_color == None else arrow_color
        self.show_sphere = show_sphere
        self.show_stars = show_stars
        self.show_arrow = show_arrow
        self.show_tag = show_tag
        self.show_tag_on_stars = show_tag_on_stars
        self.tag = tag

        self.vsphere = vp.sphere(radius=self.radius,\
                                 color=self.sphere_color,\
                                 opacity=0.3,\
                                 visible=self.show_sphere) 
        self.varrow = vp.arrow(color=self.arrow_color,\
                               shaftwidth=0.05,\
                               emissive=True,\
                               visible=self.show_arrow)
        self.vstars = [vp.sphere(radius=self.star_radius,\
                                 color=self.star_color,\
                                 opacity=0.7,\
                                 emissive=True,\
                                 visible=self.show_stars) for i in range(self.n-1)] 
        self.vtag = vp.text(text=self.tag,\
                            color=self.arrow_color,\
                            align="center",\
                            height=0.2*self.radius,\
                            emissive=True,\
                            visible=self.show_tag) 
        self.vtags = [vp.text(text=self.tag,\
                              color=self.arrow_color,\
                              align="center",\
                              height=0.125*self.radius,\
                              visible=self.show_tag_on_stars) for i in range(self.n-1)]

    def keyboard(self, event):
        key = event.key
        spin_ops = self.spin_operators()
        if key == "a":
            self.evolve(spin_ops['X'], inverse=True)
        elif key == "d":
            self.evolve(spin_ops['X'], inverse=False)
        elif key == "s":
            self.evolve(spin_ops['Z'], inverse=True)
        elif key == "w":
            self.evolve(spin_ops['Z'], inverse=False)
        elif key == "z":
            self.evolve(spin_ops['Y'], inverse=True)
        elif key == "x":    
            self.evolve(spin_ops['Y'], inverse=False)
        elif key == "e":
            if self.evolving:
                self.evolving = False
            else:
                self.evolving = True
        elif key == "r":
            self.energy = qt.rand_herm(self.n)
        elif key == "t":
            self.state = qt.rand_ket(self.n)

    def display_visuals(self):
        if self.evolving:
            self.evolve(self.energy)
        if self.show_sphere:
            self.vsphere.pos = self.center
            self.vsphere.radius = self.radius
            self.vsphere.color = self.sphere_color
            self.vsphere.opacity = 0.3
        self.state.dims = [[self.n],[1]]
        stars_xyz = q_SurfaceXYZ(self.state) if (self.show_stars or (self.tag and self.show_tag_on_stars)) else None
        if self.show_stars:
            for i, vstar in enumerate(self.vstars):
                vstar.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vstar.radius = self.star_radius
                #vstar.color = self.star_color
                vstar.opacity = 0.7
                vstar.emissive = True
        spin_axis = self.spin_axis() if (self.show_arrow or (self.tag and self.show_tag)) else None
        if self.show_arrow:
            self.varrow.pos = self.center
            self.varrow.axis = self.radius*vp.vector(*spin_axis)
            self.varrow.color = self.arrow_color
            self.varrow.shaftwidth = 0.05
            self.varrow.emissive = True
        if self.tag and self.show_tag:
            self.vtag.pos = self.radius*vp.vector(*spin_axis) + self.center
            self.vtag.color = self.arrow_color
        if self.tag and self.show_tag_on_stars:
            for i, vtag in enumerate(self.vtags):
                vtag.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vtag.color = self.arrow_color
    
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

    def invisible(self):
        self.vsphere.visible = False
        self.varrow.visible = False
        for vstar in self.vstars:
            vstar.visible = False
        self.vtag.visible = False
        for vtag in self.vtags:
            vtag.visible = False

    def visible(self):
        self.vsphere.visible = self.show_sphere
        self.varrow.visible = self.show_arrow
        for vstar in self.vstars:
            vstar.visible = self.show_stars
        self.vtag.visible = self.show_tag
        for vtag in self.vtags:
            vtag.visible = self.show_tag_on_stars

    def field_state(self):
        XYZ = q_SurfaceXYZ(self.state)
        qubits = [SurfaceXYZ_q([xyz]) for xyz in XYZ]
        return symmeterize(qubits)

vp.scene.height = 800
vp.scene.width = 800

state3d = qt.rand_ket(3)
sphere3d = MajoranaSphere(state3d, star_color=vp.color.red, arrow_color=vp.color.red)
sphere3d.energy = qt.rand_herm(3)
vp.scene.bind('keydown', sphere3d.keyboard)  

state4d = sphere3d.field_state()
sphere4d = MajoranaSphere(state4d, star_color=vp.color.orange, star_radius=0.08, arrow_color=vp.color.orange)

state8d = sphere4d.field_state()
sphere8d = MajoranaSphere(state8d, star_color=vp.color.yellow, star_radius=0.06, arrow_color=vp.color.yellow)

#state128d = sphere8d.field_state()
#sphere128d = MajoranaSphere(state128d, star_color=vp.color.green, star_radius=0.04, arrow_color=vp.color.green)

while True:
    sphere4d.state = sphere3d.field_state()
    sphere8d.state = sphere4d.field_state()
   # sphere128d.state = sphere8d.field_state()
    sphere3d.display_visuals()
    sphere4d.display_visuals()
    sphere8d.display_visuals()
    #sphere128d.display_visuals()

