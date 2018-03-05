import math
import cmath
import functools
import operator
import numpy as np
import sympy
import mpmath
import scipy
import qutip 
import vpython
import random

def im():
    return complex(0, 1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

##################################################################################################################

def qubit_to_vector(qubit):
    return np.array([[qutip.expect(qutip.sigmax(), qubit).real],\
                     [qutip.expect(qutip.sigmay(), qubit).real],\
                     [qutip.expect(qutip.sigmaz(), qubit).real],\
                     [qutip.expect(qutip.identity(2), qubit).real]])

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

class Qubit:
    def __init__(self, state, n_fiber_points=50):
        self.state = state
        self.n_fiber_points = n_fiber_points

        self.color = vpython.vector(random.random(), random.random(), random.random())
        self.varrow = vpython.arrow(pos=vpython.vector(0,0,0),\
                                    color=self.color,\
                                    shaftwidth=0.05)
        self.vbase = vpython.sphere(color=self.color,\
                                    radius=0.1,\
                                    opacity=0.7,\
                                    emissive=False)
        self.vfiber = vpython.curve(pos=[vpython.vector(0,0,0) for i in range(self.n_fiber_points)],\
                                    color=self.color)

    def spin_axis(self):
        return [qutip.expect(qutip.sigmax(), self.state),\
                qutip.expect(qutip.sigmay(), self.state),\
                qutip.expect(qutip.sigmaz(), self.state)]

    def fibration(self):
        vector = qubit_to_vector(self.state)
        base = stereographic_projection(normalize(vector))

        circle = np.linspace(0, 2*math.pi, num=self.n_fiber_points)
        fiber_points = []
        for angle in circle:
            transformation = np.array([[math.cos(angle),-1*math.sin(angle),0,0],\
                                       [math.sin(angle), math.cos(angle),0,0],\
                                       [0,0,math.cos(angle),-1*math.sin(angle)],\
                                       [0,0,math.sin(angle),math.cos(angle)]])
            fiber_points.append(np.real(stereographic_projection(normalize(np.dot(transformation, vector)))))
        return base, fiber_points

    def visualize(self):
        self.varrow.axis = vpython.vector(*self.spin_axis())
        base, fiber_points = self.fibration()
        self.vbase.pos = vpython.vector(*base)
        for i in range(self.n_fiber_points):
            if not isinstance(fiber_points[i], int):
                self.vfiber.modify(i, pos=vpython.vector(*fiber_points[i]),\
                                      color=self.color)

    def evolve(self, operator, inverse=True, dt=0.005):
        unitary = (-2*math.pi*im()*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state = unitary*self.state*unitary.dag()

##################################################################################################################

vsphere = vpython.sphere(pos=vpython.vector(0,0,0),\
                         radius=1.0,\
                         color=vpython.color.blue,\
                         opacity=0.3)

qubit = Qubit(qutip.rand_ket(2).ptrace(0))

def keyboard(event):
    global qubit
    key = event.key
    if key == "a":
        qubit.evolve(qutip.sigmax(), inverse=True)
    elif key == "d":
        qubit.evolve(qutip.sigmax(), inverse=False)
    elif key == "s":
        qubit.evolve(qutip.sigmaz(), inverse=True)
    elif key == "w":
        qubit.evolve(qutip.sigmaz(), inverse=False)
    elif key == "z":
        qubit.evolve(qutip.sigmay(), inverse=True)
    elif key == "x":
        qubit.evolve(qutip.sigmay(), inverse=False)
vpython.scene.bind('keydown', keyboard)

while True:
    vpython.rate(100)
    qubit.visualize()
