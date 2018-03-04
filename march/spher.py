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

class Variable:    
    def __init__(self):
        self.value = None
        self.dependencies = []
        self.touched = 0

    def tie(self, other, transformation):
        self.dependencies.append({"other": other, "transformation": transformation})

    def plug(self, value):
        self.value = value
        if self.touched == 0:
            self.touched = 1
        elif self.touched == 1:
            self.touched = 0
        for dependency in self.dependencies:
            other, transformation = dependency["other"], dependency["transformation"]
            if other.touched != self.touched:
                other.plug(transformation(self.value))

    def __str__(self):
        return str(self.value)

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
    coordinates = xyzs.T[0].tolist()[::-1]
    n = len(coordinates)-1
    r = math.sqrt(sum([x**2 for x in xyzs]))
    angles = []
    for i in range(n):
        if i != n-1:
            angle = coordinates[i]
            divisor = math.sqrt(sum([coordinates[j]**2 for j in range(i,n+1)]))
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
    def __init__(self, color, radius=0.1, n_fiber_points=50):
        self.color = color
        self.radius = radius
        self.n_fiber_points = n_fiber_points

        self.state = Variable() # 2x2 Density Matrix
        self.vector = Variable() # spacetime 4-vector
        self.angles = Variable() # 3 angles of 3-sphere

        self.state.tie(self.vector, qubit_to_vector)
        self.vector.tie(self.state, vector_to_qubit)
        self.vector.tie(self.angles, vector_to_angles)
        self.angles.tie(self.vector, angles_to_vector)

        self.vbase = vpython.sphere(color=self.color,\
                                    radius=self.radius,\
                                    emissive=True,\
                                    opacity=0.7)
        self.varrow = vpython.arrow(pos=vpython.vector(0,0,0),\
                                    color=self.color,\
                                    shaftwidth = 0.06)
        self.vfiber = vpython.curve(pos=[vpython.vector(0,0,0) for i in range(self.n_fiber_points)],\
                                    color=self.color)

    def visualize(self):
        vpython.rate(100)
        x, y, z = [c.real for c in self.base().T[0].tolist()]
        self.vbase.pos = vpython.vector(x, y, z)
        self.varrow.axis = vpython.vector(x, y, z)
        fiber_points = self.fiber()
        for i in range(self.n_fiber_points):
            if not isinstance(fiber_points[i], int):
                self.vfiber.modify(i, pos=vpython.vector(*fiber_points[i]))

    def base(self):
        return self.vector.value[:-1]

    def fiber(self):
        circle = np.linspace(0, 2*math.pi, num=self.n_fiber_points)
        fiber_points = []
        for angle in circle:
            transformation = np.array([[math.cos(angle),-1*math.sin(angle),0,0],\
                                       [math.sin(angle), math.cos(angle),0,0],\
                                       [0,0,math.cos(angle),-1*math.sin(angle)],\
                                       [0,0,math.sin(angle),math.cos(angle)]])
            fiber_points.append(stereographic_projection(normalize(np.dot(transformation, self.vector.value))))
        return fiber_points

    def evolve(self, operator, inverse=True, dt=0.006):
        unitary = (2*math.pi*im()*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state.plug(unitary*self.state.value*unitary.dag())

##################################################################################################################

n_qubits = 6
qubit_colors = [vpython.vector(random.random(), random.random(), random.random()) for i in range(n_qubits)]

qubits = [Qubit(qubit_colors[i]) for i in range(n_qubits)]
for qubit in qubits:
    qubit.state.plug(qutip.rand_herm(2))

active_qubit = 0
def keyboard(event):
    global qubits
    global active_qubit
    key = event.key
    if key.isdigit():
        i = int(key)
        if i < n_qubits:
            active_qubit = i
    elif key == "a":
        qubits[active_qubit].evolve(qutip.sigmax(), inverse=True)
    elif key == "d":
        qubits[active_qubit].evolve(qutip.sigmax(), inverse=False)
    elif key == "s":
        qubits[active_qubit].evolve(qutip.sigmaz(), inverse=True)
    elif key == "w":
        qubits[active_qubit].evolve(qutip.sigmaz(), inverse=False)
    elif key == "z":
        qubits[active_qubit].evolve(qutip.sigmay(), inverse=True)
    elif key == "x":
        qubits[active_qubit].evolve(qutip.sigmay(), inverse=False)
vpython.scene.bind('keydown', keyboard)

vsphere = vpython.sphere(pos=vpython.vector(0,0,0),\
                         radius=1.0,\
                         color=vpython.color.blue,\
                         opacity=0.3)

while True:
    for qubit in qubits:
        qubit.visualize()