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

class Sphere:
    def __init__(self, n_qubits, center=vpython.vector(0,0,0), radius=1.0, color=vpython.color.blue):
        self.n_qubits = n_qubits
        self.center = center
        self.radius = radius
        self.color = color

        self.state = qutip.rand_ket(2**self.n_qubits)
        self.energy = qutip.rand_herm(2**self.n_qubits)


        self.vsphere = vpython.sphere(pos=self.center,\
                                      radius=self.radius,\
                                      color=self.color,\
                                      opacity=0.3)

        qubit_colors = [vpython.vector(random.random(), random.random(), random.random())\
                            for i in range(self.n_qubits)]
        self.qubits = [Qubit(color=qubit_colors[i], parent=self, gauged=True) for i in range(self.n_qubits)]
        self.update()

    def update(self):
        self.state.dims = [[2]*self.n_qubits, [1]*self.n_qubits]
        for i in range(self.n_qubits):
            self.qubits[i].update(self.state.ptrace(i))

    def visualize(self):
        self.vsphere.pos = self.center
        self.vsphere.radius = self.radius
        self.vsphere.color = self.color
        for i in range(self.n_qubits):
            self.qubits[i].visualize()

    def evolve(self, operator=None, inverse=True, dt=0.006):
        if operator == None:
            operator = self.energy
        unitary = (-2*math.pi*im()*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state.dims = [[2**self.n_qubits],[1]]
        self.state = unitary*self.state
        self.update()

    def destroy(self):
        self.vsphere.visible = False
        for qubit in self.qubits:
            qubit.destroy()

##################################################################################################################

class Qubit:
    def __init__(self, color=vpython.color.white, n_fiber_points=50, parent=None, gauged=False):
        self.color = color
        self.n_fiber_points = n_fiber_points
        self.parent = parent
        self.gauged = gauged

        self.state = Variable() # 2x2 Density Matrix
        self.vector = Variable() # spacetime 4-vector
        self.angles = Variable() # 3 angles of 3-sphere

        self.state.tie(self.vector, qubit_to_vector)
        self.vector.tie(self.state, vector_to_qubit)
        self.vector.tie(self.angles, vector_to_angles)
        self.angles.tie(self.vector, angles_to_vector)

        self.vbase = vpython.sphere(color=self.color,\
                                    radius=0.1,\
                                    opacity=0.7,\
                                    emissive=False)
        self.varrow = vpython.arrow(pos=vpython.vector(0,0,0),\
                                    color=self.color,\
                                    shaftwidth = 0.06)
        self.vfiber = vpython.curve(pos=[vpython.vector(0,0,0) for i in range(self.n_fiber_points)],\
                                    color=self.color)

        if self.gauged:
            self.gauge_charge = 30
            self.gauge_field = Qubit(color=self.color,\
                                     n_fiber_points=self.n_fiber_points,\
                                     gauged=False)
            self.gauge_field.vbase.emissive = True
            self.gauge_field.state.plug(qutip.rand_herm(2))

    def update(self, new_state):
        if self.gauged and self.angles.value != None:
            old_angles, r0 = self.angles.value
            new_angles, r1 = vector_to_angles(qubit_to_vector(new_state))
            angle_delta = new_angles[0]-old_angles[0]

            old_vector = self.vector.value
            new_vector = qubit_to_vector(new_state)
            vector_delta = new_vector-old_vector
            vector_delta = vector_delta.T[0].tolist()

            delta = []
            for i in range(4):
                if vector_delta[i] != 0:
                    delta.append([angle_delta/vector_delta[i]])
                else:
                    delta.append([0])
            delta = np.array(delta)
            self.gauge_field.vector.plug(normalize(self.gauge_field.vector.value + (1./self.gauge_charge)*delta))
        self.state.plug(new_state)

    def visualize(self):
        vpython.rate(100)
        x, y, z = [c.real for c in self.base().T[0].tolist()]
        self.vbase.pos = vpython.vector(x, y, z) 
        self.varrow.axis = vpython.vector(x, y, z)
        fiber_points = self.fiber()
        for i in range(self.n_fiber_points):
            if not isinstance(fiber_points[i], int):
                self.vfiber.modify(i, pos=vpython.vector(*fiber_points[i]))
        if self.gauged:
            self.gauge_field.visualize()

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
        unitary = (-2*math.pi*im()*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        if self.parent:
            i = self.parent.qubits.index(self)
            upgraded = None
            if i == 0:
                upgraded = unitary
            else:
                upgraded = qutip.identity(2)
            for j in range(1, self.parent.n_qubits):
                if j == i:
                    upgraded = qutip.tensor(upgraded, unitary)
                else:
                    upgraded = qutip.tensor(upgraded, qutip.identity(2))
            self.parent.state.dims = [[2**self.parent.n_qubits],[1]]
            upgraded.dims = [[2**self.parent.n_qubits],[2**self.parent.n_qubits]]
            self.parent.state = upgraded*self.parent.state
            self.parent.update()
        else:
            if inverse:
                unitary = unitary.dag()
            self.update(unitary*self.state.value*unitary.dag())

    def destroy(self):
        self.vbase.visible = False
        self.varrow.visible = False
        self.vfiber.visible = False
        if self.gauged:
            self.gauge_field.destroy()

##################################################################################################################

n_qubits = 1
sphere = Sphere(n_qubits)

##################################################################################################################

vpython.scene.height = 600
vpython.scene.width = 800

active_qubit = 0
evolution_on = False
def keyboard(event):
    global sphere
    global n_qubits
    global active_qubit
    global evolution_on
    key = event.key
    if key.isdigit():
        i = int(key)
        if i < sphere.n_qubits:
            active_qubit = i
    elif key == "a":
        sphere.qubits[active_qubit].evolve(qutip.sigmax(), inverse=True)
    elif key == "d":
        sphere.qubits[active_qubit].evolve(qutip.sigmax(), inverse=False)
    elif key == "s":
        sphere.qubits[active_qubit].evolve(qutip.sigmaz(), inverse=True)
    elif key == "w":
        sphere.qubits[active_qubit].evolve(qutip.sigmaz(), inverse=False)
    elif key == "z":
        sphere.qubits[active_qubit].evolve(qutip.sigmay(), inverse=True)
    elif key == "x":
        sphere.qubits[active_qubit].evolve(qutip.sigmay(), inverse=False)
    elif key == "i":
        if evolution_on:
            evolution_on = False
        else:
            evolution_on = True
    elif key == "o":
        sphere.state = qutip.rand_ket(2**sphere.n_qubits)
        sphere.update()
    elif key == "p":
        sphere.energy = qutip.rand_herm(2**sphere.n_qubits)
        sphere.update()
    elif key == "+":
        evolution_on = False
        sphere.destroy()
        n_qubits += 1
        sphere = Sphere(n_qubits)
        active_qubit = 0
    elif key == "-":
        evolution_on = False
        sphere.destroy()
        n_qubits -= 1
        sphere = Sphere(n_qubits)
        active_qubit = 0
vpython.scene.bind('keydown', keyboard)

##################################################################################################################

while True:
    if evolution_on:
        sphere.evolve()
    sphere.visualize()