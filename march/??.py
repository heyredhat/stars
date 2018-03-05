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
    def __init__(self, state, field, n_fiber_points=50, emissive=False):
        self.state = state
        self.field = field 

        self.n_fiber_points = n_fiber_points
        self.emissive = emissive
        self.color = vpython.vector(random.random(), random.random(), random.random())

        self.varrow = vpython.arrow(pos=vpython.vector(0,0,0),\
                                    color=self.color,\
                                    shaftwidth=0.05,\
                                    emissive=self.emissive)
        self.vbase = vpython.sphere(color=self.color,\
                                    radius=0.1,\
                                    opacity=0.7,\
                                    emissive=self.emissive)
        self.vfiber = vpython.curve(pos=[vpython.vector(0,0,0) for i in range(self.n_fiber_points)],\
                                    color=self.color,\
                                    emissive=self.emissive)

    def spin_axis(self):
        return [qutip.expect(qutip.sigmax(), self.state).real,\
                qutip.expect(qutip.sigmay(), self.state).real,\
                qutip.expect(qutip.sigmaz(), self.state).real]

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
        if not isinstance(base, int):
            self.vbase.pos = vpython.vector(*base)
            for i in range(self.n_fiber_points):
                if not isinstance(fiber_points[i], int):
                    self.vfiber.modify(i, pos=vpython.vector(*fiber_points[i]),\
                                          color=self.color)

    def evolve(self, operator, inverse=False, dt=0.0001):
        self.field.evolve(self, operator, inverse=inverse, dt=dt)

##################################################################################################################

class Field:
    def __init__(self):
        self.qubit = Qubit(qutip.rand_ket(2).ptrace(0), self)
        self.photon = Qubit(qutip.rand_ket(2).ptrace(0), self, emissive=True)

        self.charge = 1
        self.energy = qutip.identity(2)

    def evolve(self, whom, operator, inverse=False, dt=0.0001):
        if whom == self.qubit:
            def vectorize(operator):
                op_x, op_y, op_z, op_t = qubit_to_vector(operator).T[0]
                X = vector_to_qubit(np.array([[op_x],[0],[0],[0]]))
                Y = vector_to_qubit(np.array([[0],[op_y],[0],[0]]))
                Z = vector_to_qubit(np.array([[0],[0],[op_z],[0]]))
                T = vector_to_qubit(np.array([[0],[0],[0],[op_t]]))
                return [X, Y, Z, T]
        
            def get_phase(state):
                state_matrix = state.full()
                alpha, beta = state_matrix[0,1], state_matrix[1,0]
                alpha_r, alpha_th = cmath.polar(alpha)
                beta_r, beta_th = cmath.polar(beta)
                return alpha_th-beta_th

            vectorized_qubit = vectorize(self.qubit.state)
            old_phases = [get_phase(vec_qub) for vec_qub in vectorized_qubit]

            vectorized_operator = vectorize(operator)
            vectorized_unitaries = [(-2*math.pi*im()*op*dt).expm() for op in vectorized_operator]
            if inverse:
                vectorized_unitaries = [unitary.dag() for unitary in vectorized_unitaries]

            new_XYZTstates = [op*self.qubit.state for op in vectorized_unitaries]
            new_phases = [get_phase(XYZTstate) for XYZTstate in new_XYZTstates]
        
            phase_deltas = [(new_phases[i]-old_phases[i])/2. for i in range(4)]
        
            total_unitary = (-2*math.pi*im()*operator*dt).expm()
            if inverse:
                total_unitary = total_unitary.dag()
            old_state = self.qubit.state
            new_state = total_unitary*self.qubit.state*total_unitary.dag()

            old_total_phase = get_phase(old_state)
            new_total_phase = get_phase(new_state)
            total_phase_delta = (new_total_phase-old_total_phase)/2

            old_vector = qubit_to_vector(old_state)
            new_vector = qubit_to_vector(new_state)

            vector_delta = new_vector-old_vector
            vector_delta = vector_delta.T[0].tolist()

            delta = []
            for i in range(4):
                if vector_delta[i] != 0:
                    delta.append([phase_deltas[i]/vector_delta[i]])
                else:
                    delta.append([0])
            delta = vector_to_qubit((1./self.charge)*np.array(delta))

            self.photon.state += delta
            self.qubit.state = new_state#*cmath.exp(im()*(total_phase_delta))
            self.energy = (self.energy-self.charge*self.photon.state) 

    def visualize(self):
        self.qubit.visualize()
        self.photon.visualize()


##################################################################################################################

vpython.scene.height = 600
vpython.scene.width = 800

vsphere = vpython.sphere(pos=vpython.vector(0,0,0),\
                         radius=1.0,\
                         color=vpython.color.blue,\
                         opacity=0.3)

field = Field()

def keyboard(event):
    global field
    key = event.key
    qubit = field.qubit
    if key == "a":
        #field.energy = 0.5*field.energy+qutip.sigmax()
        field.energy = field.energy-0.5*qutip.sigmax()
        #qubit.evolve(qutip.sigmax(), inverse=True)
    elif key == "d":
        #field.energy = 0.5*field.energy*qutip.sigmax()
        field.energy = field.energy+0.5*qutip.sigmax()
        #qubit.evolve(qutip.sigmax(), inverse=False)
    elif key == "s":
        #field.energy = 0.5*field.energy*qutip.sigmaz()
        field.energy = field.energy-0.5*qutip.sigmaz()
        #qubit.evolve(qutip.sigmaz(), inverse=True)
    elif key == "w":
        #field.energy = 0.5*field.energy*qutip.sigmaz()
        field.energy = field.energy+0.5*qutip.sigmaz()
        #qubit.evolve(qutip.sigmaz(), inverse=False)
    elif key == "z":
        #field.energy = 0.5*field.energy*qutip.sigmay()
        field.energy = field.energy-0.5*qutip.sigmay()
        #qubit.evolve(qutip.sigmay(), inverse=True)
    elif key == "x":
        #field.energy = 0.5*field.energy*qutip.sigmay()
        field.energy = field.energy+0.5*qutip.sigmay()
        #qubit.evolve(qutip.sigmay(), inverse=False)
    elif key == "p":
        field.qubit.state = qutip.rand_ket(2).ptrace(0)
        field.photon.state = qutip.rand_ket(2).ptrace(0)
        field.energy = qutip.identity(2)
    field.energy = field.energy
vpython.scene.bind('keydown', keyboard)

while True:
    vpython.rate(100)
    field.qubit.evolve(field.energy)
    field.visualize()
