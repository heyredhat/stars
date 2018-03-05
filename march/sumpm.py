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
        return [complex(0,0) for i in range(len(polynomial)-1)]
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
    def __init__(self, state=None, center=vpython.vector(0,0,0), radius=1, color=vpython.color.blue, star_color=vpython.color.white):
        self.state = state
        self.center = center
        self.radius = radius
        self.color = color
        self.star_color = star_color

        self.vsphere = vpython.sphere(pos=self.center,\
                                      radius=self.radius,\
                                      color=self.color,\
                                      opacity=0.4)
        self.varrow = vpython.arrow(pos=self.center,\
                                    color=self.color,\
                                    shaftwidth=0.05,\
                                    emissive=True)
        self.vbase = vpython.sphere(color=self.star_color,\
                                    radius=0.1*self.radius,\
                                    opacity=0.7,\
                                    emissive=True)

    def spin_axis(self):
        return [qutip.expect(qutip.sigmax(), self.state),\
                qutip.expect(qutip.sigmay(), self.state),\
                qutip.expect(qutip.sigmaz(), self.state)]

    def visualize(self):
        self.vsphere.radius = self.radius
        self.vbase.radius = 0.1*self.radius
        spin_axis = self.spin_axis()
        self.varrow.pos = self.center
        self.varrow.axis = vpython.vector(*spin_axis)*self.radius
        self.vbase.pos = vpython.vector(*spin_axis)*self.radius+self.center

    def evolve(self, operator, inverse=False, dt=0.005):
        unitary = (-2*math.pi*im()*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state = unitary*self.state

##################################################################################################################

class MajoranaSphere:
    def __init__(self, n, state, center, color):
        self.n = n
        self.state = state
        self.center = center
        self.color = color

        self.vsphere = vpython.sphere(pos=self.center,\
                                      radius=1.0,\
                                      color=self.color,\
                                      opacity=0.4)
        self.varrow = vpython.arrow(pos=self.center,\
                                    color=self.color,\
                                    shaftwidth=0.05,\
                                    emissive=True)
        self.vstars = [vpython.sphere(radius=0.1, emissive=True) for i in range(self.n-1)]

        self.a_choices = [Qubit(center=self.center+vpython.vector(-3,1.5,0), star_color=vpython.color.yellow),\
                          Qubit(center=self.center+vpython.vector(-1,1.5,0), star_color=vpython.color.yellow),\
                          Qubit(center=self.center+vpython.vector(1,1.5,0), star_color=vpython.color.yellow),\
                          Qubit(center=self.center+vpython.vector(3,1.5,0), star_color=vpython.color.yellow)]
        self.b_choices = [Qubit(center=self.center+vpython.vector(-3,1.5,0), star_color=vpython.color.orange),\
                          Qubit(center=self.center+vpython.vector(-1,1.5,0), star_color=vpython.color.orange),\
                          Qubit(center=self.center+vpython.vector(1,1.5,0), star_color=vpython.color.orange),\
                          Qubit(center=self.center+vpython.vector(3,1.5,0), star_color=vpython.color.orange)]

    def refract(self):
        self.state.dims = [[4],[1]]
        eigenvalues, eigenvectors = self.state.ptrace(0).eigenstates()
        for i in range(self.n):
            eigenvectors[i].dims = [[2,2], [1,1]]
            a = eigenvectors[i].ptrace(0)
            b = eigenvectors[i].ptrace(1)
            self.a_choices[i].state = a
            self.a_choices[i].radius = eigenvalues[i]
            self.b_choices[i].state = b
            self.b_choices[i].radius = eigenvalues[i]

    def spin_axis(self):
        spin = (self.n-1.)/2.
        X, Y, Z = qutip.jmat(spin)
        self.state.dims = [[self.n], [1]]
        spin_axis = [qutip.expect(X, self.state),\
                     qutip.expect(Y, self.state),\
                     qutip.expect(Z, self.state)]
        return spin_axis

    def visualize(self):
        spin_axis = self.spin_axis()
        self.varrow.pos = self.center
        self.varrow.axis = vpython.vector(*spin_axis)
        stars_xyz = q_SurfaceXYZ(self.state)
        for i in range(self.n-1):
            self.vstars[i].pos = vpython.vector(*stars_xyz[i])+self.center
        for choice in self.a_choices:
            choice.visualize()
        for choice in self.b_choices:
            choice.visualize()

##################################################################################################################

n_qubits = 2
qubits = [Qubit(state=qutip.rand_ket(2), center=vpython.vector(-1,0,0), color=vpython.color.blue),\
          Qubit(state=qutip.rand_ket(2), center=vpython.vector(1,0,0), color=vpython.color.red)]

def symmeterize(a, b):
    return 0.5*(qutip.tensor(a,b) + qutip.tensor(b,a))

alice, bob = qubits
god = MajoranaSphere(4, symmeterize(alice.state, bob.state), vpython.vector(0,1.5,0), vpython.color.green)

##################################################################################################################

vpython.scene.height = 600
vpython.scene.width = 800
vpython.scene.camera.forward = vpython.vector(0,1.5,0)

qubit_selected = 0
def keyboard(event):
    global n_qubits
    global qubits
    global qubit_selected
    key = event.key
    qubit = qubits[qubit_selected]
    if key.isdigit():
        i = int(key)
        if i < n_qubits:
            qubit_selected = i
    if key == "a":
        qubits[qubit_selected].evolve(qutip.sigmax(), inverse=True)
    elif key == "d":
        qubits[qubit_selected].evolve(qutip.sigmax(), inverse=False)
    elif key == "s":
        qubits[qubit_selected].evolve(qutip.sigmaz(), inverse=True)
    elif key == "w":
        qubits[qubit_selected].evolve(qutip.sigmaz(), inverse=False)
    elif key == "z":
        qubits[qubit_selected].evolve(qutip.sigmay(), inverse=True)
    elif key == "x":
        qubits[qubit_selected].evolve(qutip.sigmay(), inverse=False)
vpython.scene.bind('keydown', keyboard)

while True:
    vpython.rate(100)
    alice, bob = qubits
    god.state = symmeterize(alice.state, bob.state)
    god.refract()
    god.visualize()
    for qubit in qubits:
        qubit.visualize()