# TODO
#  1. Field
#  2. Embeddings
#  3. Dimensionality

###############################################################################################################

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
import itertools

###############################################################################################################

def combos(a,b):
    f = math.factorial
    return f(a) / f(b) / f(a-b)

def symmeterize(pieces, labels):
    n = len(pieces)
    unique_labels = set(labels)
    label_counts = [0 for i in range(len(unique_labels))]
    label_permutations = itertools.permutations(labels, n)
    for permutation in label_permutations:
        for i in range(len(unique_labels)):
            label_counts[i] += list(permutation).count(unique_labels[i])
    normalization = math.sqrt(math.factorial(functools.reduce(operator.mul, label_counts, 1))/math.factorial(n))    
    permutations = itertools.permutations(pieces, n)
    tensor_sum = sum([qutip.tensor(list(permutation)) for permutation in permutations])
    return normalization*tensor_sum

###############################################################################################################

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

###############################################################################################################

class Sphere:
    def __init__(self, x, y, z):
        self.center = [x, y, z]
        
        self.vsphere = None
        self.vspin_axis = None
        self.vstars = None
        
        self.state = Variable()
        self.vector = Variable()
        self.polynomial = Variable()
        self.roots = Variable()
        self.xyzs = Variable()
        self.angles = Variable()
        self.qubits = Variable()
        self.field = Variable()
          
        def state_to_vector(state):
            return state.full().T[0]
        self.state.tie(self.vector, state_to_vector)

        def vector_to_state(vector):
            return qutip.Qobj(vector)
        self.vector.tie(self.state, vector_to_state)

        def vector_to_polynomial(vector):
            polynomial = vector.tolist()
            return [(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) * polynomial[i] for i in range(len(polynomial))] 
        self.vector.tie(self.polynomial, vector_to_polynomial)

        def polynomial_to_vector(polynomial):
            coordinates = [polynomial[i]/(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) for i in range(len(polynomial))]
            return np.array(coordinates)
        self.polynomial.tie(self.vector, polynomial_to_vector)

        def polynomial_to_roots(polynomial):
            try:
                return [np.conjugate(complex(root)) for root in mpmath.polyroots(polynomial)]
            except:
                return [complex(0,0) for i in range(len(polynomial)-1)]
        self.polynomial.tie(self.roots, polynomial_to_roots)

        def roots_to_polynomial(roots):
            s = sympy.symbols("s")
            polynomial = sympy.Poly(functools.reduce(lambda a, b: a*b, [s-np.conjugate(root) for root in roots]), domain="CC")
            return [complex(c) for c in polynomial.coeffs()]
        self.roots.tie(self.polynomial, roots_to_polynomial)

        def roots_to_xyzs(roots):
            def root_to_xyz(root):
                if root == float('inf'):
                    return [0,0,1]
                x = root.real
                y = root.imag
                return [(2*x)/(1.+(x**2)+(y**2)),\
                        (2*y)/(1.+(x**2)+(y**2)),\
                        (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]
            return [root_to_xyz(root) for root in roots]
        self.roots.tie(self.xyzs, roots_to_xyzs)

        def xyzs_to_roots(xyzs):
            def xyz_to_root(xyz):
                x, y, z = xyz[0], xyz[1], xyz[2]
                if z == 1:
                    return float('inf') 
                else:
                    return complex(x/(1-z), y/(1-z))
            return [xyz_to_root(xyz) for xyz in xyzs]
        self.xyzs.tie(self.roots, xyzs_to_roots)
        
        # theta: colatitude with respect to z-axis
        # phi: longitude with respect to y-axis
                
        def xyzs_to_angles(xyzs):
            def xyz_to_angle(xyz):
                x, y, z = xyz[0], -1*xyz[1], -1*xyz[2]
                r = math.sqrt(x**2 + y**2 + z**2)
                latitude = math.asin(z/r)
                colatitude = (math.pi/2.)-latitude
                longitude = None
                if x > 0:
                    longitude = math.atan(y/x)
                elif y > 0:
                    longitude = math.atan(y/x) + math.pi
                else:
                    longitude = math.atan(y/x) - math.pi
                return (colatitude, longitude)
            return [xyz_to_angle(xyz) for xyz in xyzs]
        self.xyzs.tie(self.angles, xyzs_to_angles)
        
        def angles_to_xyzs(angles):
            def angle_to_xyz(angle):
                theta, phi = angle
                return [math.sin(theta)*math.cos(phi),\
                        -1*math.sin(theta)*math.sin(phi),\
                        -1*math.cos(theta)]
            return [angle_to_xyz(angle) for angle in angles]
        self.angles.tie(self.xyzs, angles_to_xyzs)
        
        def angles_to_qubits(angles):
            def angle_to_qubit(angle):
                theta, phi = angle
                qubit = [math.cos(theta/2.), math.sin(theta/2.)*cmath.exp(complex(0,1)*phi)]
                return qutip.Qobj(np.array(qubit))
            return [angle_to_qubit(angle) for angle in angles]
        self.angles.tie(self.qubits, angles_to_qubits)
        
        def qubits_to_angles(qubits):
            def qubit_to_angle(qubit):
                alpha, beta = qubit.full().T[0].tolist()
                alpha_r, alpha_theta = cmath.polar(alpha)
                beta_r, beta_theta = cmath.polar(beta)      
                theta = 2*math.acos(abs(alpha_r))
                phi = beta_theta-alpha_theta
                return (theta, phi)
            return [qubit_to_angle(qubit) for qubit in qubits]
        self.qubits.tie(self.angles, qubits_to_angles)
        
        def qubits_to_field(qubits):
            a = qutip.Qobj(np.array([[0,0],[1,0]]))
            b = qutip.Qobj(np.array([[1,0],[0,0]]))
            creation_operators = []
            annihilation_operators = []
            for qubit in qubits:
                alpha, beta = qubit.full().T[0].tolist()
                creation_operator = alpha*a + beta*b
                annihilation_operator = creation_operator.dag()
                creation_operators.append(creation_operator)
                annihilation_operators.append(annihilation_operator)
            
        self.qubits.tie(self.field, qubits_to_field)
        
        def field_to_qubits(qubits):
            pass
        self.field.tie(self.qubits, field_to_qubits)
        
    def visualize(self):
        if self.state.value != None:
            if self.vsphere == None:
                self.vsphere = vpython.sphere()
            if self.vspin_axis == None:
                self.vspin_axis = vpython.arrow()
            if self.vstars == None:
                self.vstars = [vpython.sphere() for i in range(len(self.xyzs.value))]
            if len(self.vstars) < len(self.xyzs.value):
                self.vstars.extend([vpython.sphere() for i in range(len(self.xyzs.value)-len(self.vstars))])
            elif len(self.vstars) > len(self.xyzs.value):
                while len(self.vstars) > len(self.xyzs.value):
                    self.vstars[-1].visible = False
                    del self.vstars[-1]
            vpython.rate(100)
            self.vsphere.pos = vpython.vector(*self.center)
            self.vsphere.radius = np.linalg.norm(np.array(self.spin_axis()))
            self.vsphere.color = vpython.color.blue
            self.vsphere.opacity = 0.4
            self.vspin_axis.pos = vpython.vector(*self.center)
            self.vspin_axis.axis = vpython.vector(*self.spin_axis())
            for i in range(len(self.vstars)):
                self.vstars[i].pos = self.vsphere.pos + self.vsphere.radius*vpython.vector(*self.xyzs.value[i])
                self.vstars[i].radius = 0.1*self.vsphere.radius
                self.vstars[i].color = vpython.color.white
                self.vstars[i].opacity = 0.8
    
    def spin_operators(self):
        if self.state.value != None:
            n = len(self.vector.value)
            spin = (n-1.)/2.
            return {"X": qutip.jmat(spin, "x"),\
                    "Y": qutip.jmat(spin, "y"),\
                    "Z": qutip.jmat(spin, "z"),\
                    "+": qutip.jmat(spin, "+"),\
                    "-": qutip.jmat(spin, "-")}
    
    def qubit_operators(self):
        return {"X": qutip.jmat(0.5, "x"),\
                "Y": qutip.jmat(0.5, "y"),\
                "Z": qutip.jmat(0.5, "z"),\
                "+": qutip.jmat(0.5, "+"),\
                "-": qutip.jmat(0.5, "-")}
    
    def spin_axis(self):
        spin_ops = self.spin_operators()
        return [-1*qutip.expect(spin_ops["X"], self.state.value),\
                qutip.expect(spin_ops["Y"], self.state.value),\
                qutip.expect(spin_ops["Z"], self.state.value)]
    
    def evolve(self, hermitian, inverse=False, dt=0.007):
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*hermitian.full()*dt))
        if inverse:
            unitary = unitary.dag()
        self.state.plug(unitary*self.state.value)
        
    def evolve_qubit(self, i, hermitian, inverse=False, dt=0.007):
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*hermitian.full()*dt))
        if inverse:
            unitary = unitary.dag()
        qubits = self.qubits.value[:]
        qubits[i] = unitary*qubits[i]
        self.qubits.plug(qubits)

###############################################################################################################


n = 2
d = 3

if len(sys.argv) == 3:
  if sys.argv[1].isdigit() and sys.argv[2].isdigit():
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    
spheres = [Sphere(0,i,0) for i in range(0,n)]

print("usage: ladder_qubits.py *n-spheres* *of-d-dimensionality")
print("click sphere to select")
print("a/d s/w z/x -> sigma X, Z, Y on whole")
print("q/1 are sigma - and sigma - dag")
print("e/3 are sigma + and sigma + dag")
print()
print("click surface star to select")
print("j/l k/i m/, -> sigma X, Z, Y on selected star")
print("u/7 are sigma - and sigma - dag")
print("o/0 are sigma + and sigma + dag")

###############################################################################################################

vpython.scene.width = 600
vpython.scene.height = 800
vpython.scene.userspin = True

selected = None
sphere_selected = 0

def mouse(event):
    global selected
    global sphere_selected
    selected = vpython.scene.mouse.pick
    for sphere in spheres:
        if sphere.vsphere == selected:
            sphere_selected = spheres.index(sphere)
vpython.scene.bind('click', mouse)

def keyboard(event):
    global sphere
    global selected
    global sphere_selected
    key = event.key
    spin_ops = spheres[sphere_selected].spin_operators()
    qubit_ops = spheres[sphere_selected].qubit_operators()
    if key == "a":  
        spheres[sphere_selected].evolve(spin_ops["X"], inverse=True)
    elif key == "d": 
        spheres[sphere_selected].evolve(spin_ops["X"], inverse=False)
    elif key == "s": 
        spheres[sphere_selected].evolve(spin_ops["Z"], inverse=True)
    elif key == "w": 
        spheres[sphere_selected].evolve(spin_ops["Z"], inverse=False)
    elif key == "z": 
        spheres[sphere_selected].evolve(spin_ops["Y"], inverse=True)
    elif key == "x": 
        spheres[sphere_selected].evolve(spin_ops["Y"], inverse=False)
    elif key == "q": 
        spheres[sphere_selected].evolve(spin_ops["-"], inverse=False)
    elif key == "1":
        spheres[sphere_selected].evolve(spin_ops["-"], inverse=True)
    elif key == "e":
        spheres[sphere_selected].evolve(spin_ops["+"], inverse=False)
    elif key == "3":
        spheres[sphere_selected].evolve(spin_ops["+"], inverse=True)   
    elif key == "j":
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["X"], inverse=True)
    elif key == "l": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["X"], inverse=False)
    elif key == "k": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["Z"], inverse=True)
    elif key == "i": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["Z"], inverse=False)
    elif key == "m": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["Y"], inverse=True)
    elif key == ",": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["Y"], inverse=False)
    elif key == "u": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["-"], inverse=False)
    elif key == "7": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["-"], inverse=True)
    elif key == "o": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["+"], inverse=False)
    elif key == "0": 
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].evolve_qubit(i, qubit_ops["+"], inverse=True)
vpython.scene.bind('keydown', keyboard)

###############################################################################################################

for i in range(n):
    spheres[i].state.plug(qutip.rand_ket(d))

for sphere in spheres:
    sphere.visualize()
        
vpython.scene.camera.follow(spheres[int(n/2)].vsphere)

while True:
    for sphere in spheres:
        sphere.visualize()