# TODO
#  1. Field
#  2. Embeddings
#  3. Dimensionality

###############################################################################################################

import sys
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
import random

###############################################################################################################

def combos(a,b):
    f = math.factorial
    return f(a) / f(b) / f(a-b)

def symmeterize(pieces, labels):
    n = len(pieces)
    unique_labels = list(set(labels))
    label_counts = [0 for i in range(len(unique_labels))]
    label_permutations = itertools.permutations(labels, n)
    for permutation in label_permutations:
        for i in range(len(unique_labels)):
            label_counts[i] += list(permutation).count(unique_labels[i])
    normalization = math.sqrt(functools.reduce(operator.mul, [math.factorial(count) for count in label_counts], 1)/math.factorial(n))    
    normalization = 1./math.sqrt(math.factorial(n))
    permutations = itertools.permutations(pieces, n)
    tensor_sum = sum([qutip.tensor(list(permutation)) for permutation in permutations])
    return normalization*tensor_sum

def collapser(dims):
    if all(isinstance(d, int) for d in dims):
        return functools.reduce(operator.mul, dims, 1)
    else:
        new_dims = []
        for d in dims:
            if isinstance(d, int):
                new_dims.append(d)
            else:
                new_dims.append(collapser(d))
        return collapser(new_dims)

def collapse(dims):
    new_dims = []
    for d in dims:
        if isinstance(d, int):
            new_dims.append(d)
        else:
            new_dims.append(collapser(d))
    return new_dims

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

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
        self.dt = 0.007
        
        self.center = [x, y, z]
        self.color = None
        self.radius = None
        
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
        
        self.dimensionality = None
        self.children = []
        
        self.embeddings = []
        self.embedded_spheres = []
          
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
                return [np.conjugate(complex(root)) for root in np.roots(polynomial)]
            except:
                raise Exception(str(polynomial))
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
                x, y, z = xyz
                r = math.sqrt(x**2 + y**2 + z**2)
                phi = math.atan2(y,x)
                theta = None
                if r != 0:
                    theta = math.acos(z/r)
                else:
                    theta = math.pi/(2*np.sign(z))
                while phi < 0:
                    phi += 2*math.pi
                return (theta, phi)
            return [xyz_to_angle(xyz) for xyz in xyzs]
        self.xyzs.tie(self.angles, xyzs_to_angles)
        
        def angles_to_xyzs(angles):
            def angle_to_xyz(angle):
                theta, phi = angle
                return [math.sin(theta)*math.cos(phi),\
                        math.sin(theta)*math.sin(phi),\
                        math.cos(theta)]
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
            def xyz_to_angle(xyz):
                x, y, z = xyz
                r = math.sqrt(x**2 + y**2 + z**2)
                phi = math.atan2(y,x)
                theta = None
                if r != 0:
                    theta = math.acos(z/r)
                else:
                    theta = math.pi/(2*np.sign(z))
                while phi < 0:
                    phi += 2*math.pi
                return (theta, phi)
            def qubit_to_angle(qubit):
                vec = qubit.full().T[0]
                first = vec[0]
                vec = vec/first
                qubit = qutip.Qobj(vec)
                dm = qubit.ptrace(0).full()
                x = float(2*dm[0][1].real)
                y = float(2*dm[1][0].imag)
                z = float((dm[0][0] - dm[1][1]).real)
                return xyz_to_angle([x, y, z])
            return [qubit_to_angle(qubit) for qubit in qubits]
        self.qubits.tie(self.angles, qubits_to_angles)
        
        def qubits_to_field(qubits):
            field = {"particles": [i for i in range(len(qubits))],\
                     "normalization": 1}
            return field
        self.qubits.tie(self.field, qubits_to_field)
        
        def field_to_qubits(field):
            particles = field["particles"]
            normalization = field["normalization"]
            return [self.qubits.value[particle] for particle in particles]
        self.field.tie(self.qubits, field_to_qubits)
        
    def visualize(self):
        if self.state.value != None:
            if self.vsphere == None:
                self.vsphere = vpython.sphere()
            if self.vspin_axis == None:
                self.vspin_axis = vpython.arrow()
            if self.vstars == None:
                self.vstars = [vpython.sphere() for i in range(len(self.xyzs.value))]
            vpython.rate(100)
            self.vsphere.pos = vpython.vector(*self.center)
            if self.radius == None:
                self.vsphere.radius = np.linalg.norm(np.array(self.spin_axis()))    
                #self.vsphere.radius = self.state.value.norm()
            else:
                #r, th = cmath.polar(self.radius)
                #self.vsphere.radius = r**2
                self.vsphere.radius = self.radius
            self.vsphere.color = self.color
            self.vsphere.opacity = 0.4
            self.vspin_axis.pos = vpython.vector(*self.center)
            self.vspin_axis.axis = vpython.vector(*(self.vsphere.radius*normalize(self.spin_axis())))
            #self.vspin_axis.axis = vpython.vector(*self.spin_axis())
            self.vspin_axis.color = self.color
            self.vspin_axis.opacity = 0.6
            for i in range(len(self.vstars)):
                self.vstars[i].pos = self.vsphere.pos + self.vsphere.radius*vpython.vector(*self.xyzs.value[i])
                self.vstars[i].radius = 0.1*self.vsphere.radius
                self.vstars[i].color = self.color
                self.vstars[i].emissive = True
                self.vstars[i].opacity = 0.8
            if self.dimensionality != None:
                self.bear_children()
            for i in range(len(self.embeddings)):
                my_projector = self.state.value.ptrace(0)
                other_state = self.embeddings[i].state.value
                projected_state = my_projector*other_state
                self.embedded_spheres[i].center = self.center
                #self.embedded_spheres[i].radius = other_state.overlap(self.state.value)
                self.embedded_spheres[i].radius = 2*np.inner(np.array(self.embeddings[i].spin_axis()), np.array(self.spin_axis()))
                self.embedded_spheres[i].state.plug(projected_state)
                self.embedded_spheres[i].visualize()
    
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
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*hermitian.full()*self.dt))
        if inverse:
            unitary = unitary.dag()
        self.state.plug(unitary*self.state.value)
        
    def evolve_qubit(self, i, hermitian, inverse=False, dt=0.007):
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*hermitian.full()*self.dt))
        if inverse:
            unitary = unitary.dag()
        qubits = self.qubits.value[:]
        qubits[i] = unitary*qubits[i]
        self.qubits.plug(qubits)
        
    def creation_operators(self):
        a = qutip.Qobj(np.array([[0,0],[1,0]]))
        b = qutip.Qobj(np.array([[1,0],[0,0]]))
        creation_operators = []
        for qubit in self.qubits.value:
            alpha, beta = qubit.full().T[0].tolist()
            creation_operator = alpha*a + beta*b
            creation_operators.append(creation_operator)
        return creation_operators

    def annihilation_operators(self):
        return [op.dag() for op in self.creation_operators()]
    
    def create(self, i):
        field = self.field.value
        normalization = field["normalization"]
        particles = field["particles"]
        normalization *= math.sqrt(len(particles)+1)
        particles.append(i)
        field["normalization"] = normalization
        field["particles"] = particles
        self.vstars.append(vpython.sphere())
        self.field.plug(field)
    
    def destroy(self, i):
        if len(self.qubits.value) > 1:
            field = self.field.value
            normalization = field["normalization"]
            particles = field["particles"]
            normalization *= math.sqrt(len(particles))*\
                             self.qubits.value[i].overlap(self.qubits.value[particles[-1]])
            del particles[-1]
            field["normalization"] = normalization
            field["particles"] = particles
            self.field.plug(field)
            self.vstars[-1].visible = False
            del self.vstars[-1]
    
    def field_state(self):
        field = self.field.value
        normalization = field["normalization"]
        particles = field["particles"]
        return normalization*symmeterize([self.qubits.value[i] for i in particles], particles)
    
    def embed(self, other):
        self.embeddings.append(other)
        embedded_sphere = Sphere(*self.center)
        embedded_sphere.color = other.color
        self.embedded_spheres.append(embedded_sphere)
        
    def unembed(self, other):
        i = self.embeddings.index(other)
        self.embeddings.remove(other)
        self.embedded_spheres[i].kill()
        
    def kill(self):
        if self.vsphere:
            self.vsphere.visible = False
            del self.vsphere
            self.vsphere = None
        if self.vspin_axis:
            self.vspin_axis.visible = False
            del self.vspin_axis
            self.vspin_axis = None
        if self.vstars:
            for vstar in self.vstars:
                vstar.visible = False
                del vstar
            self.vstars = None
        if self.embedded_spheres:
            for sphere in self.embedded_spheres:
                sphere.kill()
    
    def bear_children(self):
        if self.dimensionality != None and self.state.value != None:
            dim = collapse(self.dimensionality)
            state = self.state.value.copy()
            state.dims = [dim, [1]*len(dim)]
            distinguishable_subspaces = len(dim)
            if distinguishable_subspaces > 1:
                child_mixed_states = [state.ptrace(i) for i in range(distinguishable_subspaces)]
                child_states = []
                for child_state in child_mixed_states:
                    eigenvals, eigenvecs = child_state.eigenstates()
                    child_states.append(sum([eigenvals[i]*eigenvecs[i] for i in range(len(eigenvecs))]))
                if self.children == []:
                    for i in range(distinguishable_subspaces):
                        child_dim = None
                        if isinstance(self.dimensionality[i], int):
                            child_dim = [self.dimensionality[i]]
                        else:
                            child_dim = self.dimensionality[i]
                        child_sphere = Sphere(*self.center)
                        tcolor = vpython.color.rgb_to_hsv(self.color)
                        tcolor.x *= i/distinguishable_subspaces
                        child_sphere.color = vpython.color.hsv_to_rgb(tcolor)
                        child_sphere.dimensionality = child_dim
                        child_sphere.state.plug(child_states[i])
                        child_sphere.visualize()
                        self.children.append(child_sphere)
                else:
                    for i in range(distinguishable_subspaces):
                        self.children[i].state.plug(child_states[i])
                        self.children[i].visualize()
    
    def evolve_child(self, i, hermitian, inverse=False, dt=0.007):
        if self.dimensionality != None and self.state.value != None:
            unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*hermitian.full()*self.dt))
            if inverse:
                unitary = unitary.dag()
            dim = collapse(self.dimensionality)
            upgraded = None
            if i == 0:
                upgraded = unitary
            else:
                upgraded = qutip.identity(dim[0])
            for j in range(1, len(dim)):
                if j == i:
                    upgraded = qutip.tensor(upgraded, unitary)
                else:
                    upgraded = qutip.tensor(upgraded, qutip.identity(dim[j]))
            upgraded.dims = [[collapser(dim)], [collapser(dim)]]
            self.state.plug(upgraded*self.state.value)
            
###############################################################################################################

colors = [vpython.color.red, vpython.color.green, vpython.color.blue,\
          vpython.color.yellow, vpython.color.orange, vpython.color.cyan,\
          vpython.color.magenta]

n = 2
d = 2
mutual_embedding = True

print("usage: python stacked_and_projected.py *n-spheres* *of-d-dimensionality* *mutual_embedding:0/1*")
print("click to select an outer sphere, a star, or a child sphere (the latter by spin axis")
print("a/d s/w z/x q/1 e/3 -> +/- sigma X Z Y -/-dag +/+dag on selected sphere")
print("j/l k/i m/, u/7 o/0 -> +/- sigma X Z Y -/-dag +/+dag on selected star")
print("+/- create and destroy a new selected star")
print("` print field state woah-- this and the above probably don't work")
print("f/h g/t v/b -> sigma X Z Y on selected child sphere")

if len(sys.argv) == 4:
  if sys.argv[1].isdigit() and sys.argv[2].isdigit() and sys.argv[3].isdigit():
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    me = int(sys.argv[3])
    if me == 0:
        mutual_embedding = False
    elif me == 1:
        mutual_embedding = True

spheres = [Sphere(0,0,0) for i in range(n)]
#spheres[0].dimensionality = [2]
#spheres[1].dimensionality = [2]

sphere_colors = random.sample(colors, n)
for i in range(n):
    spheres[i].color = sphere_colors[i]

if mutual_embedding:
    for i in range(n):
        for j in range(n):
            if i != j:
                spheres[i].embed(spheres[j])

###############################################################################################################

vpython.scene.width = 600
vpython.scene.height = 800
vpython.scene.userspin = True

selected = None
sphere_selected = 0
child_sphere_selected = 0

def mouse(event):
    global selected
    global sphere_selected
    global child_sphere_selected
    selected = vpython.scene.mouse.pick
    for sphere in spheres:
        if sphere.vsphere == selected:
            sphere_selected = spheres.index(sphere)
        else:
            for child_sphere in sphere.children:
                if child_sphere.vspin_axis == selected:
                    child_sphere_selected = sphere.children.index(child_sphere)
vpython.scene.bind('click', mouse)

def keyboard(event):
    global sphere
    global selected
    global sphere_selected
    global child_sphere_selected
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
    elif key == "+":
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].create(i)
    elif key == "-":
        if selected and selected in spheres[sphere_selected].vstars:
            i = spheres[sphere_selected].vstars.index(selected)
            spheres[sphere_selected].destroy(i)    
    elif key == "`":
        field_state = spheres[sphere_selected].field_state()
        print(field_state)
    elif key == "f":
        if spheres[sphere_selected].children != []:
            child_spin_ops = spheres[sphere_selected].children[child_sphere_selected].spin_operators()
            spheres[sphere_selected].evolve_child(child_sphere_selected, child_spin_ops["X"], inverse=True)
    elif key == "h":
        if spheres[sphere_selected].children != []:
            child_spin_ops = spheres[sphere_selected].children[child_sphere_selected].spin_operators()
            spheres[sphere_selected].evolve_child(child_sphere_selected, child_spin_ops["X"], inverse=False)
    elif key == "g":
        if spheres[sphere_selected].children != []:
            child_spin_ops = spheres[sphere_selected].children[child_sphere_selected].spin_operators()
            spheres[sphere_selected].evolve_child(child_sphere_selected, child_spin_ops["Z"], inverse=True)
    elif key == "t":
        if spheres[sphere_selected].children != []:
            child_spin_ops = spheres[sphere_selected].children[child_sphere_selected].spin_operators()
            spheres[sphere_selected].evolve_child(child_sphere_selected, child_spin_ops["Z"], inverse=False)
    elif key == "v":
        if spheres[sphere_selected].children != []:
            child_spin_ops = spheres[sphere_selected].children[child_sphere_selected].spin_operators()
            spheres[sphere_selected].evolve_child(child_sphere_selected, child_spin_ops["Y"], inverse=True)
    elif key == "b":
        if spheres[sphere_selected].children != []:
            child_spin_ops = spheres[sphere_selected].children[child_sphere_selected].spin_operators()
            spheres[sphere_selected].evolve_child(child_sphere_selected, child_spin_ops["Y"], inverse=False)
vpython.scene.bind('keydown', keyboard)

###############################################################################################################

for i in range(n):
    spheres[i].state.plug(qutip.rand_ket(d))

for sphere in spheres:
    sphere.visualize()
        
vpython.scene.camera.follow(spheres[int(n/2)].vsphere)

while True:
    account = 0
    for i in range(n):
        if i == 0:
            spheres[i].center = [0,0,0]
        elif i > 0:
            account += spheres[i].vsphere.radius+spheres[i-1].vsphere.radius
            spheres[i].center = [0, account, 0]
        spheres[i].visualize()