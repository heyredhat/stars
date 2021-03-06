import qutip
import qutip as qt
import numpy as np
import vpython as vp
import math
import cmath
import sympy
import scipy
import mpmath
import operator
import threading
import itertools
import functools

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

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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

dt = 0.0001
modes = 2
freq = 1

a = qt.destroy(modes)

aX = qt.tensor(qt.destroy(modes), qt.identity(modes), qt.identity(modes))
nX = qt.tensor(qt.num(modes), qt.identity(modes), qt.identity(modes))
xX = qt.tensor((qt.destroy(modes) + qt.destroy(modes).dag())/np.sqrt(2), qt.identity(modes), qt.identity(modes))
pX = qt.tensor(-1j * (qt.destroy(modes) - qt.destroy(modes).dag())/np.sqrt(2), qt.identity(modes), qt.identity(modes))

aY = qt.tensor(qt.identity(modes), qt.destroy(modes), qt.identity(modes))
nY = qt.tensor(qt.identity(modes), qt.num(modes), qt.identity(modes))
xY = qt.tensor(qt.identity(modes), (qt.destroy(modes) + qt.destroy(modes).dag())/np.sqrt(2), qt.identity(modes))
pY = qt.tensor(qt.identity(modes), -1j * (qt.destroy(modes) - qt.destroy(modes).dag())/np.sqrt(2), qt.identity(modes))

aZ = qt.tensor(qt.identity(modes), qt.identity(modes), qt.destroy(modes))
nZ = qt.tensor(qt.identity(modes), qt.identity(modes), qt.num(modes))
xZ = qt.tensor(qt.identity(modes), qt.identity(modes), (qt.destroy(modes) + qt.destroy(modes).dag())/np.sqrt(2))
pZ = qt.tensor(qt.identity(modes), qt.identity(modes), -1j * (qt.destroy(modes) - qt.destroy(modes).dag())/np.sqrt(2))

aplus = (-1/np.sqrt(2))*(aX - 1j * aY)
aminus = (1/np.sqrt(2))*(aX + 1j * aY)

Lz = (aplus.dag() * aplus - aminus.dag()*aminus) #(1./(2*math.pi))*
Lplus = (np.sqrt(2)/1)*(aplus.dag()*aZ + aZ.dag()*aminus) #(2*math.pi)
Lminus = (np.sqrt(2)/1)*(aminus.dag()*aZ + aZ.dag()*aplus) #(2*math.pi)
Lx = (Lplus+Lminus)/2
Ly = (Lplus-Lminus)/(2j)

spin_d = 2
spin = qt.rand_ket(spin_d)

Lx = qt.tensor(Lx, qt.qeye(spin_d))
Ly = qt.tensor(Ly, qt.qeye(spin_d))
Lz = qt.tensor(Lz, qt.qeye(spin_d))

Sx = qt.tensor(qt.identity(modes), qt.identity(modes), qt.identity(modes), qt.sigmax())
Sy = qt.tensor(qt.identity(modes), qt.identity(modes), qt.identity(modes), qt.sigmay())
Sz = qt.tensor(qt.identity(modes), qt.identity(modes), qt.identity(modes), qt.sigmaz())

Jx = Lx + Sx
Jy = Ly + Sy
Jz = Lz + Sz

spin_coupling = 50

x_term = (0.5)*(Jx*Jx - Lx*Lx - Sx*Sx)
y_term = (0.5)*(Jy*Jy - Ly*Ly - Sy*Sy)
z_term = (0.5)*(Jz*Jz - Lz*Lz - Sz*Sz)

H = 2 * np.pi * freq * qt.tensor((aX.dag() * aX + 0.5)+\
                                 (aY.dag() * aY + 0.5)+\
                                 (aZ.dag() * aZ + 0.5), qt.qeye(spin_d))\
                             + spin_coupling*(x_term + y_term + z_term)

#H = spin_coupling*(x_term + y_term + z_term)

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
        self.dt = 0.0001
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

        self.active = 0

    def keyboard(self, event):
        global H, Sx, Lx, Jx, Sy, Ly, Jy, Sz, Lz, Jz, spin_coupling
        key = event.key
        spin_ops = self.spin_operators()
        if key == "a":
            if self.active == 0: # Pauli on whole
                self.evolve(spin_ops['X'], inverse=True, dt=0.01)
            elif self.active == 1: # Sx
                self.evolve(Sx, inverse=True, dt=0.01)
            elif self.active == 2: # Lx
                self.evolve(Lx, inverse=True, dt=0.01)
            elif self.active == 3: # Jx
                self.evolve(Jx, inverse=True, dt=0.01)
        elif key == "d":
            if self.active == 0: # Pauli on whole
                self.evolve(spin_ops['X'], inverse=False, dt=0.01)
            elif self.active == 1: # Sx
                self.evolve(Sx, inverse=False, dt=0.01)
            elif self.active == 2: # Lx
                self.evolve(Lx, inverse=False, dt=0.01)
            elif self.active == 3: # Jx
                self.evolve(Jx, inverse=False, dt=0.01)
        elif key == "s":
            if self.active == 0: # Pauli on whole
                self.evolve(spin_ops['Y'], inverse=True, dt=0.01)
            elif self.active == 1: # Sx
                self.evolve(Sy, inverse=True, dt=0.01)
            elif self.active == 2: # Lx
                self.evolve(Ly, inverse=True, dt=0.01)
            elif self.active == 3: # Jx
                self.evolve(Jy, inverse=True, dt=0.01)
        elif key == "w":
            if self.active == 0: # Pauli on whole
                self.evolve(spin_ops['Y'], inverse=False, dt=0.01)
            elif self.active == 1: # Sx
                self.evolve(Sy, inverse=False, dt=0.01)
            elif self.active == 2: # Lx
                self.evolve(Ly, inverse=False, dt=0.01)
            elif self.active == 3: # Jx
                self.evolve(Jy, inverse=False, dt=0.01)
        elif key == "z":
            if self.active == 0: # Pauli on whole
                self.evolve(spin_ops['Z'], inverse=True, dt=0.01)
            elif self.active == 1: # Sx
                self.evolve(Sz, inverse=True, dt=0.01)
            elif self.active == 2: # Lx
                self.evolve(Lz, inverse=True, dt=0.01)
            elif self.active == 3: # Jx
                self.evolve(Jz, inverse=True, dt=0.01)
        elif key == "x":    
            if self.active == 0: # Pauli on whole
                self.evolve(spin_ops['Z'], inverse=False, dt=0.01)
            elif self.active == 1: # Sx
                self.evolve(Sz, inverse=False, dt=0.01)
            elif self.active == 2: # Lx
                self.evolve(Lz, inverse=False, dt=0.01)
            elif self.active == 3: # Jx
                self.evolve(Jz, inverse=False, dt=0.01)
        elif key == "e":
            if self.evolving:
                self.evolving = False
            else:
                self.evolving = True
        elif key == "r":
            self.state = qt.rand_ket(self.n)
        elif key == "t":
            self.energy = qt.rand_herm(self.n)
        elif key == "y":
            self.energy = H
        elif key == "[":
            self.dt += 0.001
        elif key == "]":
            self.dt -= 0.001
        elif key == "0":
            self.active = 0
        elif key == "1":
            self.active = 1
        elif key == "2":
            self.active = 2
        elif key == "3":
            self.active = 3

    def display_visuals(self):
        if self.evolving:
            self.evolve(self.energy, dt=self.dt)
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

    def evolve(self, operator, inverse=False, dt=0.0001):
        old_dims = self.state.dims[:]
        self.state.dims = [[self.n],[1]]
        unitary = (-2*math.pi*im()*dt*operator).expm()
        if inverse:
            unitary = unitary.dag()
        unitary.dims = [[self.n],[self.n]]
        self.state = unitary*self.state
        self.state.dims = old_dims

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

##################################################################################################################

state = qt.tensor(qt.rand_ket(modes), qt.rand_ket(modes), qt.rand_ket(modes), qt.rand_ket(spin_d))

vp.scene.height = 600
vp.scene.wdith = 1000

sphere = MajoranaSphere(state)
sphere.energy = H
sphere.evolving = True

x_axis = vp.arrow(pos=vp.vector(0,0,0), axis=vp.vector(1,0,0), opacity=0.3)
y_axis = vp.arrow(pos=vp.vector(0,0,0), axis=vp.vector(0,1,0), opacity=0.3)
y_axis = vp.arrow(pos=vp.vector(0,0,0), axis=vp.vector(0,0,1), opacity=0.3)

momentum = vp.arrow(color=vp.color.red, opacity=0.5)

spinny = vp.sphere(color=vp.color.magenta, radius=0.03, emissive=True)
spin_momentum = vp.arrow(color=vp.color.magenta)
orbital_momentum = vp.arrow(color=vp.color.cyan)
ang_momentum  = vp.arrow(color=vp.color.yellow)

vp.scene.bind('keydown', sphere.keyboard)  

arrow = vp.arrow(pos=vp.vector(0,0,0), color=vp.color.green, opacity=0.5)

while True:
    copy = sphere.state.copy()
    copy.dims = [[modes, modes, modes, spin_d], [1, 1, 1, 1]]
    sphere.sphere_color = vp.color.hsv_to_rgb(vp.vector(1./qt.expect(H, copy),1,1))
    sphere.center = vp.vector(qt.expect(qt.tensor(xX, qt.qeye(spin_d)), copy),\
                              qt.expect(qt.tensor(xY, qt.qeye(spin_d)), copy),\
                              qt.expect(qt.tensor(xZ, qt.qeye(spin_d)), copy))

    momentum.pos = sphere.center
    momentum.axis = vp.vector(qt.expect(qt.tensor(pX, qt.qeye(spin_d)), copy),\
                              qt.expect(qt.tensor(pY, qt.qeye(spin_d)), copy),\
                              qt.expect(qt.tensor(pZ, qt.qeye(spin_d)), copy))

    spin_momentum.pos = sphere.center
    spin_momentum.axis = vp.vector(qt.expect(Sx, copy),\
                                   qt.expect(Sy, copy),\
                                   qt.expect(Sz, copy))
    spinny.pos = spin_momentum.axis + sphere.center

    orbital_momentum.pos = sphere.center
    orbital_momentum.axis = vp.vector(qt.expect(Lx, copy),\
                                      qt.expect(Ly, copy),\
                                      qt.expect(Lz, copy))

    ang_momentum.pos = sphere.center
    ang_momentum.axis = vp.vector(qt.expect(Jx, copy),\
                                  qt.expect(Jy, copy),\
                                  qt.expect(Jz, copy))
    arrow.axis = sphere.center
    sphere.display_visuals()

