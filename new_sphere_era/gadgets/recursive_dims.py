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

##################################################################################################################

def c_xyz(c):
    if c == float('inf'):
        return [0,0,1]
    x = c.real
    y = c.imag
    return [(2*x)/(1.+(x**2)+(y**2)),\
            (2*y)/(1.+(x**2)+(y**2)),\
            (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]

def xyz_c(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
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

def sigmoid(x):  
    return 2*(math.exp(-np.logaddexp(0, -x))-0.5)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

##################################################################################################################


class Sphere:
    def __init__(self, state=None, dimensionality=None, energy=None, parent=None):
        self.state = state
        self.dimensionality = dimensionality
        self.energy = energy
        self.parent = parent

        self.spin = self.total_spin()
        self.stars = self.constellate()
        self.children = []
        self.bear_children
        
        self.vsphere = vpython.sphere(pos=vpython.vector(*self.center()),\
                                      radius=self.radius(),\
                                      color=vpython.color.hsv_to_rgb(vpython.vector(sigmoid(self.color()),1,1)),\
                                      opacity=0.1,
                                      emissive=False)
        self.vstars = [vpython.sphere(pos=vpython.vector(*((self.radius()*star["pos"])+np.array(self.center()))),\
                                      radius=star["radius"],\
                                      color=star["color"],\
                                      opacity=star["opacity"],\
                                      emissive=True) for star in self.stars]
        #self.varrows = []
        #for vstar in self.vstars:
        #    self.varrows.append(vpython.curve(pos=[vstar.pos, vpython.vector(*self.center())],\
        #                                      color=vstar.color))
            
    def constellate(self):
        old_dims = list(self.state.dims)
        n = collapser(self.dimensionality)
        self.state.dims = [[n], [n]]
        eigenvalues, eigenvectors = self.state.eigenstates()
        normalized_eigenvalues = normalize(eigenvalues) 
        eigenstars = []
        for i in range(n):
            eigenstars.extend([{"pos": (normalized_eigenvalues[i])*np.array(xyz), 
                                "radius": 0.1*self.radius(),\
                                "color": vpython.color.hsv_to_rgb(vpython.vector(sigmoid(eigenvalues[i]),1,1)), 
                                "opacity": sigmoid(normalized_eigenvalues[i])} 
                                       for xyz in q_SurfaceXYZ(eigenvectors[i])])
        self.state.dims = old_dims
        return eigenstars
    
    def bear_children(self):
        dim = collapse(self.dimensionality)
        self.state.dims = [dim, dim]
        distinguishable_subspaces = len(dim)
        if distinguishable_subspaces > 1:
            child_states = [self.state.ptrace(i) for i in range(distinguishable_subspaces)]
            if self.children == []:
                for i in range(distinguishable_subspaces):
                    child_dim = None
                    if isinstance(self.dimensionality[i], int):
                        child_dim = [self.dimensionality[i]]
                    else:
                        child_dim = self.dimensionality[i]
                    self.children.append(Sphere(state=child_states[i], dimensionality=child_dim, parent=self))
            else:
                for i in range(distinguishable_subspaces):
                    self.children[i].state = child_states[i]
    
    def evolve(self, dt=0.01):
        if self.energy != None:
            unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*self.energy.full()*dt))
            unitary.dims = self.state.dims
            self.state = unitary*self.state*unitary.dag()

    def total_spin(self):
        old_dims = list(self.state.dims)
        n = collapser(self.dimensionality)
        self.state.dims = [[n], [n]]
        T = qutip.identity(n)
        X, Y, Z = qutip.jmat((n-1.)/2.)
        t = qutip.expect(T, self.state)
        x = qutip.expect(X, self.state)
        y = qutip.expect(Y, self.state)
        z = qutip.expect(Z, self.state)
        spin = np.array([t, x, y, z])
        magnitude = np.linalg.norm(spin)
        if magnitude != 0:
            spin = spin / magnitude
        self.state.dims = old_dims
        return spin.tolist()

    def center(self):
        if self.parent == None:
            return [0,0,0]
        else:
            parent_center = self.parent.center()
            return [parent_center[i]+self.total_spin()[i+1] for i in range(3)]

    def color(self):
        return self.total_spin()[0]

    def radius(self):
        if self.parent == None:
            return 1
        else:
            return (1./2)*self.parent.radius()
        
    def revolve(self):
        self.spin = self.total_spin()
        self.vsphere.pos = vpython.vector(*self.center())
        self.vsphere.radius = self.radius()
        self.vsphere.color = vpython.color.hsv_to_rgb(vpython.vector(sigmoid(self.color()),1,1))
            
    def cycle(self):
        self.evolve()
        self.stars = self.constellate()
        for i in range(len(self.stars)):
            self.vstars[i].pos = vpython.vector(*((self.radius()*self.stars[i]["pos"])+np.array(self.center())))
            self.vstars[i].radius = self.stars[i]["radius"]
            self.vstars[i].color = self.stars[i]["color"]
            self.vstars[i].opacity = self.stars[i]["opacity"]
            #self.varrows[i].modify(0, pos=self.vstars[i].pos)
            #self.varrows[i].modify(1, pos=vpython.vector(*self.center()))
            #self.varrows[i].color = self.vstars[i].color
        self.bear_children()
        self.revolve()
        for child in self.children:
            child.cycle()

##################################################################################################################

def random_sphere(dimensionality):
    n = collapser(dimensionality)
    pure_state = qutip.rand_ket(n)
    state = pure_state.ptrace(0)
    #state = qutip.rand_herm(n)
    energy = qutip.rand_herm(n)
    return Sphere(state=state, dimensionality=dimensionality, energy=energy)

vpython.scene.width = 900
vpython.scene.height = 900
vpython.scene.range = math.pi
vpython.scene.forward = vpython.vector(-1, 0, 0)
vpython.scene.up = vpython.vector(0, 1, 0)

sphere = random_sphere([2,2])

while True:
    vpython.rate(10)
    sphere.cycle()