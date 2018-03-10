import math
import cmath
import functools
import numpy as np
import sympy
import mpmath
import scipy
import qutip 
import random
import datetime
import primefac
import vpython

#

def I():
    return complex(0,1)

def dag(matrix):
    return np.conjugate(matrix.T)

def sigmoid(x):  
    return 2*(math.exp(-np.logaddexp(0, -x))-0.5)

#

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

#

def xyz_txyz(xyz):
    return np.array([1]+xyz)

def txyz_xyz(txyz):
    return txyz_unitSphere(txyz)[1:].tolist()

def txyz_unitSphere(txyz):
    t, x, y, z = txyz[0], txyz[1], txyz[2], txyz[3]
    return np.array([t/math.sqrt(x**2+y**2+z**2), x/t, y/t, z/t])

#

def c_txyz(c):
    if c == float('inf'):
        return np.array([1,0,0,1])
    c = np.conjugate(c)
    u = c.real
    v = c.imag
    return np.array([u**2 + v**2 + 1, 2*u, -2*v, u**2 + v**2 - 1])

def txyz_c(txyz):
    return xyz_c(txyz_xyz(txyz))

#

def xyz_hermitian(xyz):
    return txyz_hermitian(xyz_txyz(xyz))    

def txyz_hermitian(txyz):
    t, x, y, z = txyz[0], txyz[1], txyz[2], txyz[3]
    return np.array([[t+z, x-I()*y],[x+I()*y, t-z]])

def hermitian_xyz(hermitian):
    return txyz_xyz(hermitian_txyz(hermitian))

def hermitian_txyz(hermitian):
    def scalarProduct(m, n):
        return 0.5*np.trace(np.dot(np.conjugate(m).T, n))
    t = scalarProduct(hermitian, np.eye(2)).real
    x = scalarProduct(hermitian, qutip.sigmax().full()).real
    y = scalarProduct(hermitian, qutip.sigmay().full()).real
    z = scalarProduct(hermitian, qutip.sigmaz().full()).real
    return np.array([t, x, y, z])

def txyz_spacetimeInterval(txyz):
    t, x, y, z = txyz[0], txyz[1], txyz[2], txyz[3]
    return t**2 - x**2 - y**2 - z**2

def hermitian_spacetimeInterval(hermitian):
    return np.linalg.det(hermitian)

#

def c_hermitian(c):
    if c == float('inf'):
        return txyz_hermitian(np.array([1,0,0,1]))
    u = c.real
    v = c.imag
    return np.conjugate(np.array([[u**2 + v**2, u+I()*v],[u-I()*v, 1]]))

def hermitian_c(hermitian):
    return txyz_c(hermitian_txyz(hermitian))

#

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

#

def polynomial_v(polynomial):
    coordinates = [polynomial[i]/(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) for i in range(len(polynomial))]
    return np.array(coordinates)

def combos(a,b):
        f = math.factorial
        return f(a) / f(b) / f(a-b)

def v_polynomial(v):
    polynomial = v.tolist()
    return [(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) * polynomial[i] for i in range(len(polynomial))]

#

def C_v(roots):
    return polynomial_v(C_polynomial(roots))

def v_C(v):
    return polynomial_C(v_polynomial(v))

#

def v_SurfaceXYZ(v):
    return [c_xyz(c) for c in v_C(v)]

def SurfaceXYZ_v(XYZ):
    return C_v([xyz_c(xyz) for xyz in XYZ])

#

def v_SurfaceTXYZ(v):
    return [c_txyz(c) for c in v_C(v)]

def SurfaceTXYZ_v(TXYZ):
    return C_v([txyz_c(txyz) for txyz in TXYZ])

#

def v_SurfaceHERMITIAN(v):
    return [c_hermitian(c) for c in v_C(v)]

def SurfaceHERMITIAN_v(HERMITIAN):
    return C_v([hermitian_c(hermitian) for hermitian in HERMITIAN])

#

def hermitianMobiusEvolution(hermitian, mobius):
    return np.dot(mobius, np.dot(hermitian, dag(mobius)))

def txyzLorentzEvolution(txyz, lorentz):
    return np.dot(lorentz, txyz)

#

def v_hermitianMobiusEvolution_v(v, mobius):
    return SurfaceHERMITIAN_v([hermitianMobiusEvolution(hermitian, mobius) for hermitian in v_SurfaceHERMITIAN(v)])

#

def oneParameter_mobius(kind, parameter, acts_on):
    if kind == "parabolic_a":
        a = parameter
        if acts_on == "hermitian":
            return np.array([[1, a],                             [0, 1]])
        elif acts_on == "txyz":
            return np.array([[1 + (a**2)/2., a, 0, -1*(a**2)/2.],                             [a, 1, 0, -1*a],                             [0, 0, 1, 0],                             [(a**2)/2., a, 0, 1-(a**2)/2.]])
    elif kind == "parabolic_b":
        a = parameter
        if acts_on == "hermitian":
            return np.array([[1, I()*a],                             [0, 1]])
        elif acts_on == "txyz":
            return np.array([[1 + (a**2)/2., 0, a, -1*(a**2)/2.],                             [0, 1, 0, 0],                             [a, 0, 1, -1*a],                             [(a**2)/2., 0, a, 1-(a**2)/2.]])
    elif kind == "hyperbolic_z":
        b = parameter
        if acts_on == "hermitian":
            return np.array([[np.exp(b/2.), 0],                             [0, np.exp(-1*b/2.)]])
        elif acts_on == "txyz":
            return np.array([[np.cosh(b), 0, 0, np.sinh(b)],                             [0, 1, 0, 0],                             [0, 0, 1, 0],                             [np.sinh(b), 0, a, np.cosh(b)]])
    elif kind == "elliptic_x":
        theta = parameter
        if acts_on == "hermitian":
            return np.array([[np.exp(I()*theta/2.), 0],                             [0, np.exp(-1*I()*theta/2.)]])
        elif acts_on == "txyz":
            return np.array([[1, 0, 0, 0],                             [0, np.cos(theta), -1*np.sin(theta), 0],                             [0, np.sin(theta), np.cos(theta), 0],                             [0, 0, 0, 1]])
    elif kind == "elliptic_y":
        theta = parameter
        if acts_on == "hermitian":
            return np.array([[np.cos(theta/2.), -1*np.sin(theta/2.)],                             [np.sin(theta/2.), np.cos(theta/2)]])
        elif acts_on == "txyz":
            return np.array([[1, 0, 0, 0],                             [0, np.cos(theta), 0, np.sin(theta)],                             [0, 0, 1, 0],                             [0, -1*np.sin(theta), 0, np.cos(theta)]])
    elif kind == "elliptic_z":
        theta = parameter
        if acts_on == "hermitian":
            return np.array([[np.cos(theta/2.), I()*np.sin(theta/2.)],                             [I()*np.sin(theta/2.), np.cos(theta/2)]])
        elif acts_on == "txyz":
            return np.array([[1, 0, 0, 0],                             [0, 1, 0, 0],                             [0, 0, np.cos(theta), -1*np.sin(theta)],                             [0, 0, np.sin(theta), np.cos(theta)]])

#

def hermitian_unitary(hermitian, dt):
    return scipy.linalg.expm(-2*math.pi*I()*hermitian*dt)

def evolvev(v, energy, delta, sign=1):
    state = qutip.Qobj(v)
    unitary = qutip.Qobj(hermitian_unitary(energy, delta))
    if sign == -1:
        unitary = unitary.dag()
    u = unitary*state
    return u.full().T[0]

#

def random_v(n):
    return qutip.rand_ket(n).full().T[0]

def random_hermitian(n):
    return qutip.rand_herm(n).full()

def random_SurfaceHERMITIAN(n):
    return [c_hermitian(v_C(random_v(2))[0]) for i in range(n)]

# 

def offset(guy, center):
    t = [guy[0]+center[0]]
    xyz = (np.array(txyz_xyz(guy))+np.array(txyz_xyz(center))).tolist()
    return np.array(t+xyz)

def squish(txyz, amt):
    t = [txyz[0]]
    xyz = (amt*np.array(txyz_xyz(txyz))).tolist()
    return np.array(t+xyz)

def q_surfaceXYZ(q):
    return v_SurfaceXYZ(q.full().T[0])

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def q_txyz(q):
    old_dims = q.dims[:]
    n = q.shape[0]
    if q.shape[1] == 1:
        q.dims = [[n], [1]]
    else:
        q.dims = [[n], [n]]
    T = qutip.identity(n)
    X = qutip.jmat(0.5*n-0.5, "x")
    Y = qutip.jmat(0.5*n-0.5, "y")
    Z = qutip.jmat(0.5*n-0.5, "z")
    txyz = np.array([qutip.expect(T, q), qutip.expect(X, q), qutip.expect(Y, q), qutip.expect(Z, q)])
    q.dims = old_dims
    #return normalize(txyz)
    return txyz

def factorize(n):
    return list(primefac.primefac(n))

#

class Sphere:
    def __init__(self, v, center=np.array([1,0,0,0]), parent=None, delta=0.01):
        self.v, self.V = None, None
        self.center = None
        self.parent = parent
        self.delta = delta
        self.n, self.d, self.radius = None, None, None
        self.eigvals, self.eigvecs = None, None
        self.surfaceTXYZ, self.shellsTXYZ, self.exitsXYZ = None, None, None
        self.Q, self.Q_TXYZ, self.Q_Spheres = None, None, None
        self.update(v, center)
        self.vsphere = vpython.sphere(pos=vpython.vector(*txyz_xyz(self.center)),\
                                      radius=1.0,\
                                      color=vpython.color.hsv_to_rgb(vpython.vector(sigmoid(float(self.center[0])),1,1)),\
                                      opacity=0.3)
        self.vsphere_blackstars = None
        self.vshells, self.vshells_blackstars = None, None
        if self.v == None:
            self.vshells = [vpython.sphere(pos=vpython.vector(*txyz_xyz(self.center)),\
                                           radius=math.sqrt(float(l*np.conjugate(l))),\
                                           color=vpython.color.hsv_to_rgb(vpython.vector(sigmoid(float(self.center[0])),1,1)),\
                                           opacity=0.5) for l in self.eigvals]
            self.vshells_blackstars = [[vpython.sphere(pos=vpython.vector(*xyz),\
                                       radius=0.1,\
                                       color=vpython.color.black,\
                                       opacity=0.8) for xyz in shell] for shell in self.shellsXYZ]
        else:
            self.vsphere_blackstars = [vpython.sphere(pos=vpython.vector(*xyz),\
                                       radius=0.1,\
                                       color=vpython.color.black,\
                                       opacity=0.8) for xyz in self.surfaceXYZ]
        self.vwhitestars = [vpython.sphere(pos=vpython.vector(*xyz),\
                                       radius=0.1,\
                                       color=vpython.color.white,\
                                       opacity=0.8) for xyz in self.exitsXYZ]

    def update(self, v, center=None):
        if isinstance(center, np.ndarray):
            self.center = center
        self.n, self.d, self.radius = v.shape[0], len(v.dims[0]), v.norm()
        if v.shape[1] == 1:
            self.v = v
            self.V = self.v.ptrace(0)
            if self.d == 1:
                primes = factorize(self.n)
                self.v.dims = [primes, [1]*len(primes)]
                self.V.dims = [primes, primes]
        else:
            self.v = None
            self.V = v
            if self.d == 1:
                primes = factorize(self.n)
                self.V.dims = [primes, primes]
        self.eigvals, self.eigvecs = self.V.eigenstates()
        if self.v == None:
            self.shellsXYZ = [[self.eigvals[i]*np.array(xyz) for xyz in q_surfaceXYZ(self.eigvecs[i])] for i in range(n)]
        else:
            self.surfaceXYZ = [xyz for xyz in q_surfaceXYZ(self.v)]
        self.exitsXYZ = []
        for v in self.eigvecs:
            self.exitsXYZ.append(txyz_xyz(q_txyz(v))) 
        if self.d > 1 and False == True:    
            self.Q = [self.V.ptrace(i) for i in range(self.d)]
            self.Q_TXYZ = [offset(squish(q_txyz(q), 1), self.center) for q in self.Q]
            if self.Q_Spheres == None:
                self.Q_Spheres = [Sphere(self.Q[i], self.Q_TXYZ[i], parent=self) for i in range(self.d)]
            else:
                for i in range(self.d):
                    self.Q_Spheres[i].update(self.Q[i], self.Q_TXYZ[i])

    def draw(self):
        self.vsphere.pos = vpython.vector(*txyz_xyz(self.center))
        self.vsphere.radius = 1.0
        self.vsphere.color = vpython.color.hsv_to_rgb(vpython.vector(sigmoid(float(self.center[0])),1,1))
        if self.v == None:
            for vshell in self.vshells:
                vshell.pos = vpython.vector(*txyz_xyz(self.center))
                vshell.radius = math.sqrt(float(l*np.conjugate(l)))
                vshell.color = vpython.color.hsv_to_rgb(vpython.vector(sigmoid(float(self.center[0])),1,1))
            for i in range(len(self.vshells_blackstars)):
                for j in range(len(self.vshells_blackstars[i])):
                    self.vshells_blackstars[i][j].pos = vpython.vector(*self.shellsXYZ[i][j])
        else:
            for i in range(len(self.vsphere_blackstars)):
                self.vsphere_blackstars[i].pos = vpython.vector(*self.surfaceXYZ[i])
        for i in range(len(self.vwhitestars)):
            self.vwhitestars[i].pos = vpython.vector(*self.exitsXYZ[i])
        if self.Q_Spheres:
            for sphere in self.Q_Spheres:
                sphere.draw()
            
    def intersections(self):
        pass

    def mobius_transform(self, kind, delta):
        pass

    def rotate_spindle(self, q, pole, delta):
        pass

vpython.scene.width = 900
vpython.scene.height = 900
vpython.scene.range = 1.3
vpython.scene.forward = vpython.vector(-1, 0, 0)
vpython.scene.up = vpython.vector(0, 1, 0)

n = 4
sphere = Sphere(qutip.rand_ket(n))
energy = qutip.rand_herm(n)
while True:
    new_v = qutip.Qobj(evolvev(sphere.v.full().T[0], energy.full(), 0.007))
    new_v.dims = sphere.v.dims
    sphere.update(new_v)
    vpython.rate(1)
    sphere.draw()