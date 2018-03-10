
# coding: utf-8

# In[36]:


# QUANTUM ASTROLOGONOMY -- STREAMING MAJORANA'S STARS ALONG HOPF'S FIBERS
# (C) 2018 MATTHEW BENJAMIN WEISS

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
import geocoder
import vpython

import ephem
import ephem.stars

debug = False
STAR_NAMES = [star.split(",")[0] for star in ephem.stars.db.split("\n")][:-1]


# In[37]:


def I():
    return complex(0,1)

def dag(matrix):
    return np.conjugate(matrix.T)

def sigmoid(x):  
    return 2*(math.exp(-np.logaddexp(0, -x))-0.5)


# In[38]:


def c_xyz(c):
    if c == float('inf'):
        return [0,0,1]
    x = c.real
    y = c.imag
    return [(2*x)/(1.+(x**2)+(y**2)),            (2*y)/(1.+(x**2)+(y**2)),            (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]

def xyz_c(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    if z == 1:
        return float('inf') 
    else:
        return complex(x/(1-z), y/(1-z))


# In[39]:


if debug:
    c = complex(random.random(), random.random())
    xyz = c_xyz(c)
    c2 = xyz_c(xyz)
    print("c: %s\nxyz: %s\nc2: %s" % (c, xyz, c2))


# In[40]:


def xyz_txyz(xyz):
    return np.array([1]+xyz)

def txyz_xyz(txyz):
    return txyz_unitSphere(txyz)[1:].tolist()

def txyz_unitSphere(txyz):
    t, x, y, z = txyz[0], txyz[1], txyz[2], txyz[3]
    return np.array([t/math.sqrt(x**2+y**2+z**2), x/t, y/t, z/t])


# In[41]:


if debug:
    xyz = c_xyz(complex(random.random(), random.random()))
    txyz = xyz_txyz(xyz)
    xyz2 = txyz_xyz(txyz)
    print("xyz: %s\ntxyz: %s\nxyz2: %s" % (xyz, txyz, xyz2))


# In[42]:


def c_txyz(c):
    if c == float('inf'):
        return np.array([1,0,0,1])
    c = np.conjugate(c)
    u = c.real
    v = c.imag
    return np.array([u**2 + v**2 + 1, 2*u, -2*v, u**2 + v**2 - 1])

def txyz_c(txyz):
    return xyz_c(txyz_xyz(txyz))


# In[43]:


if debug: 
    txyz = c_txyz(c)
    c2 = txyz_c(txyz)
    print("c: %s\ntxyz: %s\nc2: %s" % (c, txyz, c2))


# In[44]:


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
    z = scalarProduct(hermitian, qutip.sigmaz().full(), ).real
    return np.array([t, x, y, z])

def txyz_spacetimeInterval(txyz):
    t, x, y, z = txyz[0], txyz[1], txyz[2], txyz[3]
    return t**2 - x**2 - y**2 - z**2

def hermitian_spacetimeInterval(hermitian):
    return np.linalg.det(hermitian)


# In[45]:


def c_hermitian(c):
    if c == float('inf'):
        return txyz_hermitian(np.array([1,0,0,1]))
    u = c.real
    v = c.imag
    return np.conjugate(np.array([[u**2 + v**2, u+I()*v],[u-I()*v, 1]]))

def hermitian_c(hermitian):
    return txyz_c(hermitian_txyz(hermitian))


# In[46]:


if debug:
    c = complex(random.random(), random.random())
    xyz = c_xyz(c)
    hermitian = xyz_hermitian(xyz)
    xyz2 = hermitian_xyz(hermitian)
    print("xyz:\n%s\nhermitian:\n%s\nxyz2:\n%s" % (xyz, hermitian, xyz2))
    print()
    print(hermitian_txyz(hermitian))
    print(txyz_xyz(hermitian_txyz(hermitian)))
    print()
    txyz = c_txyz(c)
    hermitian = txyz_hermitian(txyz)
    txyz2 = hermitian_txyz(hermitian)
    print("txyz:\n%s\nhermitian:\n%s\ntxyz2:\n%s" % (txyz, hermitian, txyz2))
    print()
    print(hermitian_xyz(hermitian))
    print(txyz_xyz(txyz2))
    print()
    c = complex(random.random(), random.random())
    hermitian = c_hermitian(c)
    c2 = hermitian_c(hermitian)
    print("c:\n%s\nhermitian:\n%s\nc2:\n%s" % (c, hermitian, c2))


# In[47]:


def c_altitudeAzimuth(c):
    r, th = cmath.polar(c)
    if r == 0:
        return [(math.pi/2.) - math.pi, th]
    return [(math.pi/2.) - 2*math.atan2(1,r), th]

def altitudeAzimuth_c(altitude, azimuth):
    zenith = (math.pi/2.) - altitude
    if zenith == 0:
        return float('inf')
    return cmath.rect(math.sin(zenith)/(1-math.cos(zenith)), azimuth)


# In[48]:


def hermitian_altitudeAzimuth(hermitian):
    return c_altitudeAzimuth(hermitian_c(hermitian))

def altitudeAzimuth_hermitian(altitude, azimuth):
    return c_hermitian(altitudeAzimuth_c(altitude, azimuth))


# In[49]:


if debug:
    c = complex(random.random(), random.random())
    alt_az = c_altitudeAzimuth(c)
    c2 = altitudeAzimuth_c(*alt_az)
    print("c: %s\nalt_az: %s\nc2: %s" % (c, alt_az, c2))
    print()
    alt, az = random.uniform(-1*math.pi, math.pi), random.uniform(-2*math.pi, 2*math.pi)
    c = altitudeAzimuth_c(alt, az)
    alt2, az2 = c_altitudeAzimuth(c)
    print("alt_az: %s\nc: %s\nalt_az1: %s" % ((alt,az), c, (alt2,az2)))
    print()
    alt, az = random.uniform(0, math.pi), random.uniform(0, 2*math.pi)
    hermitian = altitudeAzimuth_hermitian(alt, az)
    alt2, az2 = hermitian_altitudeAzimuth(hermitian)
    print("alt_az: %s\nhermitian: %s\nalt_az2: %s" % ((alt,az), hermitian, (alt2,az2)))


# In[50]:


def altitudeAzimuth_xyz(altitude, azimuth):
    return c_xyz(altitudeAzimuth_c(altitude, azimuth))

def xyz_altitudeAzimuth(xyz):
    return c_altitudeAzimuth(xyz_c(xyz))


# In[51]:


def latitudeLongitude_xyz(latitude, longitude):
    latitude = math.radians(-1*latitude)
    longitude = math.radians(-1*longitude)
    x = math.cos(latitude)*math.cos(longitude)
    y = math.cos(latitude)*math.sin(longitude)
    z = math.sin(latitude)
    return [x, y, z]

def xyz_latitudeLongitude(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    r = 1
    latitude = math.asin(z/r)*(180/math.pi)
    longitude = None
    if x > 0:
        longitude = math.atan(y/x)*(180/math.pi)
    elif y > 0:
        longitude = math.atan(y/x)*(180/math.pi) + 180
    else:
        longitude = math.atan(y/x)*(180/math.pi) - 180
    return (latitude, longitude)


# In[52]:


if debug:
    lat, long = random.uniform(-90., 90), random.uniform(-180, 180)
    xyz = latitudeLongitude_xyz(lat, long)
    lat2, long2 = xyz_latitudeLongitude(xyz)
    print("lat_long: %s\nxyz: %s\nlat_long2: %s" % ((lat,long), xyz, (lat2,long2)))
    print()
    xyz = c_xyz(complex(random.random(), random.random()))
    lat, long = xyz_latitudeLongitude(xyz)
    xyz2 = latitudeLongitude_xyz(lat, long)
    print("xyz: %s\nlat_long: %s\nxyz2: %s" % (xyz, (lat,long), xyz2))


# In[53]:


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


# In[54]:


def polynomial_v(polynomial):
    coordinates = [polynomial[i]/(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) for i in range(len(polynomial))]
    return np.array(coordinates)

def combos(a,b):
        f = math.factorial
        return f(a) / f(b) / f(a-b)

def v_polynomial(v):
    polynomial = v.tolist()
    return [(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) * polynomial[i] for i in range(len(polynomial))]


# In[55]:


def C_v(roots):
    return polynomial_v(C_polynomial(roots))

def v_C(v):
    return polynomial_C(v_polynomial(v))


# In[56]:


if debug:
    def random_v(n):
        return qutip.rand_ket(n).full().T[0]
    v = random_v(4)
    p = v_polynomial(v)
    v2 = polynomial_v(p)
    print("v:\n%s\np:\n%s\nv2:\n%s" % (v, p, v2))
    print()
    v = random_v(4)
    v = np.array([x/v[0] for x in v])
    p = v_polynomial(v)
    C = polynomial_C(p)
    p2 = C_polynomial(C)
    print("p:\n%s\nC:\n%s\np2:\n%s" % (p, C, p2))
    print()
    v = random_v(4)
    C = v_C(v)
    v2 = C_v(C)
    u = [x/v[0] for x in v]
    print("v:\n%s\nC:\n%s\nv2:\n%s\nv/v[0]:\n%s" % (v, C, v2, u))


# In[57]:


def v_SurfaceXYZ(v):
    return [c_xyz(c) for c in v_C(v)]

def SurfaceXYZ_v(XYZ):
    return C_v([xyz_c(xyz) for xyz in XYZ])


# In[58]:


def v_ALTITUDEaZIMUTH(v):
    return [c_altitudeAzimuth(c) for c in v_C(v)]

def ALTITUDEaZIMUTH_v(ALTITUDEaZIMUTH):
    return C_v([altitudeAzimuth_c(*altitudeAzimuth) for altitudeAzimuth in ALTITUDEaZIMUTH])


# In[59]:


def v_SurfaceHERMITIAN(v):
    return [c_hermitian(c) for c in v_C(v)]

def SurfaceHERMITIAN_v(HERMITIAN):
    return C_v([hermitian_c(hermitian) for hermitian in HERMITIAN])


# In[60]:


if debug:
    def random_v(n):
        return qutip.rand_ket(n).full().T[0]
    v = random_v(4)
    v = np.array([x/v[0] for x in v])
    H = v_SurfaceHERMITIAN(v)
    v2 = SurfaceHERMITIAN_v(H)
    print("v:\n%s\nH:\n%s\nv2:\n%s" % (v, H, v2))


# In[61]:


def hermitianMobiusEvolution(hermitian, mobius):
    return np.dot(mobius, np.dot(hermitian, dag(mobius)))

def txyzLorentzEvolution(txyz, lorentz):
    return np.dot(lorentz, txyz)


# In[62]:


def v_hermitianMobiusEvolution_v(v, mobius):
    return SurfaceHERMITIAN_v([hermitianMobiusEvolution(hermitian, mobius) for hermitian in v_SurfaceHERMITIAN(v)])


# In[63]:


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


# In[64]:


def v_InteriorHERMITIAN(v):
    q = qutip.Qobj(v)
    q.dims = [[2,2,2],[1,1,1]]
    traces = [q.ptrace(i).full() for i in range(3)]
    return traces

def upgradeOperator(operator, i):
    operator = qutip.Qobj(operator)
    total_operator = None
    if i == 0:
        total_operator = qutip.tensor(operator, qutip.identity(2), qutip.identity(2))
    elif i == 1:
        total_operator = qutip.tensor(qutip.identity(2), operator, qutip.identity(2))
    elif i == 2:
        total_operator = qutip.tensor(qutip.identity(2), qutip.identity(2), operator)
    return total_operator.full()


# In[65]:


def hermitian_unitary(hermitian, dt):
    return scipy.linalg.expm(-2*math.pi*I()*hermitian*dt)

def evolvev(v, energy, delta, sign):
    state = qutip.Qobj(v)
    unitary = qutip.Qobj(hermitian_unitary(energy, delta))
    if sign == -1:
        unitary = unitary.dag()
    u = unitary*state
    return u.full().T[0]


# In[66]:


def random_v(n):
    return qutip.rand_ket(n).full().T[0]

def random_hermitian(n):
    return qutip.rand_herm(n).full()

def random_SurfaceHERMITIAN(n):
    return [c_hermitian(v_C(random_v(2))[0]) for i in range(n)]


# In[67]:


class Sphere:
    def __init__(self, address, delta=0.01):
        self.state = random_v(8)
        self.energy = random_hermitian(8)
        self.fixed_stars = random_SurfaceHERMITIAN(len(STAR_NAMES))
        
        self.time = None
        self.latitude = 0
        self.longitude = 0
        self.address = address
        
        self.delta = delta
        self.time_delta = datetime.timedelta(minutes=1)
        
        self.astronomy_on = False
        self.astronomical_evolution_on = False
        
        self.hamiltonian_on = False
        self.hamiltonian_sign = 1
        
        self.pole = (0, "interior")
        self.touched = True
        self.virgin = True
        self.trail_mode = 0

        self.vinit()
        
    def vinit(self):
        vpython.scene.width = 900
        vpython.scene.height = 900
        vpython.scene.range = 1.3
        vpython.scene.forward = vpython.vector(-1, 0, 0)
        vpython.scene.up = vpython.vector(0, 1, 0)
        
        self.vsphere = vpython.sphere(pos=vpython.vector(0,0,0), radius=1, color=vpython.color.blue, opacity=0.5)
        self.vearth = vpython.sphere(pos=vpython.vector(0,0,0), radius=0.1, color=vpython.color.cyan, opacity=0.5, emissive=True)
        self.vobserver = vpython.sphere(pos=vpython.vector(0,0,0), radius=0.01, color=vpython.color.yellow, opacity=0.5, emissive=True, make_trail=True)

        self.vstamp = vpython.label(pos=vpython.vector(0,0,0), text="", height=10, opacity=0.6, visible=False)
        self.vfixed_stars = [vpython.sphere(radius=0.01, emissive=True, color=vpython.color.white, make_trail=False) for i in range(len(STAR_NAMES))]

        self.vplanets = [vpython.sphere(radius=0.1, emissive=True, make_trail=False) for i in range(7)]
        self.vplanets[0].color = vpython.color.yellow
        self.vplanets[1].color = vpython.color.white
        self.vplanets[2].color = vpython.color.blue
        self.vplanets[3].color = vpython.color.green
        self.vplanets[4].color = vpython.color.red
        self.vplanets[5].color = vpython.color.orange
        self.vplanets[6].color = vpython.color.gray(0.5)
        
        self.vqubits = [vpython.sphere(radius=0.1, emissive=True, make_trail=False) for i in range(3)]
        self.veigs = [vpython.sphere(color=vpython.color.black, radius=0.05, emissive=True, make_trail=False) for i in range(6)]
        self.vlines = [vpython.curve(pos=[vpython.vector(0,0,0), vpython.vector(0,0,0)]) for i in range(3)]
        self.vlines[0].color = vpython.color.red
        self.vlines[1].color = vpython.color.green
        self.vlines[2].color = vpython.color.blue
    
    def astronomical_update(self):
        observer = ephem.Observer()
        observer.date = self.time
        observer.lat = self.latitude
        observer.lon = self.longitude
        ephem_planets = [ephem.Sun(observer), ephem.Moon(observer), ephem.Mercury(observer),                         ephem.Venus(observer), ephem.Mars(observer), ephem.Jupiter(observer), ephem.Saturn(observer)]
        ephem_stars = [ephem.star(star_name, observer) for star_name in STAR_NAMES]
        self.fixed_stars = [altitudeAzimuth_hermitian(fixed_star.alt, fixed_star.az) for fixed_star in ephem_stars]
        self.state = ALTITUDEaZIMUTH_v([(planet.alt, planet.az) for planet in ephem_planets])
        self.touched = True
      
    def revolve(self):
        if self.astronomy_on:
            if self.astronomical_evolution_on:
                self.astronomical_update()
                self.time += self.time_delta
        if self.hamiltonian_on:
            self.state = evolvev(self.state, self.energy, self.delta, self.hamiltonian_sign)
            self.touched = True
        if self.touched:
            planets_XYZ = v_SurfaceXYZ(self.state)
            if not self.virgin:
                planets_XYZ = self.fix(planets_XYZ) 
            if self.virgin:
                self.virgin = False
            distinguishable_qubits = v_InteriorHERMITIAN(self.state)
            qubits_TXYZ = [hermitian_txyz(qubit) for qubit in distinguishable_qubits]
            eigs_XYZ = []
            for q in distinguishable_qubits:
                eigenvalues, eigenvectors = np.linalg.eig(q)
                for v in eigenvectors:
                    eigs_XYZ.append(v_SurfaceXYZ(v)[0])
            fixed_XYZ = [hermitian_xyz(fixed_star) for fixed_star in self.fixed_stars]
            self.draw(planets_XYZ, qubits_TXYZ, eigs_XYZ, fixed_XYZ)
            touched = False

    def draw(self, planets_XYZ, qubits_TXYZ, eigs_XYZ, fixed_XYZ):
        vpython.rate(200)
        if self.astronomy_on:
            self.vstamp.text = self.time.strftime("%c")+"\nlat: %.2f lng: %.2f" % (self.latitude, self.longitude)
        self.vobserver.pos = vpython.vector(*[0.1*coord for coord in latitudeLongitude_xyz(self.latitude, self.longitude)])
        for i in range(len(self.vplanets)):
            self.vplanets[i].pos = vpython.vector(*planets_XYZ[i])
        for i in range(len(self.vqubits)):
            txyz = qubits_TXYZ[i]
            xyz = txyz_xyz(txyz)
            TXYZ = txyz_unitSphere(txyz)
            self.vqubits[i].color = vpython.color.hsv_to_rgb(vpython.vector(sigmoid(float(txyz[0])),1,1))
            self.vqubits[i].pos = vpython.vector(*TXYZ[1:])
        for i in range(len(self.veigs)):
            self.veigs[i].pos = vpython.vector(*eigs_XYZ[i])
        for i in range(len(self.vlines)):
            self.vlines[i].modify(0, pos=vpython.vector(*eigs_XYZ[2*i]))
            self.vlines[i].modify(1, pos=vpython.vector(*eigs_XYZ[2*i+1]))
        for i in range(len(self.vfixed_stars)):
            self.vfixed_stars[i].pos = vpython.vector(*fixed_XYZ[i])
      
    def fix(self, planets_XYZ):
        ordering = []
        for i in range(7):
            vplanet = self.vplanets[i]
            x, y, z = vplanet.pos.x, vplanet.pos.y, vplanet.pos.z
            current_xyz = np.array([x, y, z])
            lowest_distance = None
            winners = []
            for j in range(7):
                new_xyz = np.array(planets_XYZ[j])
                distance = np.linalg.norm(current_xyz-new_xyz)
                if lowest_distance == None or distance < lowest_distance:
                    lowest_distance = distance
                    winners.append(j)
            for m in range(len(winners)-1, -1, -1):
                if winners[m] not in ordering:
                    ordering.append(winners[m])
                    break
                if m == 0:
                    for r in range(i, 7):
                        if r not in ordering:
                            ordering.append(i)
                            break
                        if r == 6:
                            for p in range(i, -1, -1):
                                if p not in ordering:
                                    ordering.append(p)
                                    break
        return [planets_XYZ[i] for i in ordering]
    
    #######
    
    def set_pole(self, i, kind):
        self.pole = i, kind
        
    def collapse(self, q, sign):
        total_measurement = upgradeOperator(v_InteriorHERMITIAN(self.state)[q], q)        
        eigenvalues, eigenvectors = np.linalg.eig(total_measurement)
        projection = None
        if sign == -1:
            projection = sum([eigenvectors[i] for i in range(4)])
        elif sign == 1:
            projection = sum([eigenvectors[i] for i in range(4, 8)])
        self.state = projection
        self.touched = True
    
    #######
    
    def sphere_step(self, kind, sign):
        if kind == "latitude":
            self.latitude += sign*self.delta
            if self.astronomy_on:
                self.astronomical_update()
        elif kind == "longitude":
            self.longitude += sign*self.delta
            if self.astronomy_on:
                self.astronomical_update()
        elif kind == "step":
            self.delta += sign*0.001
            self.time_delta = datetime.timedelta(seconds=(self.time_delta.seconds+sign*60))
        self.touched = True
        
    def mobius_step(self, kind):
        mobius = oneParameter_mobius(kind, self.delta, "hermitian")
        self.state = v_hermitianMobiusEvolution_v(self.state, mobius)
        for i in range(len(STAR_NAMES)):
            self.fixed_stars[i] = hermitianMobiusEvolution(self.fixed_stars[i], mobius)
        self.touched = True
        
    def qubit_step(self, q, sign):
        pole, kind = self.pole
        operator = None
        if kind == "surface":
            operator = v_SurfaceHERMITIAN(self.state)[pole]
        elif kind == "interior":
            operator = v_InteriorHERMITIAN(self.state)[pole]
        operator = hermitian_unitary(operator, self.delta)
        if sign == -1:
            operator = dag(operator)
        full_operator = upgradeOperator(operator, q)
        self.state = np.dot(full_operator, self.state)
        self.touched = True

    #######
    
    def toggle_hamiltonian(self):
        if self.hamiltonian_on:
            self.hamiltonian_on = False
        else:
            self.hamiltonian_on = True

    def toggle_hamiltonian_sign(self):
        if self.hamiltonian_sign == 1:
            self.hamiltonian_sign = -1
        elif self.hamiltonian_sign == -1:
            self.hamiltonian_sign = 1
    
    #######
    
    def random_state(self):
        self.state = random_v(8)
        self.fixed_stars = random_SurfaceHERMITIAN(len(STAR_NAMES))
        for vfixed_star in self.vfixed_stars:
            vfixed_star.color = vpython.color.white
        self.touched = True
        self.virgin = True
        self.refresh()
        
    def random_hamiltonian(self):
        self.energy = random_hermitian(8)
    
    #######
    
    def reorient(self):
        vpython.scene.forward = vpython.vector(-1, 0, 0)
        vpython.scene.up = vpython.vector(0, 1, 0)
        self.touched = True
        
    def refresh(self):
        self.vobserver.clear_trail()
        for vplanet in self.vplanets:
            vplanet.clear_trail()
        for vqubit in self.vqubits:
            vqubit.clear_trail()
        for vfixed_star in self.vfixed_stars:
            vfixed_star.clear_trail()
        
    def toggle_trail_mode(self):
        self.refresh()
        self.trail_mode += 1
        if self.trail_mode > 5:
            self.trail_mode = 0
        if self.trail_mode == 0: # no trails
            for vplanet in self.vplanets:
                vplanet.make_trail = False
            for vqubit in self.vqubits:
                vqubit.make_trail = False
            for vfixed_star in self.vfixed_stars:
                vfixed_star.make_trail = False
        elif self.trail_mode == 1: # interior trails
            for vplanet in self.vplanets:
                vplanet.make_trail = False
            for vqubit in self.vqubits:
                vqubit.make_trail = True
            self.vqubits[0].trail_color = vpython.color.red
            self.vqubits[1].trail_color = vpython.color.green
            self.vqubits[2].trail_color = vpython.color.blue
            for vfixed_star in self.vfixed_stars:
                vfixed_star.make_trail = False
        elif self.trail_mode == 2: # planet trails
            for vplanet in self.vplanets:
                vplanet.make_trail = True
            self.vplanets[0].trail_color = vpython.color.yellow
            self.vplanets[1].trail_color = vpython.color.white
            self.vplanets[2].trail_color = vpython.color.blue
            self.vplanets[3].trail_color = vpython.color.green
            self.vplanets[4].trail_color = vpython.color.red
            self.vplanets[5].trail_color = vpython.color.orange
            self.vplanets[6].trail_color = vpython.color.gray(0.5)
            for vqubit in self.vqubits:
                vqubit.make_trail = False
            for vfixed_star in self.vfixed_stars:
                vfixed_star.make_trail = False
        elif self.trail_mode == 3: # fixed star trails
            for vplanet in self.vplanets:
                vplanet.make_trail = False
            for vqubit in self.vqubits:
                vqubit.make_trail = False
            for vfixed_star in self.vfixed_stars:
                vfixed_star.make_trail = True
        elif self.trail_mode == 4: # planet and fixed star trails
            for vplanet in self.vplanets:
                vplanet.make_trail = True
            self.vplanets[0].trail_color = vpython.color.yellow
            self.vplanets[1].trail_color = vpython.color.white
            self.vplanets[2].trail_color = vpython.color.blue
            self.vplanets[3].trail_color = vpython.color.green
            self.vplanets[4].trail_color = vpython.color.red
            self.vplanets[5].trail_color = vpython.color.orange
            self.vplanets[6].trail_color = vpython.color.gray(0.5)
            for vqubit in self.vqubits:
                vqubit.make_trail = False
            for vfixed_star in self.vfixed_stars:
                vfixed_star.make_trail = True
        elif self.trail_mode == 5: # interior and planet and fixed star trails
            for vplanet in self.vplanets:
                vplanet.make_trail = True    
            self.vplanets[0].trail_color = vpython.color.yellow
            self.vplanets[1].trail_color = vpython.color.white
            self.vplanets[2].trail_color = vpython.color.blue
            self.vplanets[3].trail_color = vpython.color.green
            self.vplanets[4].trail_color = vpython.color.red
            self.vplanets[5].trail_color = vpython.color.orange
            self.vplanets[6].trail_color = vpython.color.gray(0.5)
            for vqubit in self.vqubits:
                vqubit.make_trail = True
            self.vqubits[0].trail_color = vpython.color.red
            self.vqubits[1].trail_color = vpython.color.green
            self.vqubits[2].trail_color = vpython.color.blue
            for vfixed_star in self.vfixed_stars:
                vfixed_star.make_trail = True
                
    #######
    
    def load_astronomy(self):
        self.latitude, self.longitude = None, None
        remaining_tries = 10
        while self.latitude == None and self.longitude == None:
            try:
                self.latitude, self.longitude = geocoder.google(self.address).latlng
            except:
                self.latitude, self.longitude = None, None
            if remaining_tries == 0:
                break
            remaining_tries -= 1
        self.time = datetime.datetime.now()
        self.vfixed_stars[STAR_NAMES.index("Sirius")].color = vpython.color.red
        self.vfixed_stars[STAR_NAMES.index("Betelgeuse")].color = vpython.color.yellow
        self.vfixed_stars[STAR_NAMES.index("Rigel")].color = vpython.color.yellow
        self.vfixed_stars[STAR_NAMES.index("Bellatrix")].color = vpython.color.yellow
        self.vfixed_stars[STAR_NAMES.index("Mintaka")].color = vpython.color.yellow
        self.vfixed_stars[STAR_NAMES.index("Alnilam")].color = vpython.color.yellow
        self.vfixed_stars[STAR_NAMES.index("Alnitak")].color = vpython.color.yellow
        self.vfixed_stars[STAR_NAMES.index("Saiph")].color = vpython.color.yellow
        self.vfixed_stars[STAR_NAMES.index("Polaris")].color = vpython.color.blue
        
    #######
    
    def toggle_astronomy(self):
        if self.astronomy_on:
            self.astronomy_on = False
            self.astronomical_evolution_on = False
            self.vstamp.visible = False
        else:
            self.load_astronomy()
            self.virgin = True
            self.astronomy_on = True
            self.astronomical_evolution_on = True
            self.vstamp.visible = True
        self.refresh()
        
    def toggle_astronomical_time(self):
        if self.astronomical_evolution_on:
            self.astronomical_evolution_on = False
        else:
            self.astronomical_evolution_on = True


# In[68]:


sphere = Sphere("476 Jefferson Street, Brooklyn")


# In[69]:


def mouse(event):
    global sphere
    star = vpython.scene.mouse.pick
    if star in sphere.vplanets:
        sphere.set_pole(sphere.vplanets.index(star), "surface")
    elif star in sphere.vqubits:
        sphere.set_pole(sphere.vqubits.index(star), "interior")
    elif star in sphere.veigs:
        i = sphere.veigs.index(star)
        if i == 0:
            sphere.collapse(0, -1)
        elif i == 1:
            sphere.collapse(0, 1)
        elif i == 2:
            sphere.collapse(1, -1)
        elif i == 3:
            sphere.collapse(1, 1)
        elif i == 4:
            sphere.collapse(2, -1)
        elif i == 5:
            sphere.collapse(2, 1)
            
vpython.scene.bind('click', mouse)


# In[70]:


def keyboard(event):
    global sphere
    key = event.key
    if key == "a":
        sphere.sphere_step("latitude", -1)
    elif key == "d":
        sphere.sphere_step("latitude", 1)
    elif key == "w":
        sphere.sphere_step("longitude", 1)
    elif key == "s":
        sphere.sphere_step("longitude", -1)
    elif key == "z":
        sphere.sphere_step("step", -1)
    elif key == "x":
        sphere.sphere_step("step", 1)
    elif key == "f":
        sphere.mobius_step("parabolic_a")
    elif key == "h":
        sphere.mobius_step("parabolic_b")
    elif key == "t":
        sphere.mobius_step("elliptic_x")
    elif key == "g":
        sphere.mobius_step("elliptic_y")
    elif key == "v":
        sphere.mobius_step("hyperbolic_z")
    elif key == "b":
        sphere.mobius_step("elliptic_z")
    elif key == "j":
        sphere.qubit_step(0, -1)
    elif key == "l":
        sphere.qubit_step(0, 1)
    elif key == "i":
        sphere.qubit_step(1, 1)
    elif key == "k":
        sphere.qubit_step(1, -1)
    elif key == "n":
        sphere.qubit_step(2, -1)
    elif key == "m":
        sphere.qubit_step(2, 1)
    elif key == "e":
        sphere.toggle_hamiltonian()
    elif key == "r":
        sphere.toggle_hamiltonian_sign()
    elif key == "y":
        sphere.random_state()
    elif key == "u":
        sphere.random_hamiltonian()
    elif key == "o":
        sphere.reorient()
    elif key == "p":
        sphere.toggle_trail_mode()
    elif key == "q":
        sphere.toggle_astronomy()
    elif key == "c":
        sphere.toggle_astronomical_time()

vpython.scene.bind('keydown', keyboard)


# In[ ]:


while True:
    sphere.revolve()

