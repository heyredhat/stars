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

##################################################################################################################

im = complex(0,1)
X = np.array([[0,1],\
              [1,0]])
Y = np.array([[0,-1*im],\
               [im,0]])
Z = np.array([[1,0],\
              [0,-1]])

##################################################################################################################

def TwoAngles_to_PureQubit(latitude, longitude):
    return np.array([[math.cos(latitude/2.)],\
                     [cmath.exp(complex(0,1)*longitude)*math.sin(latitude/2.)]])

def TwoAngles_to_R3(latitude, longitude):
    x = math.sin(latitude)*math.cos(longitude)
    y = math.sin(latitude)*math.sin(longitude)
    z = math.cos(latitude)
    return np.array([[x],\
                     [y],\
                     [z]])

def PureQubit_to_QubitDM(pureQubit):
    return np.outer(pureQubit, np.conjugate(pureQubit).T)

def TwoAngles_to_QubitDM(latitude, longitude):
    return (1./2.)*np.array([[1+math.cos(latitude), cmath.exp(-1*im*longitude)*math.sin(latitude)],\
                             [cmath.exp(im*longitude)*math.sin(latitude), 1-math.cos(latitude)]])

def R3_to_QubitDM(r3):
    x, y, z = r3.T[0]
    return (1./2)*(np.eye(2) + x*X + y*Y + z*Z)

def PureQubits_to_InnerSquared(pureQubitA, pureQubitB):
    amplitude = np.inner(pureQubitA, np.conjugate(pureQubitB).T)
    return amplitude*np.conjugate(amplitude)

def QubitDMs_to_InnerSquared(qubitDMA, qubitDMB):
    return np.trace(np.dot(qubitDMA, qubitDMB))

def R3s_to_InnerSquared(r3A, r3B):
    return (1./2.)*(1 + np.inner(r3A, r3B.T))

def rotation(around_axis, angle):
    return R3_to_QubitDM(around_axis) + cmath.exp(im*angle)*R3_to_QubitDM(-1*around_axis)

def PureQubit_to_C(pureQubit):
    alpha, beta = pureQubit.T[0]
    if beta == 0:
        return float('Inf')
    else:
        return alpha/beta
    
def TwoAngles_to_C(latitude, longitude):
    if math.sin(latitude) == 0:
        return float('Inf')
    else:
        cotangent = math.cos(latitude/2.)/math.sin(latitude/2.)
        return cmath.exp(-1*im*longitude)*cotangent

def R3_to_C(r3):
    x, y, z = r3.T[0]
    if z == 1:
        return float("Inf")
    return (x-im*y[1])/(1-z)


def R4_to_C2(r4):
    x, y, z, w = r4.T[0]
    return np.array([[complex(x, y)],\
                     [complex(z, w)]])

def ThreeAngles_to_R4(first_latitude, second_latitude, longitude):
    x = math.sin(first_latitude)*math.sin(second_latitude)*math.sin(longitude)
    y = math.sin(first_latitude)*math.sin(second_latitude)*math.cos(longitude)
    z = math.sin(first_latitude)*math.cos(second_latitude)
    w = math.cos(first_latitude)
    return np.array([[x],\
                     [y],\
                     [z],\
                     [w]])

def C2_to_hopfCircle(c2, angle):
    return cmath.exp(im*angle)*c2

def R4_to_hopfCircle(r4, angle):
    transformation = np.array([[math.cos(angle), -1*math.sin(angle), 0, 0],\
                               [math.sin(angle), math.cos(angle), 0, 0],\
                               [0, 0, math.cos(angle), -1*math.sin(angle)],\
                               [0, 0, math.sin(angle), math.cos(angle)]])
    return np.dot(transformation,r4)

def R4_to_R3(r4):
    x, y, z, w = r4.T[0]
    xyz = np.array([[x],\
                    [y],\
                    [z]])
    if w == 1:
        return (float("Inf"), xyz)
    else:
        return (1./(1.-w))*xyz

def C2_to_C(c2):
    alpha, beta = c2.T[0]
    if beta == 0:
        return float("Inf")
    return alpha/beta

def C_to_R3(c):
    if c == float("Inf"):
        return np.array([[0],\
                         [0],\
                         [1]])
    x = (c+np.conjugate(c))/(c*np.conjugate(c)+1)
    y = im*(c-np.conjugate(c))/(c*np.conjugate(c)+1)
    z = (c*np.conjugate(c)-1)/(c*np.conjugate(c)+1)
    return np.array([[x],\
                     [y],\
                     [z]])

def C2_to_R3(c2):
    alpha, beta = c2.T[0]
    x = 2*(np.conjugate(alpha)*beta).real
    y = 2*(np.conjugate(alpha)*beta).imag
    z = (alpha*np.conjugate(alpha)-beta*np.conjugate(beta))
    return np.inner(c2,np.conjugate(c2).T)*(np.array([[x],\
                     [y],\
                     [z]]))

##################################################################################################################

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def QubitDM_to_R4(qubitDM):
    eigenvalues, eigenvectors = qubitDM.eigenstates()
    star = np.array([[qutip.expect(qutip.sigmax(), eigenvectors[0])],\
                      [qutip.expect(qutip.sigmay(), eigenvectors[0])],\
                      [qutip.expect(qutip.sigmaz(), eigenvectors[0])],\
                      [qutip.expect(qutip.identity(2), eigenvectors[0])]])
    anti_star = np.array([[qutip.expect(qutip.sigmax(), eigenvectors[1])],\
                          [qutip.expect(qutip.sigmay(), eigenvectors[1])],\
                          [qutip.expect(qutip.sigmaz(), eigenvectors[1])],\
                          [qutip.expect(qutip.identity(2), eigenvectors[1])]])
    return (1./2.)*(eigenvalues[0]*star + eigenvalues[1]*anti_star)

def spin_operators(n):
    spin = (n-1.)/2.
    return {"X": qutip.jmat(spin, "x"),\
            "Y": qutip.jmat(spin, "y"),\
            "Z": qutip.jmat(spin, "z"),\
            "+": qutip.jmat(spin, "+"),\
            "-": qutip.jmat(spin, "-")}

def spin_axis(n, state):
    spin_ops = spin_operators(n)
    return [qutip.expect(spin_ops["X"], state),\
            qutip.expect(spin_ops["Y"], state),\
            qutip.expect(spin_ops["Z"], state)]

##################################################################################################################

vpython.scene.width = 500
vpython.scene.height = 500

n_qubits = 2

dt = 0.001
n_points = 50

state = qutip.rand_ket(2**n_qubits)
energy = qutip.rand_herm(2**n_qubits)
unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*im*energy.full()*dt))
qubits = None
qubit_colors = [vpython.vector(random.random(), random.random(), random.random()) for i in range(n_qubits)]
vsphere = vpython.sphere(pos=vpython.vector(0,0,0),\
                         radius=1.0,\
                         color=vpython.color.blue,\
                         opacity=0.4)
vspin = vpython.arrow(shaftwidth=0.01, headwidth=0.001, headlength=0.001)
vstars = [vpython.sphere(radius=0.05,\
                         color=vpython.color.white,\
                         opacity=0.8,\
                         emissive=True) for i in range((2**n_qubits)-1)]
vbases = [vpython.sphere(radius=0.1,\
                       color=qubit_colors[i],\
                       opacity=0.7,\
                       emissive=True,\
                       make_trail=False) for i in range(n_qubits)]
varrows = [vpython.arrow(color=qubit_colors[i]) for i in range(n_qubits)]
vfibers = [vpython.curve(pos=[vpython.vector(0,0,0) for i in range(n_points)],\
                         color=qubit_colors[i]) for i in range(n_qubits)]

def recreate():
    global n_qubits
    global state
    global energy
    global unitary
    global qubits
    global qubit_colors
    global vsphere
    global vstars
    global vbases
    global varrows
    global vfibers
    global vspin
    state = qutip.rand_ket(2**n_qubits)
    energy = qutip.rand_herm(2**n_qubits)
    unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*im*energy.full()*dt))
    qubits = None
    qubit_colors = [vpython.vector(random.random(), random.random(), random.random()) for i in range(n_qubits)]
    vsphere.visible = False
    del vsphere
    vsphere = vpython.sphere(pos=vpython.vector(0,0,0),\
                             radius=1.0,\
                             color=vpython.color.blue,\
                             opacity=0.4)
    vspin.visible = False
    del vspin
    vspin = vpython.arrow(shaftwidth=0.01, headwidth=0.001, headlength=0.001)
    for vstar in vstars:
        vstar.visible = False
        del vstar
    vstars = [vpython.sphere(radius=0.05,\
                             color=vpython.color.white,\
                             opacity=0.8,\
                             emissive=True) for i in range((2**n_qubits)-1)]
    for vbase in vbases:
        vbase.visible = False
        del vbase
    vbases = [vpython.sphere(radius=0.1,\
                           color=qubit_colors[i],\
                           opacity=0.7,\
                           emissive=True,\
                           make_trail=False) for i in range(n_qubits)]
    for varrow in varrows:
        varrow.visible = False
        del varrow
    varrows = [vpython.arrow(color=qubit_colors[i]) for i in range(n_qubits)]
    for vfiber in vfibers:
        vfiber.visible = False
        del vfiber
    vfibers = [vpython.curve(pos=[vpython.vector(0,0,0) for i in range(n_points)],\
                             color=qubit_colors[i]) for i in range(n_qubits)]
    
active_qubit = 0
done = False
def keyboard(event):
    global qubits
    global active_qubit
    global touched
    global energy
    global unitary
    global state
    global n_qubits
    global done
    key = event.key
    if key == "p":
        energy = qutip.rand_herm(2**n_qubits)
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*im*energy.full()*dt))
    elif key == "o":
        state = qutip.rand_ket(2**n_qubits)
    elif key == "[":
        touched = 1
    elif key == "]":
        if n_qubits > 1:
            touched = -1
    elif key == "/":
        done = True
    
vpython.scene.bind('keydown', keyboard)

circle = np.linspace(0, 2*math.pi, num=n_points)

touched = False

while not done:
    vpython.rate(50)
    if touched:
        n_qubits += touched
        recreate()
        touched = False
    state = unitary*state
    
    vspin.axis = vpython.vector(*spin_axis(2**n_qubits, state))
    
    new_stars = q_SurfaceXYZ(state)
    for i in range((2**n_qubits)-1):
        vstars[i].pos = vpython.vector(*new_stars[i])
        
    old_dims = state.dims[:]
    state.dims = [[2]*n_qubits, [1]*n_qubits]
    for i in range(n_qubits):
        qubit = state.ptrace(i)
        r4 = QubitDM_to_R4(qubit)        
        #first_latitude = qubits[i]["first_latitude"]
        #second_latitude = qubits[i]["second_latitude"]
        #longitude = qubits[i]["longitude"]
        #r4 = ThreeAngles_to_R4(first_latitude, second_latitude, longitude)
        base_x, base_y, base_z = C2_to_R3(R4_to_C2(r4)).T[0]
        base_x2 = qutip.expect(qutip.sigmax(), qubit)
        base_y2 = qutip.expect(qutip.sigmay(), qubit)
        base_z2 = qutip.expect(qutip.sigmaz(), qubit)
        vbases[i].pos = vpython.vector(base_x.real, base_y.real, base_z.real)
        varrows[i].pos = vpython.vector(0,0,0)
        varrows[i].axis = vpython.vector(base_x2.real, base_y2.real, base_z2.real)
        varrows[i].shaftwidth = 0.06
        
        hopf_points = [R4_to_hopfCircle(r4, angle) for angle in circle]
        for th in range(n_points):
            proj = R4_to_R3(hopf_points[th])
            if not isinstance(proj, tuple):
                x, y, z = proj.T[0]
                vfibers[i].modify(th, pos=vpython.vector(x.real, y.real, z.real))
            else:
                x = -1*base_x + 2*(th/n_points)
                y = -1*base_y + 2*(th/n_points)
                z = -1*base_y + 2*(th/n_points)               
                vfibers[i].modify(th, pos=vpython.vector(x.real, y.real, z.real))
    state.dims = old_dims
    #vpython.scene.camera.rotate((2*math.pi)/15, origin=vpython.vector(0,0,0))
    vpython.scene.forward = 2*vspin.axis