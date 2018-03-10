import math
import cmath
import vpython
import numpy as np
import random
import sys

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
    return np.array([[x],\
                     [y],\
                     [z]])

##################################################################################################################

n_qubits = 1

if len(sys.argv) == 2:
  if sys.argv[1].isdigit():
    n_qubits = int(sys.argv[1])

dt = 0.07
n_points = 50

qubits = [{"first_latitude": random.uniform(0, math.pi),\
           "second_latitude": random.uniform(0, math.pi),\
           "longitude": random.uniform(0, 2*math.pi)} for i in range(n_qubits)]
qubit_colors = [vpython.vector(random.random(), random.random(), random.random()) for i in range(n_qubits)]

vpython.scene.width = 1000
vpython.scene.height = 800

vsphere = vpython.sphere(pos=vpython.vector(0,0,0),\
                         radius=1.0,\
                         color=vpython.color.blue,\
                         opacity=0.6)
vbases = [vpython.sphere(radius=0.1,\
                       color=qubit_colors[i],\
                       opacity=0.7,\
                       emissive=True) for i in range(n_qubits)]
vfibers = [vpython.curve(pos=[vpython.vector(0,0,0) for i in range(n_points)],\
                         color=qubit_colors[i]) for i in range(n_qubits)]

print("usage: python three_angles_with_fibers.py *n-qubits*")
print("a vs. d : +/- first_latitude")
print("s vs. w : +/- longitude")
print("z vs. x : +/- second_latitude")
print("qubit # to switch to qubit; 0 to start")

active_qubit = 0
def keyboard(event):
    global qubits
    global active_qubit
    global touched
    key = event.key
    if key.isdigit():
        i = int(key)
        if i < n_qubits:
            active_qubit = i
            print("qubit #%d active!" % active_qubit)
    elif key == "a":
        qubits[active_qubit]["first_latitude"] -= dt
    elif key == "d":
        qubits[active_qubit]["first_latitude"] += dt
    elif key == "s":
        qubits[active_qubit]["longitude"] -= dt
    elif key == "w":
        qubits[active_qubit]["longitude"] += dt
    elif key == "z":
        qubits[active_qubit]["second_latitude"] -= dt
    elif key == "x":
        qubits[active_qubit]["second_latitude"] += dt
    while qubits[active_qubit]["first_latitude"] < 0:
        qubits[active_qubit]["first_latitude"] += math.pi
    while qubits[active_qubit]["first_latitude"] > math.pi:
        qubits[active_qubit]["first_latitude"] -= math.pi
    while qubits[active_qubit]["second_latitude"] < 0:
        qubits[active_qubit]["second_latitude"] += math.pi
    while qubits[active_qubit]["second_latitude"] > math.pi:
        qubits[active_qubit]["second_latitude"] -= math.pi
    while qubits[active_qubit]["longitude"] < 0:
        qubits[active_qubit]["longitude"] += 2*math.pi
    while qubits[active_qubit]["longitude"] >= 2*math.pi:
        qubits[active_qubit]["longitude"] -= 2*math.pi
    touched = True
vpython.scene.bind('keydown', keyboard)

touched = True
circle = np.linspace(0, 2*math.pi, num=n_points)

while True:
    vpython.rate(100)
    if touched:
        for i in range(n_qubits):
            first_latitude = qubits[i]["first_latitude"]
            second_latitude = qubits[i]["second_latitude"]
            longitude = qubits[i]["longitude"]
            
            r4 = ThreeAngles_to_R4(first_latitude, second_latitude, longitude)
            base_x, base_y, base_z = C2_to_R3(R4_to_C2(r4)).T[0]
            vbases[i].pos = vpython.vector(base_x.real, base_y.real, base_z.real)
            
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
        touched = False