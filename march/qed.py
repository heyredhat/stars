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
    return complex(0,1)

def C2_to_R4(c2):
    return np.array([[qutip.expect(qutip.sigmax(), c2)],\
                     [qutip.expect(qutip.sigmay(), c2)],\
                     [qutip.expect(qutip.sigmaz(), c2)],\
                     [qutip.expect(qutip.identity(2), c2)]])

def R4_to_C2(r4):
    x, y, z, w = r4.T[0]
    return (1./2)*(x*qutip.sigmax() + y*qutip.sigmay() + z*qutip.sigmaz() + w*qutip.identity(2))

def R4_to_R3(r4):
    x, y, z, w = r4.T[0]
    xyz = np.array([[x],\
                    [y],\
                    [z]])
    if w == 1:
        return xyz
    else:
        return (1./(1.-w))*xyz

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

dt = 0.01
qubit = qutip.rand_ket(2).ptrace(0)
photon = np.array([[0],[0],[0],[0]], dtype='complex128')

vsphere = vpython.sphere(pos=vpython.vector(0,0,0),\
                         radius=1,\
                         opacity=0.4,\
                         color=vpython.color.blue)
vbase = vpython.sphere(radius=0.1,\
                       opacity=0.8,\
                       color=vpython.color.white, emissive=True)
vphoton = vpython.sphere(radius=0.2,\
                         color=vpython.color.yellow,\
                         opacity=0.2)


def apply(transformation):
    global qubit
    qubit = transformation*qubit*transformation.dag()

def go(transformation):
    global qubit
    global photon

    qubit_matrix = qubit.full()
    first_phase = None
    if qubit_matrix[0,1] != 0:
        first_phase = np.angle(qubit_matrix[0,1]) 
    else:
        first_phase = np.angle(qubit_matrix[1,0]) 
    qubit = qubit/cmath.exp(im()*first_phase)

    transformed_qubit = transformation*qubit*transformation.dag()

    transformed_qubit_matrix = transformed_qubit.full()
    second_phase = None
    if transformed_qubit_matrix[0,1] != 0:
        second_phase = np.angle(transformed_qubit_matrix[0,1]) 
    else:
        second_phase = np.angle(transformed_qubit_matrix[1,0]) 
    transformed_qubit = transformed_qubit/cmath.exp(im()*second_phase)

    relative_phase = second_phase - first_phase

    x0, y0, z0, w0 = C2_to_R4(qubit).T[0]
    x1, y1, z1, w1 = C2_to_R4(transformed_qubit).T[0]

    dx = None
    if x1 - x0 == 0:
        dx = 0
    else:
        dx = relative_phase/(x1-x0)
    if y1 - y0 == 0:
        dy = 0
    else:
        dy = relative_phase/(y1-y0)
    if z1 - z0 == 0:
        dz = 0
    else:
        dz = relative_phase/(z1-z0)
    if w1 - w0 == 0:
        dw = 0
    else:
        dw = relative_phase/(w1-w0)
    photon += np.array([[dx],\
                              [dy],\
                              [dz],\
                              [dw]])
    photon = normalize(photon)
    qubit = transformed_qubit

def keyboard(event):
    global qubit
    unitary_x = (2*math.pi*im()*qutip.sigmax()*dt).expm()
    unitary_y = (2*math.pi*im()*qutip.sigmay()*dt).expm()
    unitary_z = (2*math.pi*im()*qutip.sigmaz()*dt).expm()
    key = event.key
    if key == "a":
        apply(unitary_x.dag())
    elif key == "d":
        apply(unitary_x)
    elif key == "s":
        apply(unitary_z.dag())
    elif key == "w":
        apply(unitary_z)
    elif key == "z":
        apply(unitary_y.dag())
    elif key == "x":
        apply(unitary_y)
vpython.scene.bind('keydown', keyboard)

vpython.scene.autoscale=False
while True:
    vpython.rate(100)
    base_state = R4_to_R3(C2_to_R4(qubit)).T[0]
    #base_x = qutip.expect(qutip.sigmax(), qubit)
    #base_y = qutip.expect(qutip.sigmay(), qubit)
    #base_z = qutip.expect(qutip.sigmaz(), qubit)
    base_x = base_state[0]
    base_y = base_state[1]
    base_z = base_state[2]
    vbase.pos = vpython.vector(base_x.real, base_y.real, base_z.real)
    photon_state = R4_to_R3(photon).T[0]
    #photon_x = photon_state[0]
    #photon_y = photon_state[1]
    #photon_z = photon_state[2]
    photon_x = qutip.expect(qutip.sigmax(), qubit)
    photon_y = qutip.expect(qutip.sigmay(), qubit)
    photon_z = qutip.expect(qutip.sigmaz(), qubit)
    vphoton.pos = vpython.vector(photon_x.real, photon_y.real, photon_z.real)
