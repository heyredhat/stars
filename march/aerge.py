import math
import cmath
import functools
import operator
import numpy as np
import sympy
import mpmath
import scipy
import qutip 
import random

def im():
    return complex(0, 1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

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

def evolve(qubit, photon, unitary, gauge_charge):
    old_state = qubit
    new_state = unitary*qubit*unitary.dag()

    old_vector = qubit_to_vector(old_state)
    new_vector = qubit_to_vector(new_state)

    old_angles, r0 = vector_to_angles(old_vector)
    new_angles, r1 = vector_to_angles(new_vector)
    angle_delta = new_angles[0]-old_angles[0]

    vector_delta = new_vector-old_vector
    vector_delta = vector_delta.T[0].tolist()

    gauge_delta = []
    for i in range(4):
        if vector_delta[i] != 0:
            delta.append([angle_delta/vector_delta[i]])
        else:
            delta.append([0])
    gauge_delta = np.array(delta)

    return (new_state*cmath.exp(-im()*angle_delta),\
            vector_to_qubit(normalize(qubit_to_vector(photon) + (1./gauge_charge)*gauge_delta)))

qubit = qutip.rand_ket(2).ptrace(0)
photon = qutip.rand_ket(2).ptrace(0)

unitary = (-2*math.pi*im()*qutip.rand_herm(2)*0.001).expm()

qubit1, qubit2 = evolve(qubit, photon, unitary, 2)

