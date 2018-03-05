import math
import cmath
import functools
import operator
import numpy as np
import sympy
import mpmath
import scipy
import qutip 
#import vpython
import random

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def qubit_to_vector(qubit):
    return np.array([[qutip.expect(qutip.sigmax(), qubit)],\
                     [qutip.expect(qutip.sigmay(), qubit)],\
                     [qutip.expect(qutip.sigmaz(), qubit)],\
                     [qutip.expect(qutip.identity(2), qubit)]])

def vector_to_qubit(vector):
    x, y, z, t = vector.T[0]
    return (1./2)*(x*qutip.sigmax() + y*qutip.sigmay() + z*qutip.sigmaz() + t*qutip.identity(2))

if False:
    qubit = qutip.rand_herm(2)
    vector = qubit_to_vector(qubit)
    qubit2 = vector_to_qubit(vector)
    print("qubit:\n%s" % qubit)
    print("vector:\n%s" % vector)
    print("qubit2:\n%s" % qubit2)

if False:
    print()

    vector = normalize(np.array([[random.uniform(-1,1)],\
                             [random.uniform(-1,1)],\
                             [random.uniform(-1,1)],\
                             [random.uniform(-1,1)]]))
    qubit = vector_to_qubit(vector)
    vector2 = qubit_to_vector(qubit)
    print("vector:\n%s" % vector)
    print("qubit:\n%s" % qubit)
    print("vector2:\n%s" % vector2)

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
def inverse_stereographic_projection(xyz):
    if isinstance(xyz, int):
        n = xyz
        return np.array([[1]]+[[0]]*(n-1))
    coordinates = xyz.T[0][::-1]
    s = sum([coordinate**2 for coordinate in coordinates])
    sphere = [[(s - 1)/(s + 1)]]
    for coordinate in coordinates:
        sphere.append([(2*coordinate)/(s + 1)])
    return np.array(sphere[::-1])

if False:
    print()

    n = 3
    angles = [random.uniform(0, math.pi) for i in range(n-1)]+[random.uniform(0, 2*math.pi)]
    xyz = angles_to_vector(angles)
    angles2 = vector_to_angles(xyz)

    print("angles:\n%s" % angles)
    print("xyz:\n%s" % xyz)
    print("angles2:\n%s" % str(angles2))

if True:
    print()

    n = 3
    angles = [random.uniform(0, math.pi) for i in range(n-1)]+[random.uniform(0, 2*math.pi)]
    xyz = angles_to_vector(angles)
    plane = stereographic_projection(xyz)
    xyz2 = inverse_stereographic_projection(plane)

    print("xyz:\n%s" % xyz)
    print("plane:\n%s" % plane)
    print("xyz2:\n%s" % xyz2)
