import random
import numpy as np
import functools
import operator
import math

# n-sphere to n+1 dimensions
# note: last angle ranges to 2pi, all previous to pi
def spherical_to_cartesian(angles, r=1):
	n = len(angles)
	coordinates = []
	for i in range(n+1):
		coordinate = r
		coordinate *= functools.reduce(operator.mul,\
						[math.sin(angles[j]) for j in range(i)], 1)
		if i != n:
			coordinate *= math.cos(angles[i])
		coordinates.append([coordinate])
	return np.array(coordinates)

# n dimensions to n-1-sphere
def cartesian_to_spherical(xyzs):
	coordinates = xyzs.T[0].tolist()
	n = len(coordinates)-1
	r = math.sqrt(sum([x**2 for x in xyzs]))
	angles = []
	for i in range(n):
		if i != n-1:
			angle = coordinates[i]
			divisor = math.sqrt(sum([coordinates[j]**2 for j in range(i,n+1)]))
			angles.append(math.acos(angle/divisor))
		else:
			angle = coordinates[-2]/math.sqrt(coordinates[-1]**2 + coordinates[-2]**2)
			angle = math.acos(angle)
			if coordinates[-1] < 0:
				angle = 2*math.pi - angle
			angles.append(angle)
	return angles, r

# n-sphere in n+1 dimensions to hyperplane in n dimensions plus infinity
# projection from [1, 0, 0, 0, ...]
def stereographic_projection(xyz):
	coordinates = xyz.T[0]
	if coordinates[0] == 1:
		return len(coordinates)
	else:
		return np.array([[coordinate/(1-coordinates[0])] for coordinate in coordinates[1:]])

# from hyperplane in n dimensions plus infinity to n-sphere in n+1 dimensions
def inverse_stereographic_projection(xyz):
	if isinstance(xyz, int):
		n = xyz
		return np.array([[1]]+[[0]]*(n-1))
	coordinates = xyz.T[0]
	s = sum([coordinate**2 for coordinate in coordinates])
	sphere = [[(s - 1)/(s + 1)]]
	for coordinate in coordinates:
		sphere.append([(2*coordinate)/(s + 1)])
	return np.array(sphere)

debug = False

if debug:
	n = 10
	angles = [random.uniform(0, math.pi) for i in range(n-1)]+[random.uniform(0, 2*math.pi)]
	xyz = spherical_to_cartesian(angles)
	angles2 = cartesian_to_spherical(xyz)

if debug:
	n = 3
	angles = [random.uniform(0, math.pi) for i in range(n-1)]+[random.uniform(0, 2*math.pi)]
	xyz = spherical_to_cartesian(angles)
	plane = stereographic_projection(xyz)
	xyz2 = inverse_stereographic_projection(plane)

