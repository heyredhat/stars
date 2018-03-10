import math
import qutip
import cmath
import random
import numpy as np
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def xyz_to_hermitian(xyz):
	x, y, z = xyz[0], xyz[1], xyz[2]
	return 0.5*(qutip.identity(2)+ x*qutip.sigmax() + y*qutip.sigmaz() + z*qutip.sigmaz())

### 

def xyz_to_angle(xyz):
	x, y, z = xyz
	r = math.sqrt(x**2 + y**2 + z**2)
	phi = math.atan2(y,x)
	theta = None
	if r != 0:
		theta = math.acos(z/r)
	else:
		theta = math.pi/(2*np.sign(z))
	while phi < 0:
		phi += 2*math.pi
	return (theta, phi)

def angle_to_xyz(angle):
    theta, phi = angle
    return [math.sin(theta)*math.cos(phi),\
            math.sin(theta)*math.sin(phi),\
            math.cos(theta)]

###

if False:
	print("angle -> xyz test")
	angle = (random.uniform(0, math.pi), random.uniform(0, 2*math.pi))
	xyz = angle_to_xyz(angle)
	angle2 = xyz_to_angle(xyz)
	print("angle:\n%s" % str(angle))
	print("xyz:\n%s" % str(xyz))
	print("angle2:\n%s" % str(angle2))

def angle_to_qubit(angle):
    theta, phi = angle
    qubit = [math.cos(theta/2.), math.sin(theta/2.)*cmath.exp(complex(0,1)*phi)]
    return qutip.Qobj(np.array(qubit))

def qubit_to_angle(qubit):
	dm = qubit.ptrace(0).full()
	x = float(2*dm[0][1].real)
	y = float(2*dm[1][0].imag)
	z = float((dm[0][0] - dm[1][1]).real)
	return xyz_to_angle([x, y, z])

if True:
	print("\nqubit -> angle test")
	qubit = qutip.rand_ket(2)
	angle = qubit_to_angle(qubit)
	qubit2 = angle_to_qubit(angle)
	print("qubit:\n%s" % str(qubit))
	print("angle:\n%s" % str(angle))
	print("qubit2:\n%s" % str(qubit2))

	print(angle_to_xyz(qubit_to_angle(qubit)))
	print(angle_to_xyz(qubit_to_angle(qubit2)))

if False:
	print("\nangle -> qubit test")
	angle = (random.uniform(0, math.pi), random.uniform(0, 2*math.pi))
	qubit = angle_to_qubit(angle)
	angle2 = qubit_to_angle(qubit)
	print("angle:\n%s" % str(angle))
	print("qubit:\n%s" % str(qubit))
	print("angle2:\n%s" % str(angle2))

