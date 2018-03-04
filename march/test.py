import qutip
import numpy as np
import random

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def C2_to_R4(c2):
    return np.array([[qutip.expect(qutip.sigmax(), c2)],\
                     [qutip.expect(qutip.sigmay(), c2)],\
                     [qutip.expect(qutip.sigmaz(), c2)],\
                     [qutip.expect(qutip.identity(2), c2)]])

def R4_to_C2(r4):
    x, y, z, w = r4.T[0]
    return (1./2)*(x*qutip.sigmax() + y*qutip.sigmay() + z*qutip.sigmaz() + w*qutip.identity(2))

c2 = qutip.rand_herm(2)
r4 = C2_to_R4(c2)
c22 = R4_to_C2(r4)
print("c2:\n%s" % c2)
print("r4:\n%s" % r4)
print("c22:\n%s" % c22)

print()

R4 = normalize(np.array([[random.uniform(-1,1)],\
			  		     [random.uniform(-1,1)],\
			  			 [random.uniform(-1,1)],\
			   			 [random.uniform(-1,1)]]))
C2 = R4_to_C2(R4)
R42 = C2_to_R4(C2)
print("R4:\n%s" % R4)
print("C2:\n%s" % C2)
print("R42:\n%s" % R42)