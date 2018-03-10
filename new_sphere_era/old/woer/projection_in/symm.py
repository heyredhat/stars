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
    return complex(0, 1)

vpython.scene.height = 600
vpython.scene.width = 800

alice = qutip.rand_ket(2)
bob = qutip.rand_ket(2)

alice_center = vpython.vector(-1,0,0)
valice = vpython.sphere(pos=alice_center,\
                         radius=1.0,\
                         color=vpython.color.blue,\
                         opacity=0.3)
valice_star = vpython.sphere(radius=0.1,\
							 color=vpython.color.white,\
							 opacity=0.7, emissive=True)
valice_arrow = vpython.arrow(color=vpython.color.blue,\
							 shaftwidth=0.06)

bob_center = vpython.vector(1,0,0)
vbob = vpython.sphere(pos=bob_center,\
                         radius=1.0,\
                         color=vpython.color.red,\
                         opacity=0.3)
vbob_star = vpython.sphere(radius=0.1,\
							 color=vpython.color.white,\
							 opacity=0.7, emissive=True)
vbob_arrow = vpython.arrow(color=vpython.color.blue,\
							 shaftwidth=0.06)

def apply(operator, state, inverse=False):
	if inverse:
		return (-2*math.pi*im()*0.005*operator).dag()*state
	return (-2*math.pi*im()*0.005*operator)*state

selected = "Bob"
touched = False
def keyboard(event):
    global selected
    global alice
    global bob
    global touched
    key = event.key
    if key == "a":
    	if selected == "Bob":
    		bob = apply(qutip.sigmax(), bob, inverse=True)
    	else:
    		alice = apply(qutip.sigmax(), alice, inverse=True)
    elif key == "d":
    	if selected == "Bob":
    		bob = apply(qutip.sigmax(), bob, inverse=False)
    	else:
    		alice = apply(qutip.sigmax(), alice, inverse=False)
    elif key == "s":
    	if selected == "Bob":
    		bob = apply(qutip.sigmaz(), bob, inverse=True)
    	else:
    		alice = apply(qutip.sigmaz(), alice, inverse=True)
    elif key == "w":
    	if selected == "Bob":
    		bob = apply(qutip.sigmaz(), bob, inverse=False)
    	else:
    		alice = apply(qutip.sigmaz(), alice, inverse=False)
    elif key == "z":
    	if selected == "Bob":
    		bob = apply(qutip.sigmay(), bob, inverse=True)
    	else:
    		alice = apply(qutip.sigmay(), alice, inverse=True)
    elif key == "x":
    	if selected == "Bob":
    		bob = apply(qutip.sigmay(), bob, inverse=False)
    	else:
    		alice = apply(qutip.sigmay(), alice, inverse=False)
    elif key == "p":
        if selected == "Bob":
        	selected = "Alice"
        else:
        	selected = "Bob"
    touched = True
vpython.scene.bind('keydown', keyboard)

def expect(state):
	return [qutip.expect(qutip.sigmax(), state),\
		    qutip.expect(qutip.sigmay(), state),\
		    qutip.expect(qutip.sigmaz(), state)]

while True:
	vpython.rate(100)
	if touched:
		alice_xyz = expect(alice)
		valice_star.pos = vpython.vector(*alice_xyz)+alice_center
		valice_arrow.pos = alice_center
		valice_arrow.axis = vpython.vector(*alice_xyz)

		bob_xyz = expect(bob)
		vbob_star.pos = vpython.vector(*bob_xyz)+bob_center
		vbob_arrow.pos = bob_center
		vbob_arrow.axis = vpython.vector(*bob_xyz)
		touched = False

