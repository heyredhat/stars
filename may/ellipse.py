import vpython as vp
import qutip as qt
import numpy as np
import scipy as sp
import cmath
import math
import os

class EllipseSphere:
	def __init__(self):
		self.state = qt.rand_ket(2)
		self.energy = qt.rand_herm(2)
		self.dt = 0.01

		self.vrod = vp.arrow(color=vp.color.red,\
							 pos=vp.vector(0,0,0),\
							 axis=vp.vector(0,0,1))
		self.vsphere = vp.sphere(color=vp.color.blue,\
								 radius=1.0,\
								 opacity=0.4)
		self.vstar = vp.sphere(pos=vp.vector(*self.spin_axis()),\
							   color=vp.color.white,\
							   radius=0.1,\
							   emissive=True) 
		self.ellipse = self.state_to_ellipse()
		self.vellipse = vp.curve(pos=self.ellipse_points(*self.ellipse))
		vp.scene.bind('keydown', self.keyboard)

		self.lit = -1
		self.evolving = True
		self.touched = False
		self.done = False

	def state_to_ellipse(self):
		psi_x, psi_y = self.state.full().T[0]
		psi_xR, psi_xTH = cmath.polar(psi_x)
		psi_yR, psi_yTH = cmath.polar(psi_y)
		beta = psi_yTH - psi_xTH
		theta = math.acos(psi_xR)
		a = math.sqrt( (1 + math.sqrt(1-(math.sin(2*theta)**2)*(math.sin(beta)**2)))/2. )
		b = math.sqrt( (1 - math.sqrt(1-(math.sin(2*theta)**2)*(math.sin(beta)**2)))/2. )
		angle = math.atan(math.tan(2*math.acos(psi_xR))*math.cos(beta))/2.
		x, y, z = self.spin_axis()
		if np.sign(z) == -1:
			angle = angle-math.pi/2
		return (a, b, angle)

	def ellipse_points(self, a, b, angle, pts=100):
		points = np.linspace(0,2*math.pi,pts)
		X = a*np.cos(points)*math.cos(angle) - math.sin(angle)*b*np.sin(points)
		Y = a*np.cos(points)*math.sin(angle) + math.cos(angle)*b*np.sin(points)
		points = [vp.vector(X[i], Y[i], 0) for i in range(pts)]
		return points

	def spin_axis(self):
		return [qt.expect(qt.sigmax(), self.state),\
				qt.expect(qt.sigmay(), self.state),\
				qt.expect(qt.sigmaz(), self.state)]

	def evolve(self, operator, inverse=False, dt=0.01):
		unitary = (-2*math.pi*complex(0,1)*operator*dt).expm()
		if inverse:
			unitary = unitary.dag()
		self.state = unitary*self.state
		self.touched = True

	def update_visuals(self):
		if self.evolving:
			self.evolve(self.energy, dt=self.dt)
		if self.touched:
			self.vstar.pos = vp.vector(*self.spin_axis())
			self.ellipse = self.state_to_ellipse()
			epts = self.ellipse_points(*self.ellipse)
			for i in range(100):
				self.vellipse.modify(i, epts[i])
			self.touched = False
		self.lit += 1
		if self.lit == 0:
			self.vellipse.modify(99, color=vp.color.white,radius=0)
		else:
			self.vellipse.modify(self.lit-1, color=vp.color.white,radius=0)
		if self.lit == 99:
			self.lit = 0
		self.vellipse.modify(self.lit, color=vp.color.yellow, radius=0.1)

	def keyboard(self, event):
		key = event.key
		if key == "a":
			self.evolve(qt.sigmax(), inverse=True)
		elif key == "d":
			self.evolve(qt.sigmax(), inverse=False)
		elif key == "s":
			self.evolve(qt.sigmay(), inverse=True)
		elif key == "w":
			self.evolve(qt.sigmay(), inverse=False)
		elif key == "z":
			self.evolve(qt.sigmaz(), inverse=True)
		elif key == "x":
			self.evolve(qt.sigmaz(), inverse=False)
		elif key == "1":
			l, v = qt.sigmax().eigenstates()
			self.state = v[0]
			self.touched = True
		elif key == "2":
			l, v = qt.sigmax().eigenstates()
			self.state = v[1]
			self.touched = True
		elif key == "3":
			l, v = qt.sigmay().eigenstates()
			self.state = v[0]
			self.touched = True
		elif key == "4":
			l, v = qt.sigmay().eigenstates()
			self.state = v[1]
			self.touched = True
		elif key == "5":
			l, v = qt.sigmaz().eigenstates()
			self.state = v[0]
			self.touched = True
		elif key == "6":
			l, v = qt.sigmaz().eigenstates()
			self.state = v[1]
			self.touched = True
		elif key == "q":
			self.done = True
		elif key == "e":
			if self.evolving:
				self.evolving = False
			else:
				self.evolving = True
		elif key == "p":
			self.energy = qt.rand_herm(2)
		elif key == "[":
			self.dt -= 0.01
		elif key == "]":
			self.dt += 0.01

print("the sphere and the ellipse!")
print("controls:")
print("q to quit")
print("a/d is -/+ x rotation")
print("s/w is -/+ y rotation")
print("z/x is -/+ z rotation")
print("1 2 3 4 5 6 collapse to X- X+ Y- Y+ Z- Z+")
print("e to pause evolution")
print("p to generate a new random energy function")
print("[ to slow evolution by 0.01")
print("] to speed it up by 0.01")

vp.scene.width = 600
vp.scene.height = 600
sphere = EllipseSphere()

while sphere.done == False:
	vp.rate(15)
	sphere.update_visuals()

os._exit(0)