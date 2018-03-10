import qutip
import vpython
import math
import scipy
import numpy as np

vpython.scene.width = 1000
vpython.scene.height = 600

def state_xyz(state, n):
    X, Y, Z = qutip.jmat((n-1.)/2.)
    x = qutip.expect((2/(n-1.))*X, state)
    y = qutip.expect((2/(n-1.))*Y, state)
    z = qutip.expect((2/(n-1.))*Z, state)
    return [x, y, z]

def sigmoid(x):  
    return 2*(math.exp(-np.logaddexp(0, -x))-0.5)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

class Sphere:
    def __init__(self, n, state, center, color, radius=1.0):
        self.n = n
        self.state = state
        self.center = center
        self.color = color
        self.radius = radius 
        
        self.vsphere = vpython.sphere(pos=vpython.vector(*self.center),\
                                      radius=self.radius,\
                                      color=self.color,\
                                      opacity=0.4)
        
        self.eigenvalues, self.eigenvectors = self.state.eigenstates()
        self.vstars = [vpython.sphere(pos=self.vsphere.pos+\
                                          self.radius*vpython.vector(*state_xyz(self.eigenvectors[i], self.n)),\
                                      radius=self.radius*0.1,\
                                      color=vpython.color.hsv_to_rgb(vpython.vector(sigmoid(self.eigenvalues[i]),1,1)),\
                                      opacity=0.3)\
                                          for i in range(len(self.eigenvectors))]
        self.varrows = [vpython.curve(pos=[self.vsphere.pos,\
                                           self.vstars[i].pos],\
                                      color=self.color)\
                                           for i in range(len(self.eigenvectors))]
        self.other = None
        self.local_other = None
        
    def set_other(self, other):
        self.other = other
        other_according_to_self = self.other.state.transform(self.state)
        xyz = state_xyz(other_according_to_self, other.n)
        center = [self.center[i]+xyz[i] for i in range(3)]
        self.local_other = Sphere(other.n, other_according_to_self,\
                                  center,\
                                  self.other.color,\
                                  self.other.radius*0.3)
    
    def update(self):
        self.vsphere.pos = vpython.vector(*self.center)
        self.eigenvalues, self.eigenvectors = self.state.eigenstates()
        highest = max(self.eigenvalues)
        II = self.eigenvalues.tolist().index(highest)
        for i in range(len(self.eigenvectors)):
            self.vstars[i].pos = vpython.vector(*self.center)+\
                                 self.radius*vpython.vector(*state_xyz(self.eigenvectors[i], self.n))
            self.vstars[i].radius = 0.1+sigmoid(self.eigenvalues[i])/3.
            self.vstars[i].color = vpython.color.hsv_to_rgb(vpython.vector(sigmoid(self.eigenvalues[i]),1,1))
            self.varrows[i].modify(0, pos=self.vsphere.pos)
            self.varrows[i].modify(1, pos=self.vstars[i].pos)
        if self.other != None:
            other_according_to_self = self.other.state.transform(self.state)
            self.local_other.state = other_according_to_self
            xyz = state_xyz(other_according_to_self, self.other.n)
            self.local_other.center = [self.vstars[II].pos.x-1*xyz[0], self.vstars[II].pos.y-1*xyz[1], self.vstars[II].pos.z-1*xyz[2]]
            #self.local_other.center = [self.vsphere.pos.x+1*xyz[0], self.vsphere.pos.y+1*xyz[1], self.vsphere.pos.z+1*xyz[2]]
            self.local_other.update()
        
    def apply(self, operator, inverse=False, dt=0.01):
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*operator.full()*dt))
        if inverse:
            unitary = unitary.dag()
        self.state = unitary*self.state*unitary.dag()

n = 2
A = Sphere(n, qutip.rand_ket(n).ptrace(0), [-1.5, 0, 0], vpython.color.red)
B = Sphere(n, qutip.rand_ket(n).ptrace(0), [1.5, 0, 0], vpython.color.blue)

A.set_other(B)
B.set_other(A)

energy = qutip.rand_herm(n)
energy_unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*energy.full()*0.01))

def keyboard(event):
    global A
    global B
    global n
    key = event.key
    operator = None
    X, Y, Z = qutip.jmat((n-1.)/2.)
    if key == "a":   #-x for A
        A.apply(X, True)
    elif key == "d": #+x for A
        A.apply(X, False)
    elif key == "s": #-z for A
        A.apply(Z, True)
    elif key == "w": #+z for A
        A.apply(Z, False)
    elif key == "z": #-y for A
        A.apply(Y, True)
    elif key == "x": #+y for A
        A.apply(Y, False)
    elif key == "j": #-x for B
        B.apply(X, True)
    elif key == "l": #+x for B
        B.apply(X, False)
    elif key == "k": #-z for B
        B.apply(Z, True)
    elif key == "i": #+z for B
        B.apply(Z, False)
    elif key == "m": #-y for B
        B.apply(Y, True)
    elif key == ",": #+y for B
        B.apply(Y, False)

vpython.scene.bind('keydown', keyboard)

while True:
    vpython.rate(50)
    #A.state = energy_unitary*A.state*energy_unitary.dag()
    A.update()
    B.update()