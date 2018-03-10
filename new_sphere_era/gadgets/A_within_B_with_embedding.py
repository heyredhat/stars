import qutip
import vpython
import math
import scipy
import numpy as np

vpython.scene.width = 1000
vpython.scene.height = 600

def state_xyz(state):
    x = qutip.expect(qutip.sigmax(), state)
    y = qutip.expect(qutip.sigmay(), state)
    z = qutip.expect(qutip.sigmaz(), state)
    return [x, y, z]

def sigmoid(x):  
    return 2*(math.exp(-np.logaddexp(0, -x))-0.5)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
    
class Sphere:
    def __init__(self, state, center, color, radius=1.0):
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
                                          self.radius*vpython.vector(*state_xyz(self.eigenvectors[i])),\
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
        xyz = state_xyz(other_according_to_self)
        center = [self.center[i]+xyz[i] for i in range(3)]
        self.local_other = Sphere(other_according_to_self,\
                                  center,\
                                  self.other.color,\
                                  self.other.radius*0.3)
    
    def update(self):
        self.vsphere.pos = vpython.vector(*self.center)
        self.eigenvalues, self.eigenvectors = self.state.eigenstates()
        for i in range(len(self.eigenvectors)):
            self.vstars[i].pos = vpython.vector(*self.center)+\
                                 self.radius*vpython.vector(*state_xyz(self.eigenvectors[i]))
            self.vstars[i].radius = 0.1+sigmoid(self.eigenvalues[i])/2.
            self.vstars[i].color = vpython.color.hsv_to_rgb(vpython.vector(sigmoid(self.eigenvalues[i]),1,1))
            self.varrows[i].modify(0, pos=self.vsphere.pos)
            self.varrows[i].modify(1, pos=self.vstars[i].pos)
        if self.other != None:
            other_according_to_self = self.other.state.transform(self.state)
            self.local_other.state = other_according_to_self
            xyz = state_xyz(other_according_to_self)
            self.local_other.center = [self.vstars[0].pos.x+2*xyz[0], self.vstars[0].pos.y+2*xyz[1], self.vstars[0].pos.z+2*xyz[2]]
            self.local_other.update()
        
    def apply(self, operator, inverse=False, dt=0.01):
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*operator.full()*dt))
        if inverse:
            unitary = unitary.dag()
        self.state = unitary*self.state*unitary.dag()

print("usage:")
print("a/d s/w z/x for sigma X Z Y on A")
print("j/l k/i m/, for sigma X Z Y on B")

A = Sphere(qutip.rand_ket(2).ptrace(0), [-1.5, 0, 0], vpython.color.red)
B = Sphere(qutip.rand_ket(2).ptrace(0), [1.5, 0, 0], vpython.color.blue)

A.set_other(B)
B.set_other(A)

def keyboard(event):
    global A
    global B
    key = event.key
    operator = None
    if key == "a":   #-x for A
        A.apply(qutip.sigmax(), True)
    elif key == "d": #+x for A
        A.apply(qutip.sigmax(), False)
    elif key == "s": #-z for A
        A.apply(qutip.sigmaz(), True)
    elif key == "w": #+z for A
        A.apply(qutip.sigmaz(), False)
    elif key == "z": #-y for A
        A.apply(qutip.sigmay(), True)
    elif key == "x": #+y for A
        A.apply(qutip.sigmay(), False)
    elif key == "j": #-x for B
        B.apply(qutip.sigmax(), True)
    elif key == "l": #+x for B
        B.apply(qutip.sigmax(), False)
    elif key == "k": #-z for B
        B.apply(qutip.sigmaz(), True)
    elif key == "i": #+z for B
        B.apply(qutip.sigmaz(), False)
    elif key == "m": #-y for B
        B.apply(qutip.sigmay(), True)
    elif key == ",": #+y for B
        B.apply(qutip.sigmay(), False)

vpython.scene.bind('keydown', keyboard)

while True:
    vpython.rate(50)
    A.update()
    B.update()