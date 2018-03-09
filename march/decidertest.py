##################################################################################################################
#
# SPHERES 1.0 (C) 2018 MATTHEW B. WEISS
#
##################################################################################################################

import os
import sys
import math
import qutip
import cmath
import sympy
import scipy
import mpmath
import pickle
import random
import crayons
import inspect
import operator
import threading
import itertools
import functools
import qutip as qt
import numpy as np
import vpython as vp
import time

##################################################################################################################

def display_loop_prompt():
    return input(crayons.cyan(":")+crayons.blue(">")+" ").lower().split()

def display_inner_prompt():
    return input(crayons.green('.')+crayons.magenta('.')+crayons.blue('.'))

def display_error(message=None):
    if message:
        print(crayons.red("?: %s" % message))
    else:
        print(crayons.red("?"))

##################################################################################################################

def im():
    return complex(0, 1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

##################################################################################################################

def c_xyz(c):
    if c == float('inf'):
        return [0,0,1]
    x = c.real
    y = c.imag
    return [-1*(2*x)/(1.+(x**2)+(y**2)),\
            (2*y)/(1.+(x**2)+(y**2)),\
            (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]

def xyz_c(xyz):
    x, y, z = -1*xyz[0], xyz[1], xyz[2]
    if z == 1:
        return float('inf') 
    else:
        return complex(x/(1-z), y/(1-z))

def polynomial_v(polynomial):
    coordinates = [polynomial[i]/(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) for i in range(len(polynomial))]
    return np.array(coordinates)

def combos(a,b):
    f = math.factorial
    return f(a) / f(b) / f(a-b)

def v_polynomial(v):
    polynomial = v.tolist()
    return [(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) * polynomial[i] for i in range(len(polynomial))] 

def C_polynomial(roots):
    s = sympy.symbols("s")
    polynomial = sympy.Poly(functools.reduce(lambda a, b: a*b, [s-np.conjugate(root) for root in roots]), domain="CC")
    return [complex(c) for c in polynomial.coeffs()]

def polynomial_C(polynomial):
    try:
        roots = [np.conjugate(complex(root)) for root in mpmath.polyroots(polynomial)]
    except:
        return [float('Inf') for i in range(len(polynomial)-1)]
    return roots

def C_v(roots):
    return polynomial_v(C_polynomial(roots))

def v_C(v):
    return polynomial_C(v_polynomial(v))

def v_SurfaceXYZ(v):
    return [c_xyz(c) for c in v_C(v)]

def SurfaceXYZ_v(XYZ):
    return C_v([xyz_c(xyz) for xyz in XYZ])

def q_SurfaceXYZ(q):
    return v_SurfaceXYZ(q.full().T[0])

def SurfaceXYZ_q(XYZ):
    return Qobj(C_v([xyz_c(xyz) for xyz in XYZ]))

##################################################################################################################

class MajoranaSphere:
    def __init__(self, state,\
                       center=None,\
                       radius=1,\
                       sphere_color=None,\
                       star_color=None,\
                       arrow_color=None,\
                       show_sphere=True,\
                       show_stars=True,\
                       show_arrow=True,\
                       show_tag=False,\
                       show_tag_on_stars=False,\
                       tag="e"):
        self.n = state.shape[0]
        self.state = state
        self.center = center if center else vp.vector(0,0,0)
        self.radius = radius
        self.sphere_color = sphere_color if sphere_color != None else vp.color.blue
        self.star_color = star_color if star_color != None else vp.color.white
        self.arrow_color = self.sphere_color if arrow_color == None else arrow_color
        self.show_sphere = show_sphere
        self.show_stars = show_stars
        self.show_arrow = show_arrow
        self.show_tag = show_tag
        self.show_tag_on_stars = show_tag_on_stars
        self.tag = tag

        self.vsphere = vp.sphere(radius=self.radius,\
                                 color=self.sphere_color,\
                                 opacity=0.3,\
                                 visible=self.show_sphere) 
        self.varrow = vp.arrow(color=self.arrow_color,\
                               shaftwidth=0.05,\
                               emissive=True,\
                               visible=self.show_arrow)
        self.vstars = [vp.sphere(radius=0.1*self.radius,\
                                 color=self.star_color,\
                                 opacity=0.7,\
                                 emissive=True,\
                                 visible=self.show_stars) for i in range(self.n-1)] 
        self.vtag = vp.text(text=self.tag,\
                            color=self.arrow_color,\
                            align="center",\
                            height=0.2*self.radius,\
                            visible=self.show_tag) 
        self.vtags = [vp.text(text=self.tag,\
                              color=self.arrow_color,\
                              align="center",\
                              height=0.2*self.radius,\
                              visible=self.show_tag_on_stars) for i in range(self.n-1)]
    
    def display_visuals(self):
        vp.rate(100)
        if self.show_sphere:
            self.vsphere.pos = self.center
            self.vsphere.radius = self.radius
            self.vsphere.color = self.sphere_color
            self.vsphere.opacity = 0.3
        self.state.dims = [[self.n],[1]]
        stars_xyz = q_SurfaceXYZ(self.state) if (self.show_stars or (self.tag and self.show_tag_on_stars)) else None
        if self.show_stars:
            for i, vstar in enumerate(self.vstars):
                vstar.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vstar.radius = 0.1*self.radius
                vstar.color = self.star_color
                vstar.opacity = 0.7
                vstar.emissive = True
        spin_axis = self.spin_axis() if (self.show_arrow or (self.tag and self.show_tag)) else None
        if self.show_arrow:
            self.varrow.pos = self.center
            self.varrow.axis = self.radius*vp.vector(*spin_axis)
            self.varrow.color = self.arrow_color
            self.varrow.shaftwidth = 0.05
            self.varrow.emissive = True
        if self.tag and self.show_tag:
            self.vtag.pos = self.radius*vp.vector(*spin_axis) + self.center
            self.vtag.color = self.arrow_color
        if self.tag and self.show_tag_on_stars:
            for i, vtag in enumerate(self.vtags):
                vtag.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vtag.color = self.arrow_color
    
    def spin_operators(self):
        spin = (self.n-1.)/2.
        return {"X": qutip.jmat(spin, "x"),\
                "Y": qutip.jmat(spin, "y"),\
                "Z": qutip.jmat(spin, "z"),\
                "+": qutip.jmat(spin, "+"),\
                "-": qutip.jmat(spin, "-")}

    def spin_axis(self):
        spin_ops = self.spin_operators()
        self.state.dims = [[self.n], [1]]
        spin_axis = np.array([[qt.expect(spin_ops["X"], self.state)],\
                              [qt.expect(spin_ops["Y"], self.state)],\
                              [qt.expect(spin_ops["Z"], self.state)],\
                              [qt.expect(qt.identity(self.n), self.state)]])
        return normalize(spin_axis[:-1])

    def evolve(self, operator, inverse=False, dt=0.01):
        unitary = (-2*math.pi*im()*dt*operator).expm()
        if inverse:
            unitary = unitary.dag()
        self.state.dims = [[self.n],[1]]
        self.state = unitary*self.state

##################################################################################################################

class MajoranaDecisionSphere:
    def __init__(self, question, question_space, initial_answer=None):
        self.question = question
        self.question_space = question_space
        self.n_choices = len(question)
        self.initial_answer = qt.Qobj(sum([question_space[i] for i in range(self.n_choices)]))\
                                if initial_answer == None else initial_answer

        self.decider_sphere = MajoranaSphere(self.initial_answer, show_sphere=True)
        self.answer_spheres = []
        self.answer_colors = [vp.vector(*np.random.rand(3)) for i in range(self.n_choices)]
        for i in range(self.n_choices):
            answer_vector = qt.Qobj(question_space[i])
            answer_vector.dims = [[self.n_choices],[1]]
            self.answer_spheres.append(MajoranaSphere(answer_vector,\
                                                      show_sphere=False,
                                                      sphere_color=self.answer_colors[i],\
                                                      star_color=self.answer_colors[i],\
                                                      show_tag=True,\
                                                      tag=self.question[i]))

        self.done = False
        self.save = False

    def keyboard(self, event):
        key = event.key
        spin_ops = self.decider_sphere.spin_operators()
        if key == "a":
            self.decider_sphere.evolve(spin_ops['X'], inverse=True)
        elif key == "d":
            self.decider_sphere.evolve(spin_ops['X'], inverse=False)
        elif key == "s":
            self.decider_sphere.evolve(spin_ops['Z'], inverse=True)
        elif key == "w":
            self.decider_sphere.evolve(spin_ops['Z'], inverse=False)
        elif key == "z":
            self.decider_sphere.evolve(spin_ops['Y'], inverse=True)
        elif key == "e":    
            self.decider_sphere.evolve(spin_ops['Y'], inverse=False)
        elif key == "x":
            self.done = True
        elif key == "q":
            self.done = True
            self.save = True

    def decide(self):
        answer = self.display_vprompt()
        if not answer:
            answer = self.display_txtprompt()
        del self.decider_sphere
        for answer_sphere in self.answer_spheres:
            del answer_sphere
        return answer

    def display_vprompt(self):
        vp.scene.bind('keydown', self.keyboard)
        print(crayons.red(    "\t               +Z             "))
        print(crayons.green(  "\t               w       +Y     "))
        print(crayons.blue(   "\t      q quit   |     e        "))
        print(crayons.yellow( "\t               |   /          "))
        print(crayons.magenta("\t               | /            "))
        print(crayons.cyan(   "\t -X a  ________*________ d +X "))
        print(crayons.magenta("\t             / |              "))
        print(crayons.yellow( "\t           /   |              "))
        print(crayons.blue(   "\t        -Y     |     x save   "))
        print(crayons.green(  "\t      z        s              "))
        print(crayons.red(    "\t              -Z              "))
        while not self.done:
            self.decider_sphere.display_visuals()
            for answer_sphere in self.answer_spheres:
                answer_sphere.display_visuals()
        vp.scene.unbind('keydown', self.keyboard)        
        if self.save:
            return self.decider_sphere.state
        else:
            return None

    def display_txtprompt(self):
        for i in range(self.n_choices):
            if i != self.n_choices-1:
                print(("\t%d"+crayons.red('.')+"%s") % (i, self.question[i]))
            else:
                print(("\t%d"+crayons.red('.')+"%s"+crayons.magenta("?")) % (i, self.question[i])) 
        done = False
        answer = None
        while not done:
            answer = display_inner_prompt()
            if answer == "":
                print("refusing question.")
                done = True
            elif answer.isdigit():
                answer_index = int(answer)
                if answer_index >= 0 and answer_index < self.n_choices:
                    return qt.Qobj(self.question_space[answer_index])
                    done = True
                else:
                    display_error(message="not enough options for you!")
            else:
                display_error(message="use answer #!")
        return answer

##################################################################################################################


maj = MajoranaDecisionSphere(["a", "b", "c"], np.eye(3))
print(maj.decide())
