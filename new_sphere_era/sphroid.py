import os
import sys
import time
import math
import qutip
import cmath
import sympy
import scipy
import ephem
import ephem.stars
import mpmath
#import pyglet
import pickle
import random
import crayons
import inspect
import geopy
import datetime
import operator
import threading
import itertools
import functools
import qutip as qt
import numpy as np
import vpython as vp

##################################################################################################################

def im():
    return complex(0, 1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

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

def spin_operators(n):
    spin = (n-1.)/2.
    return {"X": qutip.jmat(spin, "x"),\
            "Y": qutip.jmat(spin, "y"),\
            "Z": qutip.jmat(spin, "z"),\
            "+": qutip.jmat(spin, "+"),\
            "-": qutip.jmat(spin, "-")}

def spin_axis(n, state):
    state_copy = state.copy()
    state_copy.dims = [[n], [1]]
    spin_ops = spin_operators(n)
    spin_axis = np.array([[qt.expect(spin_ops["X"], state_copy)],\
                          [qt.expect(spin_ops["Y"], state_copy)],\
                          [qt.expect(spin_ops["Z"], state_copy)],\
                          [qt.expect(qt.identity(n), state_copy)]])
    return normalize(spin_axis)[:-1]

##################################################################################################################

class Soul:
    def __init__(self, name):
        self.name = name

        self.vocabulary = []
        self.counts = []
        self.cross_counts = None
        self.concordance_matrix = None
        self.symbol_basis = {}

        self.questions = []

    def update_vocabulary(self, answers):
        for i in range(len(answers)):
            answer = answers[i]
            if answer not in self.vocabulary:
                self.vocabulary.append(answer)
                self.counts.append(0)
                if len(self.vocabulary) == 1:
                    self.cross_counts = np.array([[0]], dtype=np.complex128)
                    self.concordance_matrix = np.array([[0]], dtype=np.complex128)
                else:
                    self.cross_counts = np.hstack((self.cross_counts,\
                                                            np.zeros((len(self.vocabulary)-1,1),\
                                                               dtype=self.cross_counts.dtype)))
                    self.cross_counts = np.vstack((self.cross_counts,\
                                                         np.zeros((1,len(self.vocabulary)),\
                                                             dtype=self.cross_counts.dtype)))
                    self.concordance_matrix = np.hstack((self.concordance_matrix,\
                                                            np.zeros((len(self.vocabulary)-1,1),\
                                                               dtype=self.concordance_matrix.dtype)))
                    self.concordance_matrix = np.vstack((self.concordance_matrix,\
                                                         np.zeros((1,len(self.vocabulary)),\
                                                             dtype=self.concordance_matrix.dtype)))

    def update_counts(self, answers):
        phases = np.roots([1] + [0]*(len(answers)-1) + [-1])
        for i in range(len(answers)):
            answer = answers[i]
            vocab_index = self.vocabulary.index(answer)
            self.counts[vocab_index] += 1
            for j in range(len(answers)):
                if j != i:
                    other_answer = answers[j]
                    other_vocab_index = self.vocabulary.index(other_answer)
                    self.cross_counts[vocab_index, other_vocab_index] += (phases[j]/phases[i])

    def update_symbol_basis(self):
        n = len(self.vocabulary)
        for i in range(n):
            for j in range(n):
                term = n*self.cross_counts[i][j]/(self.counts[i]*self.counts[j])
                if term == 0:
                    self.concordance_matrix[i][j] = 0
                else:
                    self.concordance_matrix[i][j] = cmath.log(n*self.cross_counts[i][j]/(self.counts[i]*self.counts[j]))
        self.left_basis, self.diag, self.right_basis = np.linalg.svd(self.concordance_matrix)
        for i in range(n):
             self.symbol_basis[self.vocabulary[i]] = (self.diag[i]**2) * qt.tensor(qt.Qobj(self.left_basis[i]), qt.Qobj(self.right_basis[i]))

    def add_question(self, answers):
        self.update_vocabulary(answers)
        self.update_counts(answers)
        self.update_symbol_basis()
        self.questions.append(Question(self, answers=answers))

##################################################################################################################

class Question:
    def __init__(self, soul, answers=None, syntax=None, children=None, dt=0.001, center=None, radius=None):
        self.soul = soul
        if answers != None:
            self.answers = answers
            self.dim = len(answers)
            self.question_space = self.construct_question_space_from_answers(self.answers)
            self.pure_answer = qt.Qobj(sum([self.question_space[i] for i in range(self.dim)]))
            self.parent = None
            self.children = None
            self.syntax = "I"
            self.state = self.pure_answer.copy()
            self.state.dims = [[self.dim], [1]]
            self.impure_answer = None
        elif syntax != None and children != None:
            self.syntax = syntax
            self.children = children
            self.answers = self.children[0].answers[:]
            for child in self.children[1:]:
                self.answers = self.tensor_answer_strings(self.answers, child.answers)
            self.dim = len(self.answers)
            self.question_space = self.children[0].question_space.copy()
            for child in self.children[1:]:
                self.question_space = np.outer(self.question_space, child.question_space)
            self.pure_answer = self.children[0].pure_answer.copy()
            for child in self.children[1:]:
                self.pure_answer = qt.tensor(self.pure_answer, child.pure_answer)
            self.parent = None
            for child in self.children:
                child.parent = self
            self.state = self.pure_answer.copy()
            self.state.dims = [[self.dim], [1]]
            self.impure_answer = None
        self.center = center
        self.radius = radius
        self.dt = dt
        self.done = False
        self.pause = False
    
    ##################################################################################################################

    def tensor_answer_strings(self, answers_a, answers_b):
        tensor_answers = []
        for a in answers_a:
            for b in answers_b:
                tensor_answers.append("(%s|%s)" % (a, b))
        return tensor_answers

    def tensor_child_colors(self, colors_a, colors_b):
        tensor_colors = []
        for a in colors_a:
            for b in colors_b:
                tensor_answers.append(a+b)
        return tensor_colors

    def construct_question_space_from_answers(self, answers):
        d = len(self.soul.vocabulary)**2    
        sentence = [] 
        for i in range(len(answers)):
            basis_copy = self.soul.symbol_basis[answers[i]].copy()
            basis_copy.dims = [[d],[1]]
            basis_vector = self.soul.symbol_basis[answers[i]].full().T[0]
            sentence.append(basis_vector)
        sentence = np.array(sentence).T
        sentence_conj = np.conjugate(sentence.T)
        question_space = []
        for i in range(len(answers)):
            row = []
            for j in range(len(answers)):
                row.append(np.inner(sentence[:,i], sentence_conj[j,:]))
            question_space.append(row)
        return np.array(question_space)

    def evolve(self, operator, inverse=False, dt=0.01):
        if self.parent != None:
            self.parent.upgrade_operator(self, operator, inverse=inverse, dt=dt)
        unitary = (-2*math.pi*im()*dt*operator).expm()
        if inverse:
            unitary = unitary.dag()
        old_dims = self.pure_answer.dims[:]
        self.pure_answer.dims = [[self.dim], [1]]
        self.state = self.pure_answer.copy()
        self.pure_answer = unitary*self.pure_answer
        self.pure_answer.dims = old_dims
        if self.children != None:
            #self.state.dims = self.pure_answer.dims[:]
            for i in range(len(self.children)):
                self.children[i].propagate(self.pure_answer.ptrace(i))
                #self.state.dims = [[self.dim], [1]]

    def upgrade_operator(self, child, operator, inverse, dt):
        i = self.children.index(child)
        self.state.dims = self.pure_answer.dims[:]
        upgraded = None
        if i == 0:
            upgraded = operator
        else:
            upgraded = qt.identity(self.state.dims[0][0])
        for j in range(1, len(self.state.dims[0])):
            if j == i:
                upgraded = qt.tensor(upgraded, operator)
            else:
                upgraded = qt.tensor(upgraded, qt.identity(self.state.dims[0][j]))
        if self.parent != None:
            self.upgrade_operator(self, upgraded, inverse=inverse, dt=dt)
        else:
            unitary = (-2*math.pi*im()*dt*upgraded).expm()
            #self.state = unitary*self.state
            old_dims = self.pure_answer.dims[:]
            self.pure_answer.dims = self.state.dims
            self.pure_answer = unitary*self.pure_answer
            self.pure_answer.dims = old_dims
            for i in range(len(self.children)):
                self.children[i].propagate(self.state.ptrace(i))

    def propagate(self, new_state):
        self.impure_answer = new_state
        if self.children != None:
            self.impure_answer.dims = [self.pure_answer.dims[0], self.pure_answer.dims[0]]
            for i in range(len(self.children)):
                self.children[i].propagate(self.impure_answer.ptrace(i))
            self.impure_answer.dims = [[self.dim], [self.dim]]

    ##################################################################################################################

    def display(self):
        self.init_visuals()
        vp.scene.bind('keydown', self.keyboard)
        vp.scene.bind('click', self.mouse)
        while not self.done:
            if not self.pause:
                self.update_visuals()
        self.invisible()

    def mouse(self, event):
        pick = vp.scene.mouse.pick
        if self.children != None:
            for child in self.children:
                if pick == child.reference_sphere:
                    vp.scene.unbind('keydown', self.keyboard)
                    vp.scene.unbind('click', self.mouse)
                    vp.scene.bind('keydown', child.keyboard)
                    vp.scene.bind('click', child.mouse)
                    return True
        if self.parent != None:
            if pick == self.parent.reference_sphere:
                vp.scene.unbind('keydown', self.keyboard)
                vp.scene.unbind('click', self.mouse)
                vp.scene.bind('keydown', self.parent.keyboard)
                vp.scene.bind('click', self.parent.mouse)
                return True
            else:
                vp.scene.unbind('keydown', self.keyboard)
                vp.scene.unbind('click', self.mouse)
                vp.scene.bind('keydown', self.parent.keyboard)
                vp.scene.bind('click', self.parent.mouse)
                found = self.parent.mouse(event)
                if not found:
                    vp.scene.bind('keydown', self.keyboard)
                    vp.scene.bind('click', self.mouse)
                    return False
                else:
                    return True

    def keyboard(self, event):
        key = event.key
        spin = (self.dim-1.)/2.
        if key == "a":
            self.evolve(qt.jmat(spin, 'x'), inverse=True)
        elif key == "d":
            self.evolve(qt.jmat(spin, 'x'), inverse=False)
        elif key == "s":
            self.evolve(qt.jmat(spin, 'z'), inverse=True)
        elif key == "w":
            self.evolve(qt.jmat(spin, 'z'), inverse=False)
        elif key == "z":
            self.evolve(qt.jmat(spin, 'y'), inverse=True)
        elif key == "x":    
            self.evolve(qt.jmat(spin, 'y'), inverse=False)
        elif key == "q":
            self.done = True
        elif key == "e":
            if self.pause == True:
                self.pause = False
            else:
                self.pause = True

    def get_center(self, child):
        if self.center != None:
            return self.center + vp.vector(*spin_axis(child.dim, child.pure_answer))
        else:
            return vp.vector(0,0,0)

    def get_radius(self, child):
        if self.radius != None:
            return self.radius*(1./len(self.children)**2)
        else:
            return 1

    def init_visuals(self):
        if self.parent != None:
            self.center = self.parent.get_center(self)
            self.radius = self.parent.get_radius(self)
        else:
            if self.center == None:
                self.center = vp.vector(0,0,0)
            if self.radius == None:
                self.radius = 1
        if self.children != None:
            for child in self.children:
                child.init_visuals()
            self.answer_colors = []
            for child_a in self.children:
                for child_b in self.children:
                    self.answer_colors.append(child_a.color+child_b.color)
            summ = self.answer_colors[0]
            for answer_color in self.answer_colors[1:]:
                summ = summ + answer_color
            summ = summ/len(self.answer_colors)
            self.color = summ
            #self.color = sum(self.answer_colors)/len(self.answer_colors)
        else:
            self.answer_colors = [np.random.randn(3) for i in range(self.dim)]
            summ = self.answer_colors[0]
            for answer_color in self.answer_colors[1:]:
                summ = summ + answer_color
            summ = summ/len(self.answer_colors)
            self.color = summ
            #self.color = sum(self.answer_colors)/len(self.answer_colors)
        i = 0
        col = vp.vector(*np.absolute(self.color))
        self.reference_sphere = vp.sphere(pos=self.center,\
                                          radius=self.radius,\
                                          color=col,\
                                          opacity=0.2)
        self.answer_spheres = []
        for answer_space in self.question_space:
            xyzs = v_SurfaceXYZ(answer_space)
            col = vp.vector(*np.absolute(self.answer_colors[i]))
            ax = vp.vector(*self.answer_colors[i])
            answer_stars = vp.points(pos=[self.center+self.radius*vp.vector(*xyz) for xyz in xyzs],\
                                     color=col,\
                                     radius=10*self.radius, opacity=0.4, emissive=True)
            answer_tags = [vp.text(text=self.answers[i],\
                                    align="center",\
                                    pos=self.center+self.radius*vp.vector(*xyz),\
                                    color=col,\
                                    height=0.125*self.radius,\
                                    opacity=0.4,
                                    axis=ax) for xyz in xyzs]
            answer_spin_axis = spin_axis(len(xyzs)+1, qt.Qobj(answer_space))
            answer_arrow = vp.arrow(pos=self.center,\
                                    axis=self.radius*vp.vector(*answer_spin_axis),\
                                    color=col,\
                                    shaftwidth=0.007, opacity=0.7)
            answer_tag = vp.text(text=self.answers[i],
                                  align="center",\
                                  pos=self.radius*vp.vector(*answer_spin_axis) + self.center,\
                                  color=col,\
                                  height=0.16*self.radius,\
                                  opacity=0.5,\
                                  axis=ax)
            self.answer_spheres.append([answer_stars, answer_tags, answer_arrow, answer_tag])
            i += 1
        pure_xyzs = v_SurfaceXYZ(self.pure_answer.full().T[0])
        pure_answer_stars = vp.points(pos=[self.center+self.radius*vp.vector(*pure_xyz) for pure_xyz in pure_xyzs],\
                                       color=vp.color.white,\
                                       radius=10*self.radius, emissive=True)
        pure_spin_axis = spin_axis(len(pure_xyzs)+1, self.pure_answer)
        pure_answer_arrow = vp.arrow(pos=self.center,\
                                     axis=self.radius*vp.vector(*pure_spin_axis),\
                                     color=vp.color.white,\
                                     shaftwidth=0.01)
        self.pure_answer_sphere = [pure_answer_stars, pure_answer_arrow]
        if self.parent != None:
            state_xyzs = v_SurfaceXYZ(self.state.full().T[0])
            state_stars = vp.points(pos=[self.center+self.radius*vp.vector(*state_xyz) for state_xyz in state_xyzs],\
                                    color=col,\
                                    radius=10*self.radius, opacity=0.8)
            state_spin_axis = spin_axis(len(state_xyzs)+1, self.state)
            state_arrow = vp.arrow(pos=self.center,\
                                   axis=self.radius*vp.vector(*state_spin_axis),\
                                   color=col,\
                                   shaftwidth=0.03, opacity=0.8)
            self.state_sphere = [state_stars, state_arrow]

    def update_visuals(self):
        if self.impure_answer != None:
            unitary = (-2*math.pi*im()*self.dt*self.impure_answer).expm()
            self.state = unitary*self.state
        if self.parent != None:
            self.center = self.parent.get_center(self)
            self.radius = self.parent.get_radius(self)
        else:
            self.center = vp.vector(0,0,0)
            self.radius = 1
        if self.children != None:
            for child in self.children:
                child.update_visuals()
        self.reference_sphere.pos = self.center
        self.reference_sphere.radius = self.radius
        counter = 0
        for answer_space in self.question_space:
            answer_stars, answer_tags, answer_arrow, answer_tag = self.answer_spheres[counter]
            xyzs = v_SurfaceXYZ(answer_space)
            for i in range(len(xyzs)):
                answer_stars.modify(i, pos=self.center+self.radius*vp.vector(*xyzs[i]))
                answer_tags[i].pos = self.center+self.radius*vp.vector(*xyzs[i])
            answer_spin_axis = spin_axis(len(xyzs)+1, qt.Qobj(answer_space))
            answer_arrow.pos = self.center
            answer_arrow.axis = self.radius*vp.vector(*answer_spin_axis)
            answer_tag.pos = self.radius*vp.vector(*answer_spin_axis) + self.center
            counter += 1  
        pure_answer_stars, pure_answer_arrow = self.pure_answer_sphere
        pure_xyzs = v_SurfaceXYZ(self.pure_answer.full().T[0])
        for i in range(len(pure_xyzs)):
            pure_answer_stars.modify(i, pos=self.center+self.radius*vp.vector(*pure_xyzs[i]))
        pure_spin_axis = spin_axis(len(pure_xyzs)+1, self.pure_answer)
        pure_answer_arrow.pos = self.center
        pure_answer_arrow.axis = self.radius*vp.vector(*pure_spin_axis)
        if self.parent != None:
            state_stars, state_arrow = self.state_sphere
            state_xyzs = v_SurfaceXYZ(self.state.full().T[0])
            for i in range(len(state_xyzs)):
                state_stars.modify(i, pos=self.center+self.radius*vp.vector(*state_xyzs[i]))
            state_spin_axis = spin_axis(len(state_xyzs)+1, self.state)
            state_arrow.pos = self.center
            state_arrow.axis = self.radius*vp.vector(*state_spin_axis)

    def invisible(self):
        self.reference_sphere.visible = False
        for answer_sphere in self.answer_spheres:
            answer_stars, answer_tags, answer_arrow, answer_tag = answer_sphere
            answer_stars.visible = False
            for tag in answer_tags:
                tag.visible = False
            answer_arrow.visible = False
            answer_tag.visible = False
        pure_answer_stars, pure_answer_arrow = self.pure_answer_sphere
        pure_answer_stars.visible = False
        pure_answer_arrow.visible = False
        if self.parent != None:
            state_stars, state_arrow = self.state_sphere
            state_stars.visible = False
            state_arrow.visible = False
        if self.children != None:
            for child in self.children:
                child.invisible()

##################################################################################################################

matthew = Soul("matthew")
matthew.add_question(["YES", "NO"])
matthew.add_question(["yes", "no"])
matthew.add_question(["yes", "no", "maybe"])
matthew.add_question(["no", "okay"])
matthew.add_question(["no", "maybe"])
matthew.add_question(["no", "maybe", "okay", "yes"])

q = Question(matthew, syntax="", children=[matthew.questions[0],matthew.questions[1],matthew.questions[3]])
q.display()
#matthew.questions[2].display()