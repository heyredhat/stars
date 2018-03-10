##################################################################################################################
#
# SPHERES 1.0 (C) 2018 MATTHEW B. WEISS
#
##################################################################################################################

import os
import sys
import time
import math
import qutip
import cmath
import sympy
import scipy
import mpmath
import pyglet
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

##################################################################################################################

def im():
    return complex(0, 1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def str_to_bool(s):
    if s == "True" or s == "true":
        return True
    elif s == "False" or s == "false":
        return False
    else:
        return None

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def totalset(iterable):
    ps = list(powerset(iterable))
    total = []
    for p in ps:
        total.extend(list(itertools.permutations(p)))
    return total

def relations(iterable):
    rel = []
    for x in totalset(iterable):
        if len(x) > 1:
            rel.append(list(x))
    return rel

def symmeterize(pieces, labels=None):
    n = len(pieces)
    if labels == None:
        labels = list(range(n))
    unique_labels = list(set(labels))
    label_counts = [0 for i in range(len(unique_labels))]
    label_permutations = itertools.permutations(labels, n)
    for permutation in label_permutations:
        for i in range(len(unique_labels)):
            label_counts[i] += list(permutation).count(unique_labels[i])
    normalization = 1./math.sqrt(functools.reduce(operator.mul, [math.factorial(count) for count in label_counts], 1)/math.factorial(n))    
    permutations = list(itertools.permutations(pieces, n))
    perm_states = []
    for permutation in permutations:
        perm_state = permutation[0]
        for state in permutation[1:]:
            perm_state = qt.tensor(perm_state, state)
        perm_state.dims = [[perm_state.shape[0]],[1]]
        perm_states.append(perm_state)
    tensor_sum = sum(perm_states)
    return normalization*tensor_sum

def perm_parity(lst):
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity    

def antisymmeterize(pieces):
    n = len(pieces)
    normalization = 1./math.sqrt(math.factorial(n))
    permutations = list(itertools.permutations(pieces, n))
    int_permutations = list(itertools.permutations(list(range(n)), n))
    perm_states = []
    for i in range(len(permutations)):
        permutation = permutations[i]
        perm_state = permutation[0]
        for state in permutation[1:]:
            perm_state = qt.tensor(perm_state, state)
        perm_state = perm_state*perm_parity(list(int_permutations[i]))
        perm_state.dims = [[perm_state.shape[0]],[1]]
        perm_states.append(perm_state)
    tensor_sum = sum(perm_states)
    return normalization*tensor_sum

def direct_sum(pieces):
    return qt.Qobj(normalize(np.concatenate([piece.full().T[0] for piece in pieces])))

##################################################################################################################

def spin_axis(state):
    n = state.shape[0]
    spin = (n-1.)/2.
    if state.isket:
        state.dims = [[n], [1]]
    else:
        state.dims = [[n],[n]]
    spin_axis = np.array([[qt.expect(qt.jmat(spin,"x"), state)],\
                          [qt.expect(qt.jmat(spin,"y"), state)],\
                          [qt.expect(qt.jmat(spin,"z"), state)],\
                          [qt.expect(qt.identity(n), state)]])
    return normalize(spin_axis[:-1])

##################################################################################################################

def qubit_to_vector(qubit):
    return np.array([[qutip.expect(qutip.sigmax(), qubit)],\
                     [qutip.expect(qutip.sigmay(), qubit)],\
                     [qutip.expect(qutip.sigmaz(), qubit)],\
                     [qutip.expect(qutip.identity(2), qubit)]])

def vector_to_qubit(vector):
    x, y, z, t = vector.T[0]
    return (1./2)*(x*qutip.sigmax() + y*qutip.sigmay() + z*qutip.sigmaz() + t*qutip.identity(2))

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
def inverse_stereographic_projectionAll(xyz):
    if isinstance(xyz, int):
        n = xyz
        return np.array([[1]]+[[0]]*(n-1))
    coordinates = xyz.T[0][::-1]
    s = sum([coordinate**2 for coordinate in coordinates])
    sphere = [[(s - 1)/(s + 1)]]
    for coordinate in coordinates:
        sphere.append([(2*coordinate)/(s + 1)])
    return np.array(sphere[::-1])

##################################################################################################################

class Variable:    
    def __init__(self):
        self.value = None
        self.dependencies = []
        self.touched = 0

    def tie(self, other, transformation):
        self.dependencies.append({"other": other, "transformation": transformation})

    def plug(self, value):
        self.value = value
        if self.touched == 0:
            self.touched = 1
        elif self.touched == 1:
            self.touched = 0
        for dependency in self.dependencies:
            other, transformation = dependency["other"], dependency["transformation"]
            if other.touched != self.touched:
                other.plug(transformation(self.value))

    def __str__(self):
        return str(self.value)

##################################################################################################################

class Soul:
    def __init__(self, name):
        self.name = name

        self.vocabulary = []
        self.concordance_matrix = None
        self.symbol_basis = {}

        self.questions = []
        self.ordering = []

        self.state = None

    def prepare_for_question(self, question):
        self.expand_vocabulary(question)
        self.update_symbol_basis(question)
        return self.construct_question_space(question)

    def expand_vocabulary(self, question):
        for i in range(len(question)):
            word = question[i]
            if word not in self.vocabulary:
                self.vocabulary.append(word)
                if len(self.vocabulary) == 1:
                    self.concordance_matrix = np.array([[0]], dtype=np.complex128)
                else:
                    self.concordance_matrix = np.hstack((self.concordance_matrix,\
                                                            np.zeros((len(self.vocabulary)-1,1),\
                                                               dtype=self.concordance_matrix.dtype)))
                    self.concordance_matrix = np.vstack((self.concordance_matrix,\
                                                         np.zeros((1,len(self.vocabulary)),\
                                                             dtype=self.concordance_matrix.dtype)))
    
    def update_symbol_basis(self, question):
        phases = np.roots([1] + [0]*(len(question)-1) + [-1])
        for i in range(len(question)):
            word = question[i]
            for j in range(len(question)):
                if j != i:
                    other_word = question[j]
                    self.concordance_matrix[self.vocabulary.index(word)]\
                                           [self.vocabulary.index(other_word)]\
                                                += (phases[j]/phases[i])
        self.left_basis, self.diag, self.right_basis = np.linalg.svd(normalize(self.concordance_matrix))
        for i in range(len(self.vocabulary)):
             self.symbol_basis[self.vocabulary[i]] = qt.tensor(qt.Qobj(self.left_basis[i]), qt.Qobj(self.right_basis[i]))*self.diag[i]**2

    def construct_question_space(self, question):
        d = len(self.vocabulary)**2    
        sentence = [] 
        for i in range(len(question)):
            basis_copy = self.symbol_basis[question[i]].copy()
            basis_copy.dims = [[d],[1]]
            basis_vector = self.symbol_basis[question[i]].full().T[0]
            sentence.append(basis_vector)
        sentence = np.array(sentence).T
        sentence_conj = np.conjugate(sentence.T)
        question_space = []
        for i in range(len(question)):
            row = []
            for j in range(len(question)):
                row.append(np.inner(sentence[:,i], sentence_conj[j,:]))
            question_space.append(row)
        return np.array(question_space)

    def add_question(self, question, answer):
        self.questions.append([question, answer])
        self.ordering.append(len(self.questions)-1)

    def change_answer(self, question_index, new_answer):
        question, old_answer = self.questions[question_index]
        self.questions[question_index] = [question, new_answer]

    def question_to_probabilities(self, question_index):
        question, answer = self.questions[question_index]
        question_space = self.construct_question_space(question)
        probabilities = []
        for i in range(len(question)):
            amplitude = np.inner(answer.full().T[0], np.conjugate(question_space[i].T))
            probabilities.append((amplitude*np.conjugate(amplitude)).real)
        probabilities = normalize(np.array(probabilities)).tolist()
        return [[question[i],probabilities[i]] for i in range(len(probabilities))]

    def construct_state(self):
        if len(self.ordering) == 1:
            return self.unorder(self.ordering[0])

    def unorder(self, ordering):
        how, questions, dims = ordering
        QUESTIONS = [None]*len(questions)
        STATE = None
        for i in range(len(questions)):
            if isinstance(questions[i], int):
                QUESTIONS[i] = self.questions[i][1]
            else:
                QUESTIONS[i] = self.unorder(questions[i])
        if how == "before":
            STATE = direct_sum(QUESTIONS)
        elif how == "after":
            STATE = direct_sum(QUESTIONS[::-1])
        elif how == "excludes":
            STATE = antisymmeterize(QUESTIONS)
        elif how == "coexists":
            STATE = symmeterize(QUESTIONS)
        elif how == "covers":
            STATE = qt.tensor(QUESTIONS)
        elif how == "is_covered_by":
            STATE = qt.tensor(QUESTIONS[::-1])
        elif how == "sum":
            STATE = sum(QUESTIONS)
        return STATE

##################################################################################################################

class Spheres:
    def __init__(self):
        self.souls = {}
        self.questions = []

    def add_soul(self, name):
        self.souls[name] = Soul(name)

    def add_question(self, answers):
        self.questions.append(answers)

    def remove_soul(self, name):
        del self.souls[name]

    def remove_question(self, question_index):
        del self.questions[question_index]

    def clear_souls(self):
        self.souls = {}

    def clear_questions(self):
        self.questions = []

    def clear(self):
        self.clear_souls()
        self.clear_questions()

spheres = pickle.load(open(sys.argv[1], "rb")) if len(sys.argv) == 2 else Spheres()

##################################################################################################################

def display_startup_message():
    os.system("clear")
    print("welcome to %s" % \
            crayons.red("s")+\
            crayons.white("p")+\
            crayons.yellow("h")+\
            crayons.green("e")+\
            crayons.cyan("r")+\
            crayons.blue("e")+\
            crayons.magenta("s"))

def display_loop_prompt():
    return input(crayons.cyan(":")+crayons.blue(">")+" ").lower().split()

def display_inner_prompt():
    return input(crayons.green('.')+crayons.magenta('.')+crayons.blue('.'))

def display_error(message=None):
    if message:
        print(crayons.red("?:")+crayons.blue(" %s" % message))
    else:
        print(crayons.red("?"))

def display_question(soul, question):
    s = ""
    if isinstance(question, int):
        s += "{#%s:%s,d=%d}" % (question, soul.questions[question][0], len(soul.questions[question][0]))
    else:
        how, questions, dims = question
        s += "(%s:%s,d=%s)" % (how, ",".join([display_question(soul, q) for q in questions]), str(dims))
    return s

##################################################################################################################

def does_soul_exist(name):
    global spheres
    if name in spheres.souls.keys():
        return True
    else:
        display_error(message="no soul named %s!" % (name))
        return False

def does_question_exist(question_index):
    global spheres
    if question_index.isdigit():
        question_index = int(question_index)
        if question_index >= 0 and question_index < len(spheres.questions):
            return True
        else:
            display_error(message="no question #%d!" % (question_index))
            return False
    else:
        display_error(message="use question #!")
        return False

def does_question_index_exist_for_soul(soul_name, soul_question_index):
    global spheres
    if does_soul_exist(soul_name):
        if soul_question_index.isdigit():
            soul_question_index = int(soul_question_index)
            if soul_question_index >= 0 and soul_question_index < len(spheres.souls[soul_name].questions):
                return True
            else:
                display_error(message="no question #%d for %s!" % (soul_question_index, soul_name))
                return False
        else:
            display_error(message="use question # for %s!" % (soul_name))
            return False

def does_question_exist_for_soul(soul_name, question_answers):
    global spheres
    if does_soul_exist(soul_name):
        soul = spheres.souls[soul_name]
        for question in soul.questions:
            answers, answer = question
            if answers == question_answers:
                return True
        return False

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
                            emissive=True,\
                            visible=self.show_tag) 
        self.vtags = [vp.text(text=self.tag,\
                              color=self.arrow_color,\
                              align="center",\
                              height=0.125*self.radius,\
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

    def invisible(self):
        self.vsphere.visible = False
        self.varrow.visible = False
        for vstar in self.vstars:
            vstar.visible = False
        self.vtag.visible = False
        for vtag in self.vtags:
            vtag.visible = False

    def visible(self):
        self.vsphere.visible = self.show_sphere
        self.varrow.visible = self.show_arrow
        for vstar in self.vstars:
            vstar.visible = self.show_stars
        self.vtag.visible = self.show_tag
        for vtag in self.vtags:
            vtag.visible = self.show_tag_on_stars

##################################################################################################################

class MajoranaDensitySphere:
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

        self.eigenvalues, self.eigenvectors = self.state.eigenstates()
        self.normalized_eigenvalues = normalize(self.eigenvalues)
        self.vspheres = [MajoranaSphere(self.eigenvectors[i],\
                                        center=self.center,\
                                        radius=self.radius*self.normalized_eigenvalues[i],\
                                        sphere_color=self.sphere_color,\
                                        star_color=self.star_color,\
                                        show_stars=self.show_stars,\
                                        show_arrow=self.show_arrow,\
                                        show_tag=self.show_tag,\
                                        show_tag_on_stars=self.show_tag_on_stars,\
                                        tag=self.tag)\
                            for i in range(len(self.eigenvectors))]

    def display_visuals(self):
        for vsphere in self.vspheres:
            vsphere.display_visuals()

    def spin_operators(self):
        spin = (self.n-1.)/2.
        return {"X": qutip.jmat(spin, "x"),\
                "Y": qutip.jmat(spin, "y"),\
                "Z": qutip.jmat(spin, "z"),\
                "+": qutip.jmat(spin, "+"),\
                "-": qutip.jmat(spin, "-")}

    def spin_axis(self):
        spin_ops = self.spin_operators()
        self.state.dims = [[self.n], [self.n]]
        spin_axis = np.array([[qt.expect(spin_ops["X"], self.state)],\
                              [qt.expect(spin_ops["Y"], self.state)],\
                              [qt.expect(spin_ops["Z"], self.state)],\
                              [qt.expect(qt.identity(self.n), self.state)]])
        return normalize(spin_axis[:-1])

    def evolve(self, operator, inverse=False, dt=0.01):
        unitary = (-2*math.pi*im()*dt*operator).expm()
        if inverse:
            unitary = unitary.dag()
        self.state.dims = [[self.n],[self.n]]
        self.state = unitary*self.state*unitary.dag()

    def invisible(self):
        for vsphere in self.vspheres:
            vsphere.invisible()

    def visible(self):
        for vsphere in self.vspheres:
            vsphere.visible()

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
        elif key == "x":    
            self.decider_sphere.evolve(spin_ops['Y'], inverse=False)
        elif key == "q":
            self.done = True
        elif key == "e":
            self.done = True
            self.save = True

    def decide(self):
        answer = self.display_vprompt()
        if not answer:
            answer = self.display_txtprompt()
        self.decider_sphere.invisible()
        del self.decider_sphere
        for answer_sphere in self.answer_spheres:
            answer_sphere.invisible()
            del answer_sphere
        return answer

    def display_vprompt(self):
        vp.scene.bind('keydown', self.keyboard)
        print(crayons.red(    "\t               +Z             "))
        print(crayons.green(  "\t               w       +Y     "))
        print(crayons.blue(   "\t      q quit   |     x        "))
        print(crayons.yellow( "\t               |   /          "))
        print(crayons.magenta("\t               | /            "))
        print(crayons.cyan(   "\t -X a  ________*________ d +X "))
        print(crayons.magenta("\t             / |              "))
        print(crayons.yellow( "\t           /   |              "))
        print(crayons.blue(   "\t        -Y     |     e save   "))
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

class VocabularySphere:
    def __init__(self, soul):
        self.soul = soul
        self.word_colors = [vp.vector(*np.random.rand(3)) for v in self.soul.vocabulary]
        self.show_stars = True if len(self.soul.vocabulary) < 7 else False
        self.word_spheres = [MajoranaSphere(self.soul.symbol_basis[self.soul.vocabulary[i]],\
                                            sphere_color=self.word_colors[i],\
                                            star_color=self.word_colors[i],\
                                            tag=self.soul.vocabulary[i],\
                                            show_tag=True,\
                                            show_stars=self.show_stars,\
                                            show_tag_on_stars=True)\
                                                 for i in range(len(self.soul.vocabulary))]
        self.done = False

    def keyboard(self, event):
        key = event.key
        if key == "q":
            self.done = True

    def display(self):
        vp.scene.bind('keydown', self.keyboard)
        print(crayons.magenta("q to quit...")) 
        while not self.done:                   
            for word_sphere in self.word_spheres:
                word_sphere.display_visuals()
        for word_sphere in self.word_spheres:
            word_sphere.invisible()
            del word_sphere
        vp.scene.unbind('keydown', self.keyboard)   

##################################################################################################################

class OrderingSphere:
    def __init__(self, soul):
        self.commands = {"a":"before",\
                         "d":"after", \
                         "s":"excludes", \
                         "w":"coexists", \
                         "z":"is_covered_by", \
                         "x":"covers", \
                         "c":"sum", \
                         "q":"quit", \
                         "e":"defer"}

        self.soul = soul
        self.done = False

    def order(self):
        ordering = self.soul.ordering[:]
        while not self.done:
            if len(ordering) == 1:
                print("all tidied up!")
                self.done = True
            else:
                questions = random.choice(relations(ordering))
                how = self.display_assign_ordering(questions)
                if how == "quit":
                    self.done = True
                elif how != "defer":
                    dims = self.recurse(questions, how)
                    ordering.append([how, questions, dims])
                    for question in questions:
                        ordering.remove(question)
        return ordering

    def recurse(self, questions, how):
        dims = None
        if how in ["before", "after"]:
            dims = 0
        elif how in ["covers", "is_covered_by", "coexists", "excludes"]:
            dims = 1
        elif how == "sum":
            dims = None
        for question in questions:
            if isinstance(question, int):
                if how in ["before", "after"]:
                    dims += len(self.soul.questions[question][0])
                elif how in ["covers", "is_covered_by", "coexists", "excludes"]:
                    dims *= len(self.soul.questions[question][0])
                elif how == "sum":
                    dims = len(self.soul.questions[question][0])
            else:
                lower_how, lower_questions, lower_dims = question
                if how in ["before", "after"]:
                    dims += self.recurse(lower_questions, lower_how)
                elif how in ["covers", "is_covered_by", "coexists", "excludes"]:
                    dims *= self.recurse(lower_questions, lower_how)
                else:
                    dims = self.recurse(lower_questions, lower_how)
        return dims

    def display_assign_ordering(self, questions):
        for i in range(len(questions)):
            print(display_question(self.soul, questions[i]))
            if i < len(questions)-1:
                print("\t."+crayons.red("V")+crayons.blue("S")+".")
        print(crayons.red(    "\t                coexists               "))
        print(crayons.green(  "\t                   w       covers      "))
        print(crayons.blue(   "\t      q            |     x             "))
        print(crayons.yellow( "\t       quit...     |   /               "))
        print(crayons.magenta("\t                   | /                 "))
        print(crayons.cyan(   "\t before a  _______SUM_______ d after   "))
        print(crayons.magenta("\t                /  c                   "))
        print(crayons.yellow( "\t        is    /    |     e             "))
        print(crayons.blue(   "\t   covered  z      |      defer...     "))
        print(crayons.green(  "\t        by         s                   "))
        print(crayons.red(    "\t               excludes                "))
        ordering = display_inner_prompt()
        while ordering not in self.commands.keys():
            ordering = display_inner_prompt()
        return self.commands[ordering]

##################################################################################################################

class StateSphere:
    def __init__(self, soul):
        self.soul = soul
        self.state = self.soul.state.copy()
        self.ordering = self.soul.ordering[0]

        self.done = False

    def keyboard(self, event):
        key = event.key
        if key == "q":
            self.done = True

    def display(self):
        self.vspheres = self.create_recursive(True, self.state, self.ordering, vp.vector(0,0,0), 1)
        vp.scene.bind('keydown', self.keyboard)
        print(crayons.magenta("q to quit...")) 
        while not self.done:                   
            self.display_recursively(self.vspheres)
        self.destroy_recursively(self.vspheres)
        vp.scene.unbind('keydown', self.keyboard)

    def display_recursively(self, vspheres):
        if not isinstance(vspheres, list):
            vspheres.display_visuals()
        elif vspheres == []:
            return
        else:
            background, foreground = vspheres
            background.display_visuals()
            for fore in foreground:
                if isinstance(fore, list):
                    self.display_recursively(fore)
                else:
                    fore.display_visuals()

    def destroy_recursively(self, vspheres):
        if not isinstance(vspheres, list):
            vspheres.invisible()
        elif vspheres == []:
            return
        else:
            background, foreground = vspheres
            background.invisible()
            for fore in foreground:
                if isinstance(fore, list):
                    for f in fore:
                        self.destroy_recursively(f)
                else:
                    fore.invisible()
                    del fore

    def create_recursive(self, start, state, ordering, center, radius):
        state = state.copy()
        background_sphere = None
        new_center = center
        new_radius = radius
        if not start:
            background_center = vp.vector(*spin_axis(state))
            new_center = center + background_center
            new_radius = 0.3*radius
        if state.isket:
            color = vp.vector(*np.random.rand(3))
            background_sphere = MajoranaSphere(state,\
                                           sphere_color=color,\
                                           star_color=color,\
                                           show_stars=True,\
                                           center=new_center,\
                                           radius=new_radius)
        else:
            color = vp.vector(*np.random.rand(3))
            background_sphere = MajoranaDensitySphere(state, sphere_color=color, star_color=color, center=new_center, radius=new_radius, show_stars=True)
    
        if isinstance(ordering, int):
            return [background_sphere, []]
        else:
            how, questions, dims = ordering
            new_dims = []
            for question in questions:
                question_dims = None
                if isinstance(question, int):
                    new_dims.append(len(self.soul.questions[question][0]))
                else:
                    inner_how, inner_questions, inner_dims = question
                    new_dims.append(inner_dims)
            foreground_spheres = []
            if how in ["before", "after"]:
                for i in range(len(questions)):
                    if i == 0:
                        foreground_spheres.append(self.create_recursive(False, qt.Qobj(state.full().T[0][0:new_dims[0]]), questions[i], new_center, new_radius))
                    else:
                        foreground_spheres.append(self.create_recursive(False, qt.Qobj(state.full().T[0][new_dims[i-1]:new_dims[i]]), questions[i], new_center, new_radius))
            elif how in ["excludes", "coexists", "is_covered_by", "covers"]:
                if state.isket:
                    state.dims = [new_dims, [1]*len(new_dims)]
                else:
                    state.dims = [new_dims, new_dims]
                for i in range(len(new_dims)):
                    foreground_spheres.append(self.create_recursive(False, state.ptrace(i), questions[i], new_center, new_radius))
            elif how == "sum":
                for i in range(len(questions)):
                    foreground_spheres.append(self.create_recursive(False, state/len(questions), questions[i], new_center, new_radius))
            return [background_sphere, foreground_spheres]

##################################################################################################################

class QubitSphere:
    def __init__(self, color=vp.color.white,\
                       n_fiber_points=100,\
                       parent=None):
        self.color = color
        self.n_fiber_points = n_fiber_points
        self.parent = parent

        self.state = Variable() # 2x2 Density Matrix
        self.vector = Variable() # spacetime 4-vector
        self.angles = Variable() # 3 angles of 3-sphere

        self.state.tie(self.vector, qubit_to_vector)
        self.vector.tie(self.state, vector_to_qubit)
        self.vector.tie(self.angles, vector_to_angles)
        self.angles.tie(self.vector, angles_to_vector)

        self.vbase = vp.sphere(color=self.color,\
                                    radius=0.1,\
                                    opacity=0.7,\
                                    emissive=False)
        self.varrow = vp.arrow(pos=vp.vector(0,0,0),\
                                    color=self.color,\
                                    shaftwidth=0.03)
        self.vfiber = vp.curve(pos=[vp.vector(0,0,0) for i in range(self.n_fiber_points)],\
                                    color=self.color)

    def update(self, new_state):
        self.state.plug(new_state)

    def display_visuals(self):
        base = self.base()
        if isinstance(base, int):
            return
        base_x, base_y, base_z = [c.real for c in self.base().T[0].tolist()]
        self.vbase.pos = vp.vector(base_x, base_y, base_z)
        arrow_x, arrow_y, arrow_z = [c.real for c in self.spin_axis()]
        self.varrow.axis = vp.vector(arrow_x, arrow_y, arrow_z)
        fiber_points = self.fiber()
        for i in range(self.n_fiber_points):
            if not isinstance(fiber_points[i], int):
                    self.vfiber.modify(i, pos=vp.vector(*fiber_points[i]),\
                                          color=self.color)

    def base(self):
        return stereographic_projection(normalize(self.vector.value))

    def fiber(self):
        circle = np.linspace(0, 2*math.pi, num=self.n_fiber_points)
        fiber_points = []
        for angle in circle:
            transformation = np.array([[math.cos(angle),-1*math.sin(angle),0,0],\
                                       [math.sin(angle), math.cos(angle),0,0],\
                                       [0,0,math.cos(angle),-1*math.sin(angle)],\
                                       [0,0,math.sin(angle),math.cos(angle)]])
            fiber_points.append(np.real(stereographic_projection(normalize(np.dot(transformation, self.vector.value)))))
        return fiber_points

    def evolve(self, operator, inverse=True, dt=0.005):
        if self.parent:
            i = self.parent.qubits.index(self)
            upgraded = None
            if i == 0:
                upgraded = operator
            else:
                upgraded = qutip.identity(2)
            for j in range(1, self.parent.n_qubits):
                if j == i:
                    upgraded = qutip.tensor(upgraded, operator)
                else:
                    upgraded = qutip.tensor(upgraded, qutip.identity(2))
            unitary = (-2*math.pi*im()*upgraded*dt).expm()
            if inverse:
                unitary = unitary.dag()
            self.parent.state.dims = [[2**self.parent.n_qubits],[1]]
            unitary.dims = [[2**self.parent.n_qubits],[2**self.parent.n_qubits]]
            self.parent.state = unitary*self.parent.state
            self.parent.update()
        else:
            unitary = (-2*math.pi*im()*operator*dt).expm()
            if inverse:
                unitary = unitary.dag()
            self.update(unitary*self.state.value*unitary.dag())

    def invisible(self):
        self.vbase.visible = False
        self.varrow.visible = False
        self.vfiber.visible = False

    def spin_axis(self):
        return [qutip.expect(qutip.sigmax(), self.state.value),\
                qutip.expect(qutip.sigmay(), self.state.value),\
                qutip.expect(qutip.sigmaz(), self.state.value)]

class MultiQubitSphere:
    def __init__(self, n_qubits,\
                       center=vp.vector(0,0,0),\
                       radius=1,\
                       color=vp.color.blue,\
                       show_stars=False,\
                       show_total_spin=False,\
                       camera_follows_total_spin=False):
        self.n_qubits = n_qubits
        self.center = center
        self.radius = radius
        self.color = color

        self.show_stars = show_stars
        self.show_total_spin = show_total_spin
        self.camera_follows_total_spin = camera_follows_total_spin

        self.state = qutip.rand_ket(2**self.n_qubits)
        self.energy = qutip.rand_herm(2**self.n_qubits)

        self.vsphere = MajoranaSphere(self.state,\
                                      center=self.center,\
                                      radius=self.radius,\
                                      sphere_color=self.color,\
                                      show_stars=self.show_stars,\
                                      show_arrow=self.show_total_spin)

        self.qubit_colors = [vp.vector(*np.random.rand(3)) for i in range(self.n_qubits)]
        self.qubits = [QubitSphere(color=self.qubit_colors[i], parent=self) for i in range(self.n_qubits)]

        self.active_qubit = 0
        self.evolution_on = False
        self.done = False

        self.update()

    def keyboard(self, event):
        key = event.key
        if key.isdigit():
            i = int(key)
            if i < self.n_qubits:
                self.active_qubit = i
                print(crayons.magenta("qubit #%d active!" % self.active_qubit))
        elif key == "`":
             self.active_qubit = -1
             print(crayons.magenta("the whole is active!"))
        elif key == "a":
            if self.active_qubit == -1:
                self.evolve(self.spin_operators()["X"], inverse=True)
                self.update()
            else:
                self.qubits[self.active_qubit].evolve(qutip.sigmax(), inverse=True)
        elif key == "d":
            if self.active_qubit == -1:
                self.evolve(self.spin_operators()["X"], inverse=False)
                self.update()
            else:
                self.qubits[self.active_qubit].evolve(qutip.sigmax(), inverse=False)
        elif key == "s":
            if self.active_qubit == -1:
                self.evolve(self.spin_operators()["Z"], inverse=True)
                self.update()
            else:
                self.qubits[self.active_qubit].evolve(qutip.sigmaz(), inverse=True)
        elif key == "w":
            if self.active_qubit == -1:
                self.evolve(self.spin_operators()["Z"], inverse=True)
                self.update()
            else:
                self.qubits[self.active_qubit].evolve(qutip.sigmaz(), inverse=False)
        elif key == "z":
            if self.active_qubit == -1:
                self.evolve(self.spin_operators()["Y"], inverse=True)
                self.update()
            else:
                self.qubits[self.active_qubit].evolve(qutip.sigmay(), inverse=True)
        elif key == "x":
            if self.active_qubit == -1:
                self.evolve(self.spin_operators()["Y"], inverse=True)
                self.update()
            else:
                self.qubits[self.active_qubit].evolve(qutip.sigmay(), inverse=False)
        elif key == "i":
            if self.evolution_on:
                self.evolution_on = False
            else:
                self.evolution_on = True
        elif key == "o":
            self.state = qutip.rand_ket(2**self.n_qubits)
            self.update()
        elif key == "p":
            self.energy = qutip.rand_herm(2**self.n_qubits)
            self.update()
        elif key == "q":
            self.done = True

    def display(self):
        print(crayons.red(    "\t               +Z             "))
        print(crayons.green(  "\t               w       +Y     "))
        print(crayons.blue(   "\t      q quit   |     x        "))
        print(crayons.yellow( "\t               |   /          "))
        print(crayons.magenta("\t               | /            "))
        print(crayons.cyan(   "\t -X a  ________*________ d +X "))
        print(crayons.magenta("\t             / |              "))
        print(crayons.yellow( "\t           /   |  i evolve on/off"))
        print(crayons.blue(   "\t        -Y     |  o new state"))
        print(crayons.green(  "\t      z        s  p new energy"))
        print(crayons.red(    "\t              -Z              "))
        print(crayons.blue("\tqubit # to activate qubit"))
        print(crayons.blue("\t` to activate whole"))
        print(crayons.magenta("qubit #%d active!" % self.active_qubit))
        vp.scene.bind('keydown', self.keyboard)
        while not self.done: 
            if self.evolution_on:
                self.evolve()                  
            self.display_visuals()
            if self.camera_follows_total_spin:
                vp.scene.forward = 2*vp.vector(*self.spin_axis())
        self.invisible()
        vp.scene.unbind('keydown', self.keyboard)

    def update(self):
        state_copy = self.state.copy()
        state_copy.dims = [[2**self.n_qubits], [1]]
        self.vsphere.state = state_copy
        self.state.dims = [[2]*self.n_qubits, [1]*self.n_qubits]
        for i in range(self.n_qubits):
            self.qubits[i].update(self.state.ptrace(i))

    def display_visuals(self):
        self.vsphere.display_visuals()
        for i in range(self.n_qubits):
            self.qubits[i].display_visuals()

    def spin_operators(self):
        n = 2**self.n_qubits
        spin = (n-1.)/2.
        return {"X": qutip.jmat(spin, "x"),\
                "Y": qutip.jmat(spin, "y"),\
                "Z": qutip.jmat(spin, "z"),\
                "+": qutip.jmat(spin, "+"),\
                "-": qutip.jmat(spin, "-")}

    def spin_axis(self):
        n = 2**self.n_qubits
        spin = (n-1.)/2.
        X, Y, Z = qutip.jmat(spin)
        state_copy = self.state.copy()
        state_copy.dims = [[2**self.n_qubits],[1]]
        spin_axis = [qutip.expect(X, state_copy),\
                     qutip.expect(Y, state_copy),\
                     qutip.expect(Z, state_copy)]
        return spin_axis

    def evolve(self, operator=None, inverse=True, dt=0.005):
        if operator == None:
            operator = self.energy
        unitary = (-2*math.pi*im()*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state.dims = [[2**self.n_qubits],[1]]
        self.state = unitary*self.state
        self.update()

    def invisible(self):
        self.vsphere.invisible()
        for qubit in self.qubits:
            qubit.invisible()

##################################################################################################################

def help___():
    for attr in globals().keys():
        if attr.endswith("___"):
            action_name = attr[:-3]
            action = globals().get(action_name+"___")
            doc = action.__doc__
            if doc != None:
                print("\t%s" % (doc))
            else:
                print("\t%s" % (action_name))

def q___():
    """q: quit"""
    print("goodbye!")
    os._exit(0)

def save___(filename):
    """save *filename*"""
    global spheres
    try:
        pickle.dump(spheres, open(filename, "wb"))
        spheres.clear()
    except:
        display_error(message=sys.exc_info()[0])

def load___(filename):
    """load *filename*"""
    global spheres
    try:
        spheres = pickle.load(open(filename, "rb"))
    except:
        display_error(message=sys.exc_info()[0])

def souls___():
    """souls: list of"""
    global spheres
    if len(spheres.souls) == 0:
        display_error(message="no one here!")
    else:
        print(crayons.blue("souls:"))
        for name in spheres.souls.keys():
            print("  . %s" % name)

def questions___():
    """questions: list of"""
    global spheres
    if len(spheres.questions) == 0:
        display_error(message="no questions!")
    else:
        print(crayons.blue("questions:"))
        for i, question in enumerate(spheres.questions):
            print("  %d. %s" % (i, ", ".join(question)))

def create___(what, soul_name=None):
    """create *soul/question* (*soul_name*)"""
    global spheres
    if what == "soul":
        if soul_name == None:
            print(crayons.blue("name?"))
            soul_name = display_inner_prompt()
        if soul_name in spheres.souls.keys():
            display_error(message="already a soul named %s!" % (soul_name))
        else:
            spheres.add_soul(soul_name)
    elif what == "question":
        answers = []
        next_answer = input("\t.")
        while next_answer != "":
            answers.append(next_answer)
            next_answer = input("\t.")
        spheres.add_question(answers)
    else:
        display_error("what's %s?" % what)

def destroy___(what, which):
    """destroy *soul/question* *soul/question-#* """
    global spheres
    if what == "soul":
        if does_soul_exist(which):
            spheres.remove_soul(which)
    elif what == "question":
        if does_question_exist(which):
            spheres.remove_question(int(which))
    else:
        display_error("what's %s?" % what)

def clear___(what):
    """clear *souls/questions/all*"""
    global spheres
    if what == "souls":
        spheres.clear_souls()
    elif what == "questions":
        spheres.clear_questions()
    elif what == "all":
        spheres.clear()
    else:
        display_error("what's %s?" % what)

def soul___(soul_name):
    """soul *name*"""
    global spheres
    if does_soul_exist(soul_name):
        soul = spheres.souls[soul_name]
        rep = crayons.red("**************************************************************\n")
        rep += crayons.magenta("%s:\n" % soul.name)
        rep += crayons.green("  vocabulary:\n")
        for i in range(len(soul.vocabulary)):
            v = soul.vocabulary[i]
            rep += crayons.blue("    %d.%s\n      " % (i, v))
            for e in soul.symbol_basis[v].full().T[0].tolist():
                rep += '[{0.real:.2f}+{0.imag:.2f}i] '.format(e)
            rep += "\n"
        rep += crayons.yellow("  concordance_matrix:\n")
        rep += str(soul.concordance_matrix) + "\n"
        rep += crayons.cyan("  questions:\n")
        for i in range(len(soul.questions)):
            question, answer = soul.questions[i]
            rep += crayons.red("    %d.'%s'\n    " % (i, ", ".join(question)))
            for e in answer.full().T[0].tolist():
                rep += '[{0.real:.2f}+{0.imag:.2f}i] '.format(e) 
            rep += "\n"
            probabilities = soul.question_to_probabilities(i)
            for probability in probabilities:
                answer, prob = probability
                rep += crayons.green("\t\t.%s:" % answer)+crayons.blue(" %.3f%%\n" % prob)
        rep += crayons.magenta("  orderings:\n")
        for question in soul.ordering:
            rep += "\t.%s\n" % (display_question(soul, question))
        rep += crayons.yellow("  state:\n")
        rep += str(soul.state) + "\n"
        rep += crayons.blue("**************************************************************")
        print(rep)

def ask___(soul_name, question_index):
    """ask *name* *question-#*"""
    global spheres
    if does_soul_exist(soul_name):
        if does_question_exist(question_index):
            question_index = int(question_index)
            soul = spheres.souls[soul_name]
            question = spheres.questions[question_index]
            question_space = soul.prepare_for_question(question)
            decision_sphere = MajoranaDecisionSphere(question, question_space)
            answer = decision_sphere.decide()
            if answer:
                soul.add_question(question, answer)

def repose___(soul_name, soul_question_index):
    """repose *name* *soul-question-#*"""
    global spheres
    if does_soul_exist(soul_name):
        if does_question_index_exist_for_soul(soul_name, soul_question_index):
            soul_question_index = int(soul_question_index)
            soul = spheres.souls[soul_name]
            question, answer = soul.questions[soul_question_index]
            question_space = soul.construct_question_space(question)
            decision_sphere = MajoranaDecisionSphere(question, question_space, initial_answer=answer)
            new_answer = decision_sphere.decide()
            if new_answer:
                soul.change_answer(soul_question_index, answer)

def vocab___(soul_name):
    """vocab *name*"""
    global spheres
    if does_soul_exist(soul_name):
        soul = spheres.souls[soul_name]
        if len(soul.vocabulary) == 0:
            display_error(message="%s has no words!" % soul_name)
        else:
            vocab_sphere = VocabularySphere(soul)
            vocab_sphere.display()

def order___(soul_name):
    """order *name*"""
    global spheres
    if does_soul_exist(soul_name):
        soul = spheres.souls[soul_name]
        if len(soul.ordering) != 1:
            ordering_sphere = OrderingSphere(soul)
            soul.ordering = ordering_sphere.order()
            if len(soul.ordering) == 1:
                soul.state = soul.construct_state()
        else:
            display_error(message="%s in order!" % soul_name)
            soul.state = soul.construct_state()

def state___(soul_name):
    """state *soul*"""
    global spheres
    if does_soul_exist(soul_name):
        soul = spheres.souls[soul_name]
        if soul.state != None:
            state_sphere = StateSphere(soul)
            state_sphere.display()
        else:
            display_error(message="must order %s! no state yet." % soul_name)

def qubits___(n_qubits, show_stars=False, 
                        show_total_spin=False,
                        camera_follows_total_spin=False):
    """qubits *n-qubits* (*show-stars* *show-total-spin* *camera-follows-total-spin*)"""
    if n_qubits.isdigit():
        n_qubits = int(n_qubits)
        if n_qubits < 1:
            display_error(message="at least 1 qubit please!")
        else:
            if isinstance(show_stars, str):
                show_stars = str_to_bool(show_stars)
            if isinstance(show_total_spin, str):
                show_total_spin = str_to_bool(show_total_spin)
            if isinstance(camera_follows_total_spin, str):
                camera_follows_total_spin = str_to_bool(camera_follows_total_spin)
            multi_qubit = MultiQubitSphere(n_qubits, show_stars=show_stars,\
                                                     show_total_spin=show_total_spin,\
                                                     camera_follows_total_spin=camera_follows_total_spin)
            multi_qubit.display()
    else:
        display_error(message="must be a #!")

def cmd_loop():
    global spheres
    display_startup_message()
    while True:
        vp.rate(100)
        commands = display_loop_prompt()
        if len(commands) > 0:
            action = globals().get(commands[0]+"___")
            if action != None:
                args = inspect.getargspec(action).args
                defaults = inspect.getargspec(action).defaults
                if defaults == None:
                    if len(commands)-1 == len(args):
                        action(*commands[1:])
                    else:
                        display_error(message=action.__doc__)
                else:
                    required = len(args)-len(defaults)
                    if len(commands)-1 >= required and len(commands)-1 <= len(args): 
                        action(*commands[1:])
                    else:
                        display_error(message=action.__doc__)
            else:
                display_error()

##################################################################################################################

song = pyglet.media.load("sphurs.mp3")
looper = pyglet.media.SourceGroup(song.audio_format, None)
looper.loop = True
looper.queue(song)
player = pyglet.media.Player()
player.queue(looper)
player.play()

##################################################################################################################

cmd_loop()

##################################################################################################################