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

spheres = Spheres()

##################################################################################################################

vis_new_key = False
vcurrent_key = None
def vkeyboard(event):
    global vis_new_key
    global vcurrent_key
    vis_new_key = True
    vcurrent_key = event.key
vp.scene.bind('keydown', vkeyboard)

def get_vkey():
    global vis_new_key
    global vcurrent_key
    if vis_new_key:
        vis_new_key = False
        return vcurrent_key
    else:
        return ''

vis_new_click = False
vcurrent_click = None
def vmouse(event):
    global vis_new_click
    global vcurrent_click
    vis_new_click = True
    vcurrent_click = vp.scene.mouse.pick
vp.scene.bind('click', vmouse)

def get_vclick():
    global vis_new_click
    global vcurrent_click
    if vis_new_click:
        vis_new_click = False
        return vcurrent_click
    else:
        return None

##################################################################################################################

def im():
    return complex(0, 1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

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

    def change_answer(self, question_index, answer):
        self.questions[question_index] = answer

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
                       center=vp.vector(0,0,0),\
                       radius=1,\
                       tag=None,\
                       sphere_color=vp.color.blue,\
                       star_color=vp.color.white,\
                       arrow_color=None,\
                       show_sphere=True,\
                       show_stars=True,\
                       show_arrow=True,\
                       show_tag=True,\
                       show_tag_on_stars=False):
        self.n = state.shape[0]
        self.state = state
        self.center = center
        self.radius = radius
        self.tag = tag
        self.sphere_color = sphere_color
        self.star_color = star_color
        self.arrow_color = self.sphere_color if arrow_color == None else arrow_color
        self.show_sphere = show_sphere
        self.show_stars = show_stars
        self.show_arrow = show_arrow
        self.show_tag = show_tag
        self.show_tag_on_stars= show_tag_on_stars

        self.vsphere = vp.sphere(visible=False) 
        self.varrow = vp.arrow(visible=False)
        self.vstars = [vp.sphere(visible=False) for i in range(self.n-1)] 
        self.vtag = vp.text(text="e", visible=False) 
        self.vtags = [vp.text(text="e", visible=False) for i in range(self.n-1)]
    
    def display_visuals(self):
        if self.show_sphere:
            self.vsphere.pos = self.center
            self.vsphere.radius = self.radius
            self.vsphere.color = self.sphere_color
            self.vsphere.opacity = 0.3
            self.vsphere.visible = True
        else:
            self.vsphere.visible = False
        self.state.dims = [[self.n],[1]]
        stars_xyz = q_SurfaceXYZ(self.state) if (self.show_stars or (self.tag and self.show_tag_on_stars)) else None
        if self.show_stars:
            for i, vstar in enumerate(self.vstars):
                vstar.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vstar.radius = 0.1*self.radius
                vstar.color = self.star_color
                vstar.opacity = 0.5
                vstar.emissive = True
                vstar.visible = True
        else:
            for vstar in self.vstars:
                vstar.visible = False
        spin_axis = self.spin_axis() if (self.show_arrow or (self.tag and self.show_tag)) else None
        if self.show_arrow:
            self.varrow.pos = self.center
            self.varrow.axis = self.radius*vp.vector(*spin_axis)
            self.varrow.color = self.arrow_color
            self.varrow.shaftwidth = 0.05
            self.varrow.emissive = True
            self.varrow.visible = True
        else:
            self.varrow.visible = False
        if self.tag and self.show_tag:
            self.vtag.text = self.tag
            self.vtag.pos = self.radius*vp.vector(*spin_axis) + self.center
            self.vtag.align = "center"
            self.vtag.color = self.arrow_color
            self.vtag.height = 0.2*self.radius
            self.vtag.visible = True
        else:
            self.vtag.visible = False
        if self.tag and self.show_tag_on_stars:
            for i, vtag in enumerate(self.vtags):
                vtag.text = self.tag
                vtag.pos = self.radius*vp.vector(*stars_xyz[i]) + self.center
                vtag.align = "center"
                vtag.color = self.arrow_color
                vtag.height = 0.2*self.radius
                vtag.visible = True
        else:
            for vtag in self.vtags:
                vtag.visible = False

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
    def __init__(self, sphere_command, question, question_space, initial_answer=None):
        self.sphere_command = sphere_command
        self.question = question
        self.question_space = question_space
        self.n_choices = len(question)
        self.initial_answer = qt.Qobj(sum([question_space[i] for i in range(self.n_choices)]))\
                                if initial_answer == None else initial_answer

        self.decider_sphere = MajoranaSphere(self.initial_answer)
        self.answer_spheres = []
        self.answer_colors = [vp.vector(*np.random.rand(3)) for i in range(self.n_choices)]
        for i in range(self.n_choices):
            answer_vector = qt.Qobj(question_space[i])
            answer_vector.dims = [[self.n_choices],[1]]
            self.answer_spheres.append(MajoranaSphere(answer_vector,\
                                                      sphere_color=self.answer_colors[i],\
                                                      star_color=self.answer_colors[i],\
                                                      tag=self.question[i]))

    def decide(self):
        answer = self.display_vprompt()
        if not answer:
            answer = self.display_txtprompt()
        del self.decider_sphere
        for answer_sphere in self.answer_sphere:
            del answer_sphere
        return answer

    def display_vprompt(self):
        self.decider_sphere.display_visuals()
        for answer_sphere in self.answer_spheres:
            answer_sphere.display_visuals()
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
    
    def id(self):
        spin_ops = self.decider_sphere.spin_operators()
        key = get_vkey()
        while key != "q" and key != "x":
            if key == "a":
                self.decider_sphere.apply(spin_ops['X'], inverse=False)
            elif key == "d":
                self.decider_sphere.apply(spin_ops['X'], inverse=False)
            elif key == "s":
                self.decider_sphere.apply(spin_ops['Z'], inverse=True)
            elif key == "w":
                self.decider_sphere.apply(spin_ops['Z'], inverse=False)
            elif key == "z":
                self.decider_sphere.apply(spin_ops['Y'], inverse=True)
            elif key == "e":    
                self.decider_sphere.apply(spin_ops['Y'], inverse=False)
            self.decider_sphere.display_visuals()
            key = get_vkey()
        if key == "q":
            return None
        else:
            return self.decider_sphere.state

    def display_txtprompt(self):
        for i in range(self.n_choices):
            if i != self.n_choices-1:
                print(("\t%d"+crayons.red('.')+"%s") % (i, question[i]))
            else:
                print(("\t%d"+crayons.red('.')+"%s"+crayons.magenta("?")) % (i, question[i])) 
        done = False
        answer = None
        while not done:
            answer = self.sphere_command.display_inner_prompt()
            if answer == "":
                print("refusing question.")
                done = True
            elif answer.isdigit():
                answer_index = int(answer)
                if answer_index >= 0 and answer_index < self.n_choices:
                    return qt.Qobj(self.question_space[answer_index])
                    done = True
                else:
                    self.sphere_command.display_error(message="not enough options for you!")
            else:
                self.sphere_command.display_error(message="use answer #!")
        return answer

##################################################################################################################

class VocabularySphere:
    def __init__(self, sphere_command, soul):
        self.sphere_command = sphere_command
        self.soul = soul
        self.word_colors = [vp.vector(*np.random.rand(3)) for v in self.soul.vocabulary]
        self.show_stars = True if len(self.soul.vocabulary) < 7 else False
        self.word_spheres = [MajoranaSphere(self.soul.symbol_basis[self.soul.vocabulary[i]],\
                                             color=self.word_colors[i],\
                                              star_color=self.word_colors[i],\
                                              tag=self.soul.vocabulary[i],\
                                              show_stars=self.show_stars)\
                                                 for i in range(len(self.soul.vocabulary))]

    def display(self):
        self.sphere_command.vycle()                                           
        print(crayons.magenta("q to quit..."))                    
        for word_sphere in self.word_spheres:
            word_sphere.visualize()
        key = self.sphere_command.get_vkey()
        while key != "q":
            key = self.sphere_command.get_vkey()
        for word_sphere in self.word_spheres:
            del word_sphere

##################################################################################################################

class SphereCommand:
    def __init__(self):
        self.display_startup_message()

    def cmd_loop(self):
        commands = self.display_loop_prompt()
        if len(commands) > 0:
            action = getattr(self, commands[0]+"___", None)
            if action:
                if len(inspect.getargspec(action).args)-1 == len(commands)-1:
                    if len(commands) > 1:
                        action(*commands[1:])
                    else:
                        action()
                else:
                    self.display_error(message=action.__doc__)
            else:
                self.display_error()

    ##############################################################################################################

    def help___(self):
        for attr in dir(self):
            if attr.endswith("___"):
                action = attr[:-3]
                doc = getattr(self, attr).__doc__
                if doc:
                    print("\t%s" % (doc))
                else:
                    print("\t%s" % (action))

    def q___(self):
        """q: quit"""
        print("goodbye!")
        os._exit(0)

    def save___(self, filename):
        global spheres
        """save *filename*"""
        try:
            pickle.dump(spheres, open(filename, "wb"))
            spheres.clear()
        except:
            self.display_error(message=sys.exc_info()[0])

    def load___(self, filename):
        """load *filename*"""
        try:
            self.spheres = pickle.load(open(filename, "rb"))
        except:
            self.display_error(message=sys.exc_info()[0])

    def souls___(self):
        """souls: list of"""
        if len(self.spheres.souls) == 0:
            self.display_error(message="no one here!")
        else:
            print(crayons.blue("souls:"))
            for name in self.spheres.souls.keys():
                print("  . %s" % name)

    def questions___(self):
        """questions: list of"""
        if len(self.spheres.questions) == 0:
            self.display_error(message="no questions!")
        else:
            print(crayons.blue("questions:"))
            for i, question in enumerate(self.spheres.questions):
                print("  %d. %s" % (i, ", ".join(question)))

    def create___(self, what):
        """create *soul/question*"""
        if what == "soul":
            print(crayons.blue("name?"))
            soul_name = self.display_inner_prompt()
            while soul_name in self.spheres.souls.keys():
                self.display_error(message="already a soul named %s!" % (soul_name))
                soul_name = self.display_inner_prompt()
            self.spheres.add_soul(soul_name)
        else:
            if what == "question":
                answers = []
                next_answer = input("\t.")
                while next_answer != "":
                    answers.append(next_answer)
                    next_answer = input("\t.")
                self.spheres.add_question(answers)

    def destroy___(self, what, which):
        """destroy *soul/question* *soul/question-#* """
        if what == "soul":
            if self.does_soul_exist(which):
                self.spheres.remove_soul(which)
        elif what == "question":
            if self.does_question_exist(which):
                self.spheres.remove_question(int(which))

    def clear___(self, what):
        """clear *souls/questions/all*"""
        if what == "souls":
            self.spheres.clear_souls()
        elif what == "questions":
            self.spheres.clear_questions()
        elif what == "all":
            self.spheres.clear()

    def soul___(self, soul_name):
        """soul *name*"""
        if self.does_soul_exist(soul_name):
            soul = self.spheres.souls[soul_name]
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
                rep += crayons.red("    %d.'%s'\n      " % (i, ", ".join(question)))
                for e in answer.full().T[0].tolist():
                    rep += '[{0.real:.2f}+{0.imag:.2f}i] '.format(e)            
            rep += crayons.blue("**************************************************************")
            print(rep)

    def ask___(self, soul_name, question_index):
        """ask *name* *question-#*"""
        if self.does_soul_exist(soul_name):
            if self.does_question_exist(question_index):
                question_index = int(question_index)
                soul = self.spheres.souls[soul_name]
                question = self.spheres.questions[question_index]
                question_space = soul.prepare_for_question(question)
                decision_sphere = MajoranaDecisionSphere(self, question, question_space)
                answer = decision_sphere.decide()
                if answer:
                    soul.add_question(question, answer)

    def repose___(self, soul_name, soul_question_index):
        """repose *name* *soul-question-#*"""
        if self.does_soul_exist(soul_name):
            if self.does_question_exist_for_soul(soul_question_index):
                soul_question_index = int(soul_question_index)
                soul = self.spheres.soul[soul_name]
                question, answer = soul.questions[soul_question_index]
                question_space = soul.construct_question_space(question)
                decision_sphere = MajoranaDecisionSphere(self, question, question_space, initial_answer=answer)
                new_answer = decision_sphere.decide()
                if new_answer:
                    soul.change_answer(soul_question_index, answer)

    def vocab___(self, soul_name):
        """vocab *name*"""
        if self.does_soul_exist(soul_name):
            soul = self.sphere.souls[soul_name]
            if len(soul.vocabulary) == 0:
                self.display_error(message="%s has no words!" % soul_name)
            else:
                vocab_sphere = VocabularySphere(self, soul)
                vocab_sphere.display()

    ##############################################################################################################

    def display_startup_message(self):
        os.system("clear")
        print("welcome to %s" % \
                crayons.red("s")+\
                crayons.white("p")+\
                crayons.yellow("h")+\
                crayons.green("e")+\
                crayons.cyan("r")+\
                crayons.blue("e")+\
                crayons.magenta("s"))

    def display_loop_prompt(self):
        return input(crayons.cyan(":")+crayons.blue(">")+" ").lower().split()

    def display_inner_prompt(self):
        return input(crayons.green('.')+crayons.magenta('.')+crayons.blue('.'))

    def display_error(self, message=None):
        if message:
            print(crayons.red("?: %s" % message))
        else:
            print(crayons.red("?"))

    ##############################################################################################################
    
    def does_soul_exist(self, name):
        if name in self.spheres.souls.keys():
            return True
        else:
            self.display_error(message="no soul named %s!" % (name))
            return False

    def does_question_exist(self, question_index):
        if question_index.isdigit():
            question_index = int(question_index)
            if question_index >= 0 and question_index < len(self.spheres.questions):
                return True
            else:
                self.display_error(message="no question #%d!" % (question_index))
                return False
        else:
            self.display_error(message="use question #!")
            return False

    def does_question_exist_for_soul(self, soul_name, soul_question_index):
        if does_soul_exist(soul_name):
            if soul_question_index.isdigit():
                soul_question_index = int(soul_question_index)
                if soul_question_index >= 0 and soul_question_index < len(self.sphere.souls[soul_name].questions):
                    return True
                else:
                    self.display_error(message="no question #%d for %s!" % (soul_question_index, soul_name))
                    return False
            else:
                self.display_error(message="use question # for %s!" % (soul_name))
                return False

    ##############################################################################################################

##################################################################################################################

sphere_commander = SphereCommand(spheres)

vp.rate(100)
while True:
    sphere_commander.cmd_loop()

##################################################################################################################