import os
import sys
import pickle
import random
import crayons
import threading
import qutip as qt
import numpy as np
import vpython as vp
import math
import qutip
import cmath
import functools
import operator
import sympy
import mpmath
import scipy
import itertools

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
        return [complex(0,0) for i in range(len(polynomial)-1)]
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
    def __init__(self, n, state=None,\
                          center=vp.vector(0,0,0),\
                          radius=1,\
                          color=vp.color.blue,\
                          star_color=vp.color.white,
                          tag=None):
        self.n = n
        if state == None:
            self.state = qutip.rand_ket(n)
        else:
            self.state = state
        self.center = center
        self.radius = radius
        if color == None:
            self.color = vp.vector(random.random(),\
                                   random.random(),\
                                   random.random())
        else:
            self.color = color
        self.star_color = star_color
        self.tag = tag

        self.vsphere = vp.sphere(pos=self.center,\
                                 radius=self.radius,\
                                 color=self.color,\
                                 opacity=0.3)
        self.varrow = vp.arrow(pos=self.center,\
                                   color=self.color,\
                                   shaftwidth=0.05,\
                                   emissive=True)
       	self.text = vp.text(text=self.tag,\
       						align='center',\
       						color=self.color,\
       						height=0.2)
        self.vstars = [vp.sphere(radius=0.1*self.radius,\
        						 opacity=0.5,\
                                 emissive=True) for i in range(self.n-1)]

    def spin_axis(self):
        spin = (self.n-1.)/2.
        X, Y, Z = qutip.jmat(spin)
        self.state.dims = [[self.n], [1]]
        spin_axis = [qt.expect(X, self.state),\
                     qt.expect(Y, self.state),\
                     qt.expect(Z, self.state),\
                     qt.expect(qt.identity(self.n), self.state)]
        spin_axis = normalize(np.array(spin_axis))
        return spin_axis[:-1].tolist()

    def visualize(self):
        self.vsphere.pos = self.center
        self.vsphere.radius = self.radius
        self.vsphere.color = self.color
        spin_axis = self.spin_axis()
        self.varrow.pos = self.center
        self.varrow.axis = vp.vector(*spin_axis)*self.radius
        self.varrow.color = self.color
        self.text.pos = vp.vector(*spin_axis)*self.radius + self.center
        self.state.dims = [[self.n],[1]]
        stars_xyz = q_SurfaceXYZ(self.state)
        for i in range(self.n-1):
            self.vstars[i].pos = vp.vector(*stars_xyz[i])*self.radius + self.center
            self.vstars[i].color = self.star_color

    def __del__(self):
    	self.vsphere.visible = False
    	self.varrow.visible = False
    	self.text.visible = False
    	for vstar in self.vstars:
    		vstar.visible = False

##################################################################################################################

class Soul:
	def __init__(self, name):
		self.name = name

		self.color = vp.vector(*np.random.rand(3))
		self.radius = random.random()/20
		self.vsphere = vp.sphere(color=self.color,\
								 radius=self.radius)

		self.vocabulary = []
		self.concordance_matrix = None
		self.left_basis, self.diag, self.right_basis = None, None, None
		self.symbol_basis = {}

		self.questions = []

	def ask(self, question):
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
		phases = np.roots([1] + [0]*(len(question)-1) + [-1])
		for i in range(len(question)):
			word = question[i]
			for j in range(len(question)):
				if j != i:
					other_word = question[j]
					self.concordance_matrix[self.vocabulary.index(word)]\
										   [self.vocabulary.index(other_word)]\
												+= phases[j]/phases[i]
		self.left_basis, self.diag, self.right_basis = np.linalg.svd(self.concordance_matrix)
		for i in range(len(self.vocabulary)):
			 self.symbol_basis[self.vocabulary[i]] = qt.tensor(qt.Qobj(self.left_basis[i]), qt.Qobj(self.right_basis[i]))*self.diag[i]**2
		
		answer_spheres = []
		answer_colors = [vp.vector(*np.random.rand(3)) for i in range(len(question))]
		for i in range(len(question)):
			answer_vector = self.symbol_basis[question[i]].copy()
			d = len(self.vocabulary)**2
			answer_vector.dims = [[d],[1]]
			answer_sphere = MajoranaSphere(d, answer_vector,\
								color=answer_colors[i],\
								star_color=answer_colors[i],
								tag=question[i])
			answer_sphere.visualize()
			answer_spheres.append(answer_sphere)

		for i in range(len(question)):
			if i != len(question)-1:
				print(("\t%d" % (i))+crayons.red('.')+("%s" % (question[i])))
			else:
				print(("\t%d" % (i))+crayons.red('.')+("%s" % (question[i]))+crayons.magenta("?"))
		done = False
		while not done:
			answer = input(crayons.green('.')+crayons.magenta('.')+crayons.blue('.'))
			if answer == "":
				print("refusing question.")
				done = True
			elif answer.isdigit():
				answer_index = int(answer)
				if answer_index >= 0 and answer_index < len(question):
					self.questions.append([question, qt.basis(len(question), answer_index)])
					done = True
				else:
					print("not enough options!")
			else:
				print("use answer #!")
			
	def __getstate__(self):
		stuff = self.__dict__
		stuff['vsphere'].visible = False
		del stuff['vsphere']
		return stuff

	def __setstate__(self, stuff):
		self.__dict__ = stuff
		self.vsphere = vp.sphere(color=self.color,\
							     radius=self.radius)

	def __del__(self):
		stuff = self.__dict__
		if 'vsphere' in stuff:
			stuff['vsphere'].visible = False

##################################################################################################################

souls = []
questions = []

if len(sys.argv) == 2:
	filename = sys.argv[1]
	try:
		souls, questions = pickle.load(open(filename, "rb"))
	except:
		print("?: %s" % sys.exc_info()[0])

##################################################################################################################

def soul_index_for_name(name):
	global souls
	for i in range(len(souls)):
		if souls[i].name == name:
			return i
	return -1

def cmd_loop():
	global souls
	global questions
	os.system('clear')
	spheres = crayons.red('s')+crayons.white('p')+crayons.yellow('h')+crayons.green('e')+crayons.cyan('r')+crayons.blue('e')+crayons.magenta('s')
	print("welcome to %s" % spheres)
	while True:
		cmd = input(crayons.cyan(':')+crayons.blue('>')+" ")
		cmds = cmd.lower().split()
		n = len(cmds)
		if n > 0:
			if cmds[0] == "q":
				print("goodbye!")
				os._exit(0)
			elif cmds[0] == "?":
				print("\tq: quit")
				print("\tsave *filename*")
				print("\tload *filename*")
				print("\tsouls: list of")
				print("\tcreate soul *name*")
				print("\tdestroy soul *name*")
				print("\tclear souls")
				print("\tquestions: list of")
				print("\tcreate question")
				print("\tdestroy question *question-#*")
				print("\tclear questions")
				print("\task *soul* *question-#*")
			elif cmds[0] == "save" and n == 2:
				filename = cmds[1]
				try:
					pickle.dump([souls, questions], open(filename, "wb"))
					souls, questions = [], []
				except:
					print("?: %s" % sys.exc_info()[0])
			elif cmds[0] == "load" and n == 2:
				filename = cmds[1]
				try:
					souls, questions = pickle.load(open(filename, "rb"))
				except:
					print("?: %s" % sys.exc_info()[0])
			elif cmds[0] == "clear" and n == 2:
				if cmds[1] == "souls":
					souls = []
				elif cmds[1] == "questions":
					questions = []
			elif cmds[0] == "souls":
				if len(souls) == 0:
					print("no one here!")
				else:
					print("souls:")
					for i in range(len(souls)):
						print("  %s" % souls[i].name)
			elif cmds[0] == "questions":
				if len(questions) == 0:
					print("no questions!")
				else:
					print("questions:")
					for i in range(len(questions)):
						print(("\t%d. " % (i)) + ", ".join(questions[i]))
			elif cmds[0] == "create":
				if n == 2 and cmds[1] == "question":
					answers = []
					next_answer = input("\t.")
					while next_answer != "":
						answers.append(next_answer)
						next_answer = input("\t.")
					questions.append(answers)
					print("created question '%s.'" % (", ".join(questions[-1])))
				elif n == 3 and cmds[1] == "soul":
					found = False
					for soul in souls:
						if soul.name == cmds[2]:
							print("already a soul named %s!" % cmds[2])
							found = True
					if not found:
						souls.append(Soul(cmds[2]))
						print("created soul %s." % cmds[2])
			elif cmds[0] == "destroy" and n == 3:
				if cmds[1] == "soul":
					death = None
					for i in range(len(souls)):
						if souls[i].name == cmds[2]:
							death = souls[i]
					if death == None:
						print("%s not here!" % cmds[2])
					else:
						souls.remove(death)
						del death
						print("destroyed soul %s." % cmds[2])
				elif cmds[1] == "question":
					if cmds[2].isdigit():
						i = int(cmds[2])
						if i >= 0 and i < len(questions):
							print("destroyed question '%s.'" % (", ".join(questions[i])))
							del questions[i]
			elif cmds[0] == "ask" and n == 3:
				soul_index = soul_index_for_name(cmds[1])
				if soul_index != -1:
					if cmds[2].isdigit():
						question_index = int(cmds[2])
						if question_index >= 0 and question_index < len(questions):
							souls[soul_index].ask(questions[question_index])
						else:
							print("no question #%d!" % question_index)
					else:
						print("use question #!")
				else:
					print("no soul named %s!" % cmds[1])
			else:
				print("?")

self.cmd_thread = threading.Thread(target=cmd_loop())
self.cmd_thread.start()

##################################################################################################################

while True:
	vp.rate(100)