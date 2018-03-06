import os
import sys
import pickle
import random
import threading
import qutip as qt
import numpy as np
import vpython as vp

##################################################################################################################

class Soul:
	def __init__(self, name):
		self.name = name
		self.vsphere = vp.sphere(color=vp.vector(*np.random.rand(3)),\
								 radius=random.random())

	def __getstate__(self):
		stuff = self.__dict__
		stuff['vsphere'] = {'color': self.vsphere.color,\
							'radius': self.vsphere.radius}
		return stuff

	def __setstate__(self, stuff):
		stuff['vsphere'] = vp.sphere(color=stuff['vsphere']['color'],\
									 radius=stuff['vsphere']['radius'])
		self.__dict__ = stuff

	def __del__(self):
		stuff = self.__dict__
		if not isinstance(stuff['vsphere'], dict):
			self.vsphere.visible = False
			del self.vsphere

##################################################################################################################

souls = []
questions = []

##################################################################################################################

def cmd_loop():
	global souls
	global questions
	os.system('clear')
	print("welcome to spheres")
	while True:
		cmd = input(":> ")
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
				print("\tdestroy question #")
				print("\tclear questions")
			elif cmds[0] == "save" and n == 2:
				filename = cmds[1]
				try:
					pickle.dump([souls, questions], open(filename, "wb"))
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
			else:
				print("?")

self.cmd_thread = threading.Thread(target=cmd_loop())
self.cmd_thread.start()

##################################################################################################################

while True:
	vp.rate(100)