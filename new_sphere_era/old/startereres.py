class Spheres:
	def __init__(self, cmd_server=TextCmdServer(),\
					   v_server=VisualCmdServer()):
		self.cmd_server = cmd_server
		self.v_server = v_server

		self.souls = {}
		self.questions = []

	def start(self):
		self.cmd_server.start()
		self.v_server.start()

	def cycle(self):
		self.v_server.cycle()

##################################################################################################################

class VisualCmdServer():
	def __init__(self, update_rate=100):
		self.update_rate = update_rate

	def start(self):
		pass

	def cycle(self):
		vp.rate(self.update_rate)

##################################################################################################################

spheres = Spheres()
spheres.start()
