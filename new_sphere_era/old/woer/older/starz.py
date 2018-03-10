###

def combos(a,b):
    f = math.factorial
    return f(a) / f(b) / f(a-b)

###

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
			self.touched = 1:
		elif self.touched == 1:
			self.touched = 0
		for dependency in self.dependencies:
			other, transformation = dependency["other"], dependency["transformation"]
			if other.touched != self.touched:
				other.plug(transformation(self.value))

	def __str__(self):
		return str(self.value)

###

class Sphere:
	def __init__(self):
		self.dimensionality = Variable()
		self.state = Variable()
		self.vector = Variable()
		self.polynomial = Variable()
    	self.roots = Variable()
    	self.xyzs = Variable()
    	self.angles = Variable()
    	self.qubits = Variable()
    	self.field = Variable()

		def state_to_vector(state):
			return state.full().T[0]
		self.state.tie(self.vector, state_to_vector)

		def vector_to_state(vector):
			return qutip.Qobj(vector)
		self.vector.tie(self.state, vector_to_state)

		def vector_to_polynomial(vector):
			polynomial = vector.tolist()
    		return [(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) * polynomial[i] for i in range(len(polynomial))] 
		self.vector.tie(self.polynomial, vector_to_polynomial)

		def polynomial_to_vector(polynomial):
			coordinates = [polynomial[i]/(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) for i in range(len(polynomial))]
    		return np.array(coordinates)
    	self.polynomial.tie(self.vector, polynomial_to_vector)

    	def polynomial_to_roots(polynomial):
    		try:
        		return [np.conjugate(complex(root)) for root in mpmath.polyroots(polynomial)]
    		except:
        		return [complex(0,0) for i in range(len(polynomial)-1)]
        self.polynomial.tie(self.roots, polynomial_to_roots)

        def roots_to_polynomial(roots):
        	s = sympy.symbols("s")
    		polynomial = sympy.Poly(functools.reduce(lambda a, b: a*b, [s-np.conjugate(root) for root in roots]), domain="CC")
    		return [complex(c) for c in polynomial.coeffs()]
    	self.roots.tie(self.polynomial, roots_to_polynomial)

    	def roots_to_xyzs(roots):
    		def root_to_xyz(root):
			    if root == float('inf'):
			        return [0,0,1]
			    x = root.real
			    y = root.imag
			    return [(2*x)/(1.+(x**2)+(y**2)),\
			            (2*y)/(1.+(x**2)+(y**2)),\
			            (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]
			return [root_to_xyz(root) for root in roots]
		self.roots.tie(self.xyzs, roots_to_xyzs)

		def xyzs_to_roots(xyzs):
			def xyz_to_root(xyz):
				x, y, z = xyz[0], xyz[1], xyz[2]
			    if z == 1:
			        return float('inf') 
			    else:
			        return complex(x/(1-z), y/(1-z))
			return [xyz_to_root(xyz) for xyz in xyzs]
		self.xyzs.tie(self.roots, xyzs_to_roots)


		def angles_to_xyzs(angles):
			def angle_to_xyz(angle):
				pass
			return [angle_to_xyz(angle) for angle in angles]


				def spin_axis(self):
		state = self.state.value
		n = state.shape[0]
    	X, Y, Z = qutip.jmat((n-1.)/2.)
    	return [state.expect(X, state), state.expect(Y, state), state.expect(Z, state)]

    def star_creators(self):
    	pass

    def star_destroyers(self):
    	pass
