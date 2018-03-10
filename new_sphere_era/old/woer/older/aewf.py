
# R4 -> C2
def complexify(r4):
	x, y, z, t = r4.T[0]
    return np.array([[complex(x, y)],\
                     [complex(z, t)]])

# C2 -> R4
def uncomplexify(c2):
	alpha, beta = c2.T[0]
	x = alpha.real
	y = alpha.imag
	z = beta.real
	w = beta.imag
	return np.array([[x],\
					 [y],\
					 [z],\
					 [t]])

# C2 -> R3
def reverse_stereographic_projection(c2):
    alpha, beta = c2.T[0]
    x = 2*(np.conjugate(alpha)*beta).real
    y = 2*(np.conjugate(alpha)*beta).imag
    z = (alpha*np.conjugate(alpha)-beta*np.conjugate(beta))
    return np.inner(c2,np.conjugate(c2).T)*(np.array([[x],\
                     								  [y],\
                     								  [z]]))

# R3 -> C2
def stereographic_projection(r3):
	coordinates = stereographic_projectionAll(r3).T[0]

