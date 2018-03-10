import numpy as np
import qutip
import operator
import functools
import scipy
import math

##################################################################################################################

def collapser(dims):
    if all(isinstance(d, int) for d in dims):
        return functools.reduce(operator.mul, dims, 1)
    else:
        new_dims = []
        for d in dims:
            if isinstance(d, int):
                new_dims.append(d)
            else:
                new_dims.append(collapser(d))
        return collapser(new_dims)

def collapse(dims):
    new_dims = []
    for d in dims:
        if isinstance(d, int):
            new_dims.append(d)
        else:
            new_dims.append(collapser(d))
    return new_dims

##################################################################################################################


class Sphere:
    def __init__(self, state=None, dimensionality=None, energy=None, parent=None):
        self.state = state
        self.dimensionality = dimensionality
        self.energy = energy
        self.parent = parent
        self.children = []
        self.bear_children()
        self.spin = self.total_spin()
        
    def bear_children(self):
        dim = collapse(self.dimensionality)
        self.state.dims = [dim, dim]
        distinguishable_subspaces = len(dim)
        if distinguishable_subspaces > 1:
            child_states = [self.state.ptrace(i) for i in range(distinguishable_subspaces)]
            if self.children == []:
                for i in range(distinguishable_subspaces):
                    child_dim = None
                    if isinstance(self.dimensionality[i], int):
                        child_dim = [self.dimensionality[i]]
                    else:
                        child_dim = self.dimensionality[i]
                    self.children.append(Sphere(state=child_states[i], dimensionality=child_dim, parent=self))
            else:
                for i in range(distinguishable_subspaces):
                    self.children[i].state = child_states[i]
    
    def evolve(self, dt=0.01):
        dim = collapse(self.dimensionality)
        unitary = qutip.Qobj(scipy.linalg.expm(-2*math.pi*complex(0,1)*self.energy.full()*dt))
        unitary.dims = [dim, dim]
        self.state = unitary*self.state*unitary.dag()

    def cycle(self):
        self.evolve()
        self.bear_children()
        self.spin = self.total_spin()

    def total_spin(self):
        old_dims = list(self.state.dims)
        n = collapser(self.dimensionality)
        self.state.dims = [[n], [n]]
        T = qutip.identity(n)
        X, Y, Z = qutip.jmat((n-1.)/2.)
        t = qutip.expect(T, self.state)
        x = qutip.expect(X, self.state)
        y = qutip.expect(Y, self.state)
        z = qutip.expect(Z, self.state)
        spin = np.array([t, x, y, z])
        magnitude = np.linalg.norm(spin)
        if magnitude != 0:
            spin = spin / magnitude
        self.state.dims = old_dims
        return spin.tolist()

    def center(self):
        if self.parent == None:
            return [0,0,0]
        else:
            parent_center = self.parent.center()
            return [parent_center[i]+self.spin[i+1] for i in range(3)]

    def color(self):
        return self.spin[0]

    def radius(self):
        if self.parent == None:
            return 1
        else:
            return 1./4**self.parent.radius()

    def __repr__(self):
        n = collapser(self.dimensionality)
        s = "{SPHERE(%d):\n\tDIMENSIONALITY: %s\n\tCENTER: %s\n\tCOLOR: %s\n\tRADIUS:\t%s" % (n, self.dimensionality, self.center(), self.color(), self.radius())
        if self.children == []:
            return s + "}\n"
        else:
            s += "\nCHILDREN:\n"
            for child in self.children:
                s += "\t%s\n" % str(child)
            return s + "}"


##################################################################################################################

def random_sphere(dimensionality):
    n = collapser(dimensionality)
    pure_state = qutip.rand_ket(n)
    state = pure_state.ptrace(0)
    energy = qutip.rand_herm(n)
    return Sphere(state=state, dimensionality=dimensionality, energy=energy)

sphere = random_sphere([2, [2,3], [2]])



