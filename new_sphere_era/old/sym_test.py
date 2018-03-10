import itertools
import math
import qutip as qt
import numpy as np
import scipy
import functools
import operator

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

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

pieces = [qt.rand_ket(2), qt.rand_ket(3)]
sym = direct_sum(pieces)
