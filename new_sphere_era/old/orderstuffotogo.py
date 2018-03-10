	def order__(self, soul_name):
		"""order *name*"""
		if self.does_soul_exist(soul_name):
			self.sphere.souls[soul_name].order()

			def symmeterize(pieces, labels=None):
    if labels == None:
        labels = list(range(len(pieces)))
    n = len(pieces)
    unique_labels = list(set(labels))
    label_counts = [0 for i in range(len(unique_labels))]
    label_permutations = itertools.permutations(labels, n)
    for permutation in label_permutations:
        for i in range(len(unique_labels)):
            label_counts[i] += list(permutation).count(unique_labels[i])
    normalization = 1./math.sqrt(functools.reduce(operator.mul, [math.factorial(count) for count in label_counts], 1)/math.factorial(n))    
    #normalization = 1./math.sqrt(math.factorial(n))
    permutations = itertools.permutations(pieces, n)
    tensor_sum = sum([qutip.tensor(list(permutation)) for permutation in permutations])
    return normalization*tensor_sum

def perm_parity(lst):
    '''\
    Given a permutation of the digits 0..N in order as a list, 
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
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
    permutations = itertools.permutations(pieces, n)
    tensor_sum = sum([qutip.tensor(perm_parity(permutation)*list(permutation)) for permutation in permutations])
    return normalization*tensor_sum

def direct_sum(a, b):
    return qt.Qobj(scipy.linalg.block_diag(a.full(),b.full()))


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
	            question, answer = self.questions[i]
	            rep += crayons.red("    %d.'%s'\n      " % (i, ", ".join(question)))
	            #for e in answer.full().T[0].tolist():
	            #    rep += '[{0.real:.2f}+{0.imag:.2f}i] '.format(e)
	            probs = soul.question_to_probabilities(i)
	            #print(probs)
	            for p in probs:
	                rep += "\t%s: %.6f%%\n" % (p[0], p[1])
	        rep += crayons.magenta("  orderings:\n")
	        for e in soul.ordering:
	            rep += "\t.%s\n" % (soul.combo_question_string(e))
	        rep += crayons.yellow("  state:\n")
	        rep += str(soul.state) + "\n"
	        rep += crayons.blue("**************************************************************")
	        print(rep)
