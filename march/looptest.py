import itertools

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

x = [0, 1]