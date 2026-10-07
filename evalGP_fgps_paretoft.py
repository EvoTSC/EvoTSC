import random
import hashlib
from collections import defaultdict
from deap import tools, gp
import gp_restrict as gp_restrict


def _dominates(a, b):
    better_or_equal = True
    strictly_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            better_or_equal = False
            break
        if ai > bi:
            strictly_better = True
    return better_or_equal and strictly_better


def _pareto_front(individuals, key):
    front = []
    vecs = [tuple(key(ind)) for ind in individuals]
    for i, ind in enumerate(individuals):
        vi = vecs[i]
        dominated = False
        for j, vj in enumerate(vecs):
            if j == i:
                continue
            if _dominates(vj, vi):
                dominated = True
                break
        if not dominated:
            front.append(ind)
    return front


def selParetoTournament(population, k, tourn_frac=0.15, attr="fold_accs", fallback_to_avg_fitness=True):
    if k <= 0:
        return []

    n = len(population)
    if n == 0:
        return []

    m = int(round(float(tourn_frac) * n))
    m = max(1, min(m, n))

    dim = None
    for ind in population:
        vec = getattr(ind, attr, None)
        if vec is not None and len(vec) > 0:
            dim = len(vec)
            break
    if dim is None:
        dim = 1

    def _key(ind):
        vec = getattr(ind, attr, None)
        if vec is not None and len(vec) == dim:
            return vec
        if fallback_to_avg_fitness and getattr(ind, "fitness", None) is not None and getattr(ind.fitness, "valid", False):
            try:
                v = float(ind.fitness.values[0])
            except Exception:
                v = 0.0
            return (v,) * dim
        return (0.0,) * dim

    chosen = []
    rounds = 0
    max_rounds = 50 * max(1, k)

    while len(chosen) < k and rounds < max_rounds:
        rounds += 1

        subset = random.sample(population, m)
        front = _pareto_front(subset, _key)
        if not front:
            continue

        random.shuffle(front)
        remaining = k - len(chosen)
        take = min(len(front), remaining)
        chosen.extend(front[:take])

    if len(chosen) < k:
        for _ in range(k - len(chosen)):
            chosen.append(random.choice(population))

    return chosen


def pop_compare(ind1, ind2):
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:],1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:],1):
        types2[node.ret].append(idx)
    return types1==types2


def _node_depths(ind):
    depths = [0] * len(ind)
    stack = []
    for i, node in enumerate(ind):
        while stack and stack[-1] == 0:
            stack.pop()
        depths[i] = len(stack)
        if stack:
            stack[-1] -= 1
        arity = getattr(node, "arity", 0)
        if arity and arity > 0:
            stack.append(int(arity))
    return depths


def _subtree_heights(ind):
    subh = [0] * len(ind)
    stack = []
    for i in range(len(ind) - 1, -1, -1):
        node = ind[i]
        arity = getattr(node, "arity", 0)
        if not arity:
            h = 0
        else:
            childs = [stack.pop() for _ in range(int(arity))]
            h = 1 + max(childs) if childs else 1
        subh[i] = h
        stack.append(h)
    return subh


def safe_cx_one_point(ind1, ind2, max_depth):
    if max_depth is None:
        return ind1, ind2

    d1 = _node_depths(ind1)
    d2 = _node_depths(ind2)
    h1 = _subtree_heights(ind1)
    h2 = _subtree_heights(ind2)

    idx1 = defaultdict(list)
    idx2 = defaultdict(list)
    for i, node in enumerate(ind1):
        idx1[getattr(node, "ret", None)].append(i)
    for j, node in enumerate(ind2):
        idx2[getattr(node, "ret", None)].append(j)

    candidates = []
    for t, I in idx1.items():
        J = idx2.get(t, [])
        if not J:
            continue
        for i in I:
            budget_i = int(max_depth) - int(d1[i])
            if budget_i < 0:
                continue
            for j in J:
                if h2[j] <= budget_i and h1[i] <= (int(max_depth) - int(d2[j])):
                    s1 = ind1.searchSubtree(i)
                    s2 = ind2.searchSubtree(j)
                    if list(ind1[s1]) != list(ind2[s2]):
                        candidates.append((s1, s2))

    if not candidates:
        return ind1, ind2

    s1, s2 = random.choice(candidates)
    tmp1 = ind1[s1]
    tmp2 = ind2[s2]
    ind1[s1] = tmp2
    ind2[s2] = tmp1
    return ind1, ind2


def safe_mut_uniform(ind, pset, max_depth, min_=0, max_=None):
    if max_depth is None or len(ind) == 0:
        return (ind,)

    depths = _node_depths(ind)
    idx = random.randrange(len(ind))
    node = ind[idx]
    budget = int(max_depth) - int(depths[idx])
    if budget < 0:
        return (ind,)

    if max_ is None:
        max_here = int(budget)
    else:
        max_here = min(int(max_), int(budget))
    min_here = min(int(min_), int(max_here))
    type_ = getattr(node, "ret", None)

    expr = gp_restrict.genFull(pset=pset, min_=min_here, max_=max_here, type_=type_)
    sl = ind.searchSubtree(idx)
    ind[sl] = expr
    return (ind,)


def varAnd(population, toolbox, cxpb, mutpb, n_retry=5):
    offspring = [toolbox.clone(ind) for ind in population]
    den = (cxpb + mutpb)
    new_cxpb = (cxpb / den) if den > 0 else 0.0

    def _mutate_retry(ind):
        base = toolbox.clone(ind)
        for _ in range(n_retry):
            child, = toolbox.mutate(toolbox.clone(base))
            if child != base:
                return child
        child, = gp.mutShrink(toolbox.clone(base))
        return child

    def _mate_retry(ind1, ind2):
        p1, p2 = toolbox.clone(ind1), toolbox.clone(ind2)

        if (p1 == p2) or pop_compare(p1, p2):
            return _mutate_retry(p1), _mutate_retry(p2)

        c1, c2 = toolbox.clone(p1), toolbox.clone(p2)
        c1, c2 = toolbox.mate(c1, c2)

        if (c1 == p1) and (c2 == p2):
            return _mutate_retry(p1), _mutate_retry(p2)

        return c1, c2

    i = 0
    while i < len(offspring):
        if i + 1 < len(offspring) and random.random() < new_cxpb:
            c1, c2 = _mate_retry(offspring[i], offspring[i + 1])
            offspring[i], offspring[i + 1] = c1, c2
            del offspring[i].fitness.values, offspring[i + 1].fitness.values
            i += 2
        else:
            child = _mutate_retry(offspring[i])
            offspring[i] = child
            del offspring[i].fitness.values
            i += 1

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen, halloffame=None):

    eval_cache = {}

    def _fp(ind):
        return hashlib.md5(str(ind).encode("utf-8")).hexdigest()

    def _eval_invalid_with_cache(ind_list):
        to_eval = []
        for ind in ind_list:
            key = _fp(ind)
            if key in eval_cache:
                fit, fold_accs = eval_cache[key]
                ind.fitness.values = fit
                if fold_accs is not None:
                    ind.fold_accs = fold_accs
            else:
                to_eval.append(ind)

        if to_eval:
            fitnesses = list(map(toolbox.evaluate, to_eval))
            for ind, fit in zip(to_eval, fitnesses):
                ind.fitness.values = fit
                eval_cache[_fp(ind)] = (fit, getattr(ind, 'fold_accs', None))

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    _eval_invalid_with_cache(invalid_ind)

    if halloffame is not None:
        halloffame.update(population)

    for _ in range(int(ngen)):
        n = len(population)

        k = 0
        elites = []
        if elitpb and elitpb > 0.0:
            k = max(1, int(round(elitpb * n)))
            k = min(k, n)
            elites = list(map(toolbox.clone, tools.selBest(population, k)))

        offspring = toolbox.select(population, n - k)
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        _eval_invalid_with_cache(invalid_ind)

        if k > 0:
            offspring = elites + offspring

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring

        best_ind = tools.selBest(population, 1)[0]
        if best_ind.fitness.valid and float(best_ind.fitness.values[0]) >= 100.0:
            break

    return population
