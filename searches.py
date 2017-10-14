import queue
import sys
import hashlib
from copy import deepcopy
import collections
import bisect


def memoize(fn, slot=None, maxsize=32):
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                # print(val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


class Queue:
    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)


class PriorityQueue(Queue):
    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def __repr__(self):
        return str(self.A)

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)


infinity = float('inf')

# ______________________________________________________________________________


class Problem(object):
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self, state):
        raise NotImplementedError


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        self.num = None
        if parent:
            self.depth = parent.depth + 1

    def set_num(self, num):
        self.num = num

    def get_num(self):
        return self.num

    def __repr__(self):
        return "<Node {}>{},{},{}".format(self.state, self.action, self.path_cost, self.num)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        return [node.action for node in self.path()[1:]]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# ______________________________________________________________________________


class Puzzle(Problem):
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        allowed_actions = []
        i = state.index(0)
        r = i // 3
        c = i % 3
        if r == 1 or r == 2:
            allowed_actions.append(1)
        if c == 1 or c == 0:
            allowed_actions.append(2)
        if r == 1 or r == 0:
            allowed_actions.append(3)
        if c == 1 or c == 2:
            allowed_actions.append(4)
        return allowed_actions

    def result(self, state, action):
        i = state.index(0)
        statex = state[:]
        if action == 1:
            i_2 = i - 3
        elif action == 3:
            i_2 = i + 3
        elif action == 4:
            i_2 = i - 1
        elif action == 2:
            i_2 = i + 1
        statex[i], statex[i_2] = statex[i_2], statex[i]
        return(statex)

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1  # s1 is parent, s2 current child via action

    def value(self, state):
        raise NotImplementedError

    def h(self, node, typ):
        # no of misplaced tiles
        if typ == "Misplaced":
            misplaced = 0
            for i, x in enumerate(node.state):
                if i != 0 and i != x:
                    misplaced += 3
            # print(misplaced)
            return (misplaced)
        # manhattan dist
        elif typ == "Manhattan":
            manhattan = 0
            for i, x in enumerate(node.state):
                if i != 0 and i != x:
                    real_r = i // 3
                    real_c = i % 3
                    ideal_r = x // 3
                    ideal_c = x % 3
                    manhattan += abs(real_r - ideal_r) + abs(real_c - ideal_c)
            # print(manhattan)
            return manhattan

        elif typ == "None":
            return 0

        return 1


class River_Cross(Problem):
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        allowed_actions = []
        possible_actions = ['1m', '1c', '2m', '2c', '1m1c']

        side = 0
        side_op = 1
        if(state[1][2]):
            side = 1
            side_op = 0

        for action in possible_actions:
            if action == '1m':
                if ((state[side][0] - 1 != 2 or state[side][1] != 3) and (state[side_op][0] + 1 != 2 or state[side_op][1] != 3) and (state[side][0] - 1 != 1 or state[side][1] != 3) and (state[side_op][0] + 1 != 1 or state[side_op][1] != 3) and (state[side][0] - 1 != 1 or state[side][1] != 2) and (state[side_op][0] + 1 != 1 or state[side_op][1] != 2)) and state[side][0] - 1 >= 0 and state[side_op][0] + 1 <= 3:
                    allowed_actions.append(action)
            elif action == '1c':
                if ((state[side][0] != 2 or state[side][1] - 1 != 3) and (state[side_op][0] != 2 or state[side_op][1] + 1 != 3) and (state[side][0] != 1 or state[side][1] - 1 != 3) and (state[side_op][0] != 1 or state[side_op][1] + 1 != 3) and (state[side][0] != 1 or state[side][1] - 1 != 2) and (state[side_op][0] != 1 or state[side_op][1] + 1 != 2)) and state[side][1] - 1 >= 0 and state[side_op][1] + 1 <= 3:
                    allowed_actions.append(action)
            elif action == '2m':
                if ((state[side][0] - 2 != 2 or state[side][1] != 3) and (state[side_op][0] + 2 != 2 or state[side_op][1] != 3) and (state[side][0] - 2 != 1 or state[side][1] != 3) and (state[side_op][0] + 2 != 1 or state[side_op][1] != 3) and (state[side][0] - 2 != 1 or state[side][1] != 2) and (state[side_op][0] + 2 != 1 or state[side_op][1] != 2)) and state[side][0] - 2 >= 0 and state[side_op][0] + 2 <= 3:
                    allowed_actions.append(action)
            elif action == '2c':
                if ((state[side][0] != 2 or state[side][1] - 2 != 3) and (state[side_op][0] != 2 or state[side_op][1] + 2 != 3) and (state[side][0] != 1 or state[side][1] - 2 != 3) and (state[side_op][0] != 1 or state[side_op][1] + 2 != 3) and (state[side][0] != 1 or state[side][1] - 2 != 2) and (state[side_op][0] != 1 or state[side_op][1] + 2 != 2)) and state[side][1] - 2 >= 0 and state[side_op][1] + 2 <= 3:
                    allowed_actions.append(action)
            elif action == '1m1c':
                if ((state[side][0] - 1 != 2 or state[side][1] - 1 != 3) and (state[side_op][0] + 1 != 2 or state[side_op][1] + 1 != 3) and (state[side][0] - 1 != 1 or state[side][1] - 1 != 3) and (state[side_op][0] + 1 != 1 or state[side_op][1] + 1 != 3) and (state[side][0] - 1 != 1 or state[side][1] - 1 != 2) and (state[side_op][0] + 1 != 1 or state[side_op][1] + 1 != 2)) and state[side][0] - 1 >= 0 and state[side_op][0] + 1 <= 3 and state[side][1] - 1 >= 0 and state[side_op][1] + 1 <= 3:
                    allowed_actions.append(action)

        return allowed_actions

    def result(self, state, action):

        side = 0
        side_op = 1
        statey = deepcopy(state)
        # print("hellp",state)
        if(state[1][2]):
            side = 1
            side_op = 0

        if action == '1m':
            statey[side][0] = statey[side][0] - 1
            statey[side_op][0] = statey[side_op][0] + 1
        elif action == '1c':
            statey[side][1] = statey[side][1] - 1
            statey[side_op][1] = statey[side_op][1] + 1
        elif action == '2m':
            statey[side][0] = statey[side][0] - 2
            statey[side_op][0] = statey[side_op][0] + 2
        elif action == '2c':
            statey[side][1] = statey[side][1] - 2
            statey[side_op][1] = statey[side_op][1] + 2
        elif action == '1m1c':
            statey[side][0] = statey[side][0] - 1
            statey[side_op][0] = statey[side_op][0] + 1
            statey[side][1] = statey[side][1] - 1
            statey[side_op][1] = statey[side_op][1] + 1

        statey[0][2], statey[1][2] = statey[1][2], statey[0][2]
        return statey

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1  # s1 is parent, s2 current child via action

    def value(self, state):
        raise NotImplementedError


def bfs(problem):
    initial = Node(problem.initial)
    # print(initial)
    if problem.goal_test(initial.state):
        return 1
    frontier = []
    frontier_states = set()
    frontier.append(initial)
    frontier_states.add(hashlib.sha224(
        str(initial.state).encode('utf-8')).hexdigest())
    explored = set()
    moves = 0
    while frontier:
        node = frontier.pop(0)
        moves += 1
        if problem.goal_test(node.state):
            num = 0
            for i, nodes in enumerate(node.path()):
                print(nodes.state)
                num = i
            print("\nMoves: %s\n" % num)
            return 1
        state_str = hashlib.sha224(str(node.state).encode('utf-8')).hexdigest()
        explored.add(state_str)
        frontier_states.remove(state_str)
        for child in node.expand(problem):
            child_hash = hashlib.sha224(
                str(child.state).encode('utf-8')).hexdigest()
            if (child_hash not in explored and child_hash not in frontier_states):
                frontier.append(child)
                frontier_states.add(child_hash)
    return 0


def depth_limited_search(problem, limit=50):
    rec = 1
    explored = set()

    def recursive_dls(node, problem, limit, rec, explored):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            state_str = hashlib.sha224(
                str(node.state).encode('utf-8')).hexdigest()
            explored.add(state_str)
            for child in node.expand(problem):
                if hashlib.sha224(str(child.state).encode('utf-8')).hexdigest() not in explored:
                    result = recursive_dls(
                        child, problem, limit - 1, rec + 1, explored)
                    if result == 'cutoff':
                        cutoff_occurred = True
                    elif result is not None:
                        return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit, rec, explored)


def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            num = 0
            for i, nodes in enumerate(result.path()):
                print(nodes.state)
                num = i
            print("\nMoves: %s\n" % num)
            return result


def best_first_graph_search(problem, f):
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    frontier_states = set()
    frontier_states.add(hashlib.sha224(
        str(node.state).encode('utf-8')).hexdigest())
    explored = set()
    moves = 0
    while frontier:
        node = frontier.pop()
        moves += 1
        if problem.goal_test(node.state):
            num = 0
            for i, nodes in enumerate(node.path()):
                print(nodes.state)
                num = i
            print("\nMoves: %s\n" % num)
            return node
        state_str = hashlib.sha224(str(node.state).encode('utf-8')).hexdigest()
        explored.add(state_str)
        frontier_states.remove(state_str)
        for child in node.expand(problem):
            child_hash = hashlib.sha224(
                str(child.state).encode('utf-8')).hexdigest()
            if (child_hash not in explored and child_hash not in frontier_states):
                frontier.append(child)
                frontier_states.add(child_hash)
            elif child_hash in frontier_states:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    frontier_states.add(child_hash)
    return None


def astar_search(problem, h=None, typ="Manhattan"):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n, typ))


# 8Puzzle
print("---Q1------------------8PUZZLE-----------------------------")
'''Input in the form of a list with the zero being the empty cell.
    Divide by 3 of index of number - Quotient for row number, Remainder for column number
'''
initial = [7, 2, 4, 5, 0, 6, 8, 3, 1]
goal = [0, 1, 2, 3, 4, 5, 6, 7, 8]
p = Puzzle(initial, goal)
print("---Q1A---------------UNINFORMED BFS------------------------")
bfs(p)
print("-----------------------------------------------------------")
print("---Q1B---------Iterative Deepening Search------------------")
iterative_deepening_search(p)
print("-----------------------------------------------------------")
print("---Q1C1--------------A* - Misplaced------------------------")
astar_search(p, typ="Misplaced")
print("-----------------------------------------------------------")
print("---Q1C2--------------A* - Manhattan------------------------")
astar_search(p)
print("-----------------------------------------------------------")
# River Crossing
print("---Q2----------------River Crossing------------------------")
'''Input in the form of a list of list of M,C,B (Missionaries, Cannibals and Boat)
    on both sides of the river
'''
initial = [[3, 3, 1], [0, 0, 0]]
goal = [[0, 0, 0], [3, 3, 1]]
p = River_Cross(initial, goal)
print("---------------------UNINFORMED BFS------------------------")
print("--M--C--B-||-M--C--B--")
bfs(p)
