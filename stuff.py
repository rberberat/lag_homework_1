import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy

def setupMatrix1():
    nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    edges = [[0, 1], [0, 6], [0, 7],
             [1, 2], [1, 7],
             [2, 1], [2, 7],
             [3, 5], [3, 7],
             [4, 5],
             [5, 6],
             [6, 5],
             [7, 6], ]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    incidence_matrix = -nx.incidence_matrix(G, oriented=True)

    return np.array(incidence_matrix.toarray())

def setupMatrix2():
    nodes = [1, 2, 3, 4]
    edges = [[1,2], [1,3], [1,4], [2,3],
             [2,4], [3,1], [4,1], [4,3]]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    incidence_matrix = -nx.incidence_matrix(G, oriented=True)

    return np.array(incidence_matrix.toarray())

def setupMatrix3():
    nodes = [1,2,3]
    edges = [[1,2],[2,1],[2,3],[3,1]]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    incidence_matrix = -nx.incidence_matrix(G, oriented=True)

    return np.array(incidence_matrix.toarray())

def turn_im_into_am( E ):
    edge_starting_points = np.where(E.T > 0)
    edge_end_points = np.where(E.T < 0)

    g = nx.DiGraph()

    for i in range(len(E)):
        g.add_node(i)

    for i in range(len(edge_starting_points[1])):
        foo = edge_starting_points[1][i]
        bar = edge_end_points[1][i]
        g.add_edge(foo, bar)

    am = nx.adjacency_matrix(g)

    return np.array(am.toarray())

def turn_am_into_im( A ):
    g = nx.DiGraph(A)

    im = -nx.incidence_matrix(g, oriented=True)

    return np.array(im.toarray())

def create_lm_from_im(im):
    am = turn_im_into_am(im)
    am = am.astype(float)

    foo = np.sum(am, axis=1, dtype=float)

    for i, s in enumerate(foo):
        foo[i] = 1/s

    for iterable, value in enumerate(am):
        am[iterable] = value*foo[iterable]

    return am.T

def create_meaning_vector_from_im(im, alpha=0.85):
    foo = create_lm_from_im(im)
    foo = foo*alpha

    bar = np.ones((len(foo), len(foo)), dtype=float)
    bar = bar/len(foo)
    bar = bar*(1-alpha)

    baz = foo+bar
    print("baz", baz)

    bomb = np.zeros((len(baz), len(baz)), dtype=float)
    for i,s in enumerate(bomb):
        bomb[i][i] = 1

    print("bomb", bomb)

    baz = baz-bomb
    print("baz", baz)

    nullvector = np.zeros(len(baz), dtype=float)
    print("nullvector", nullvector)

    solution = np.linalg.inv(baz)
    print(solution.astype(float))
    print()

    print(baz*solution)

    return None

def guassian_algorithm(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i, 0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = guassian_algorithm(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = guassian_algorithm(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

E = setupMatrix2()
create_meaning_vector_from_im(E)