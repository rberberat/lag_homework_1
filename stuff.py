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

    length = len(foo)
    bar = np.ones((length, length), dtype=float)
    bar = bar/length
    bar = bar*(1-alpha)

    baz = foo+bar #G
    print("baz", baz)

    bomb = np.zeros((len(baz), len(baz)), dtype=float)
    for i,s in enumerate(bomb):
        bomb[i][i] = 1.0 #En

    print("bomb", bomb)

    baz = baz-bomb #G-En
    print("baz", baz)

    print("stuffy", np.zeros((1,3), dtype=float))
    print("sol", np.linalg.solve(baz, np.zeros((3), dtype=float)))


    return None

E = setupMatrix3()
create_meaning_vector_from_im(E)