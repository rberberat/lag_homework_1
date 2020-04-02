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

    # Create Link Matrix and multiply by alpha to get P
    P = create_lm_from_im(im)
    P = P * alpha

    # Create Matrix of Random Surf-behaviour and multiply by 1-alpha to get S
    S = np.ones((len(P), len(P)), dtype=float)/len(P)
    S = S * (1-alpha)

    # Combine P and S to get the Google Matrix
    G = P + S #G

    # Create Identity Matrix En
    En = np.eye(len(G), dtype=float)

    # Subtract Identity Matrix (En) from Google Matrix (G) to get our A for the formula Ax = B
    A = G - En #G-En

    # Create B for the Formula Ax = B. A Vector of Ones is chosen, since a vector of 0s would yield an unusable result.
    B = np.ones((len(A)), dtype=float)

    # Solve the Formula for x.
    solution = np.linalg.solve(A, B)

    # Normalize the Solution
    solution_normalizer = np.linalg.norm(solution)
    normalized_solution = []
    for result in solution:
        normalized_result = result / solution_normalizer
        if normalized_result < 0:
            normalized_result = 0
        normalized_solution.append(normalized_result)

    return normalized_solution

def create_PageRank_with_NXAlgo(im, alpha=0.85):
    g = nx.DiGraph(turn_im_into_am(im))

    return list(nx.pagerank(g, alpha=alpha).values())

def plot_graph(A, knoten_gewichte=None):
    """
    Funktion zur graphischen Darstellung eines Graphen.
    Benutzt das 'spring layout', eventuell muss die Funktion mehrere Male ausgeführt werden, bis eine schöne Darstellung
    des Graphen vorliegt.

    Arguments:
    A -- Adjazenzmatrix (shape (n_knoten,n_knoten))
    knoten_gewichte -- Liste mit Gewichte für jeden Knoten im Graphen (bei None erhalten alle Knoten die gleichen Gewichte)
    """

    if knoten_gewichte is None:
        knoten_gewichte = np.array([1] * A.shape[0])

    assert (len(knoten_gewichte) == A.shape[0])

    knoten_gewichte = knoten_gewichte / np.mean(knoten_gewichte)

    plt.figure(figsize=(8, 8))
    G = nx.DiGraph(A)
    pos = nx.layout.spring_layout(G)
    options = {
        'node_color': '#dd0000',
        'node_size': knoten_gewichte * 2500,
        'width': 3,
        'arrowstyle': '-|>',
        'arrowsize': 12,
    }

    nx.draw_networkx(G, pos, arrows=True, **options)
    plt.axis("off")
    plt.show()

def plot_Graph_from_Im(IM, use_student_made_algo=False):
    AM = turn_im_into_am(IM)

    if(use_student_made_algo):
        weights = create_meaning_vector_from_im(IM)
    else:
        weights = create_PageRank_with_NXAlgo(IM)

    plot_graph(AM, knoten_gewichte=weights)

E = setupMatrix1()
plot_Graph_from_Im(E, use_student_made_algo=True)
plot_Graph_from_Im(E, use_student_made_algo=False)