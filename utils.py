import networkx as nx
from networkx.utils import np_random_state
import numpy as np
import torch 
import random
import os


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def calculate_distances(layout_seed, layout_type, G, edge_index, dim):
    set_seed(layout_seed)
    if layout_type == 'fdl':
        pos = nx.spring_layout(G, seed=layout_seed, dim=dim)
        edge_distances = torch.zeros(edge_index.shape[1])
        for j in range(edge_index.shape[1]):
            u = edge_index[0, j].item()
            v = edge_index[1, j].item()
            u_pos = torch.tensor(pos[u])
            v_pos = torch.tensor(pos[v])
            distance = torch.norm(u_pos - v_pos)
            edge_distances[j] = distance.item()
        return edge_distances

    if layout_type == 'kk':
        pos = nx.kamada_kawai_layout(G, dim=dim)
        edge_distances = torch.zeros(edge_index.shape[1])
        for j in range(edge_index.shape[1]):
            u = edge_index[0, j].item()
            v = edge_index[1, j].item()
            u_pos = torch.tensor(pos[u])
            v_pos = torch.tensor(pos[v])
            distance = torch.norm(u_pos - v_pos)
            edge_distances[j] = distance.item()
        return edge_distances
    
    
def add_gaussian_noise(input_array, epoch, max_epoch=50, initial_std_dev=1e-3, final_std_dev=0):
    """
    parameters:
    - input_array
    - epoch
    - max_epoch
    - initial_std_dev
    - final_std_dev

    Return：
    - noise array
    """
    eps = 1e-6
    alpha = 0.5 * (1 + np.cos(np.pi * epoch / max_epoch))  # CosineAnnealingLR-like schedule
    std_dev = initial_std_dev + (final_std_dev - initial_std_dev) * alpha
    noise = np.random.normal(0, std_dev, input_array.shape) 
    noisy_array = input_array + noise
    return noisy_array

def add_langevin_noise(input_array,temperature=1e-7,iteration=0,max_iteration=50):
    """
    parameters:
    - input pos arr

    Return：
    - noise pos array
    """
    if iteration>= max_iteration- 2:
        return input_array
    noise = temperature*np.random.normal(0, 1, input_array.shape) 
    #noise = np.sqrt(temperature/np.log(iteration+2))*np.random.normal(0, 1, input_array.shape) 
    noisy_array = input_array + noise
    return noisy_array

# Reference: networkX

def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


@np_random_state(10)
def spring_layout(
    G,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    seed=None,
    sampling='base',
    temperature=1e-7,
):
    """Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.

    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `k` helps determine the distance between nodes,
    though `scale` and `center` determine the size and place after
    rescaling occurs at the end of the simulation.

    Fixing some nodes doesn't allow them to move in the simulation.
    It also turns off the rescaling feature at the simulation's end.
    In addition, setting `scale` to `None` turns off rescaling.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    fixed : list or None  optional (default=None)
        Nodes to keep fixed at initial position.
        Nodes not in ``G.nodes`` are ignored.
        ValueError raised if `fixed` specified and `pos` not.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  Larger means a stronger attractive force.
        If None, then all edge weights are 1.

    scale : number or None (default: 1)
        Scale factor for positions. Not used unless `fixed is None`.
        If scale is None, no rescaling is performed.

    center : array-like or None
        Coordinate pair around which to center the layout.
        Not used unless `fixed is None`.

    dim : int
        Dimension of layout.

    seed : int, RandomState instance or None  optional (default=None)
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spring_layout(G)

    # The same using longer but equivalent function name
    >>> pos = nx.fruchterman_reingold_layout(G)
    """
    import numpy as np

    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed if node in nfixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    try:
        # Sparse matrix
        if len(G) < 500:  # sparse solver for large graphs
            raise ValueError
        A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _sparse_fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed, sampling, temperature
        )
    except ValueError:
        A = nx.to_numpy_array(G, weight=weight)
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed, sampling, temperature
        )
    if fixed is None and scale is not None:
        pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    #(pos)
    return pos



fruchterman_reingold_layout = spring_layout


@np_random_state(7)
def _fruchterman_reingold(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None, sampling='base',temperature=1e-7
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k)
        )
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        if sampling =='noise':
            # pos=add_gaussian_noise(pos,iteration)
            pos=add_langevin_noise(pos,iteration=iteration,max_iteration=iterations,temperature=temperature)

        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
    return pos


@np_random_state(7)
def _sparse_fruchterman_reingold(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None, sampling='base',temperature=1e-7
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    import numpy as np
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err
    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except AttributeError:
        A = (sp.sparse.coo_array(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = A.getrowview(i).toarray()  # TODO: revisit w/ sparse 1D container
            # displacement "force"
            displacement[:, i] += (
                delta * (k * k / distance**2 - Ai * distance / k)
            ).sum(axis=1)
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        if sampling=='noise':
            pos=add_langevin_noise(pos,iteration=iteration,max_iteration=iterations,temperature=temperature)
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
    return pos



def kamada_kawai_layout(
    G, dist=None, pos=None, weight="weight", scale=1, center=None, dim=2
):
    """Position nodes using Kamada-Kawai path-length cost-function.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    dist : dict (default=None)
        A two-level dictionary of optimal distances between nodes,
        indexed by source and destination node.
        If None, the distance is computed using shortest_path_length().

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        circular_layout() for dim >= 2 and a linear layout for dim == 1.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.kamada_kawai_layout(G)
    """
    import numpy as np

    G, center = _process_params(G, center, dim)
    nNodes = len(G)
    if nNodes == 0:
        return {}

    if dist is None:
        dist = dict(nx.shortest_path_length(G, weight=weight))
    dist_mtx = 1e6 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]

    if pos is None:
        if dim >= 3:
            pos = random_layout(G, dim=dim)
        elif dim == 2:
            pos = circular_layout(G, dim=dim)
        else:
            pos = dict(zip(G, np.linspace(0, 1, len(G))))
    pos_arr = np.array([pos[n] for n in G])

    pos = _kamada_kawai_solve(dist_mtx, pos_arr, dim)

    pos = rescale_layout(pos, scale=scale) + center
    return dict(zip(G, pos))



def _kamada_kawai_solve(dist_mtx, pos_arr, dim):
    # Anneal node locations based on the Kamada-Kawai cost-function,
    # using the supplied matrix of preferred inter-node distances,
    # and starting locations.

    import numpy as np
    import scipy as sp
    import scipy.optimize  # call as sp.optimize

    meanwt = 1e-3
    costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3), meanwt, dim)

    optresult = sp.optimize.minimize(
        _kamada_kawai_costfn,
        pos_arr.ravel(),
        method="L-BFGS-B",
        args=costargs,
        jac=True,
    )

    return optresult.x.reshape((-1, dim))


def _kamada_kawai_costfn(pos_vec, np, invdist, meanweight, dim):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset**2)
    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum(
        "ij,ij,ijk->jk", invdist, offset, direction
    )

    # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos**2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())
    return pos


def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    See Also
    --------
    rescale_layout_dict
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos

def rescale_layout_dict(pos, scale=1):
    """Return a dictionary of scaled positions keyed by node

    Parameters
    ----------
    pos : A dictionary of positions keyed by node

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : A dictionary of positions keyed by node

    Examples
    --------
    >>> import numpy as np
    >>> pos = {0: np.array((0, 0)), 1: np.array((1, 1)), 2: np.array((0.5, 0.5))}
    >>> nx.rescale_layout_dict(pos)
    {0: array([-1., -1.]), 1: array([1., 1.]), 2: array([0., 0.])}

    >>> pos = {0: np.array((0, 0)), 1: np.array((-1, 1)), 2: np.array((-0.5, 0.5))}
    >>> nx.rescale_layout_dict(pos, scale=2)
    {0: array([ 2., -2.]), 1: array([-2.,  2.]), 2: array([0., 0.])}

    See Also
    --------
    rescale_layout
    """
    import numpy as np

    if not pos:  # empty_graph
        return {}
    pos_v = np.array(list(pos.values()))
    pos_v = rescale_layout(pos_v, scale=scale)
    return dict(zip(pos, pos_v))

@np_random_state(3)
def random_layout(G, center=None, dim=2, seed=None):
    """Position nodes uniformly at random in the unit square.

    For every node, a position is generated by choosing each of dim
    coordinates uniformly at random on the interval [0.0, 1.0).

    NumPy (http://scipy.org) is required for this function.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    seed : int, RandomState instance or None  optional (default=None)
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.lollipop_graph(4, 3)
    >>> pos = nx.random_layout(G)

    """
    import numpy as np

    G, center = _process_params(G, center, dim)
    pos = seed.rand(len(G), dim) + center
    pos = pos.astype(np.float32)
    pos = dict(zip(G, pos))

    return pos

def circular_layout(G, scale=1, center=None, dim=2):
    # dim=2 only
    """Position nodes on a circle.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.
        If dim>2, the remaining dimensions are set to zero
        in the returned positions.
        If dim<2, a ValueError is raised.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    ValueError
        If dim < 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.circular_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np

    if dim < 2:
        raise ValueError("cannot handle dimensions < 2")

    G, center = _process_params(G, center, dim)

    paddims = max(0, (dim - 2))

    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {nx.utils.arbitrary_element(G): center}
    else:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, len(G) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack(
            [np.cos(theta), np.sin(theta), np.zeros((len(G), paddims))]
        )
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))

    return pos


# ##################
# import multiprocessing as mp
# import networkx as nx
# import pandas as pd
# import subprocess
# import random

# import pkg_resources
# import tempfile
# import os

# classpath = (
#     pkg_resources.resource_filename("./forceatlas", "ext/forceatlas2.jar") +
#     ":" +
#     pkg_resources.resource_filename("./forceatlas", "ext/gephi-toolkit-0.9.2-all.jar")
# )

# def temp_filename() -> str:
#     return next(tempfile._get_candidate_names())
    
# def fa2_layout(
#     G,
#     pos=None,
#     iterations=50,
#     threshold=None,
#     directed=False,
#     dim=2,
#     splits=None,
#     theta=1.2,
#     update_iter=1,
#     update_center=False,
#     jitter_tolerance=1,
#     lin_log_mode=False,
#     repulsion=None,
#     gravity=1,
#     strong_gravity_mode=False,
#     outbound_attraction_distribution=False,
#     n_jobs=mp.cpu_count(),
#     seed=None,
# ):
#     """Position nodes using ForceAtlas2 force-directed algorithm.

#     Parameters
#     ----------
#     G : NetworkX graph or list of nodes
#         A position will be assigned to every node in G.

#     pos : dict or None  optional (default=None)
#         Initial positions for nodes as a dictionary with node as keys
#         and values as a coordinate list or tuple.  If None, then use
#         random initial positions.

#     iterations : int  optional (default=50)
#         Maximum number of iterations taken

#     threshold : float or None  optional (default=None)
#         Threshold for relative error in node position changes.
#         The iteration stops if the error is below this threshold.
        
#     directed : bool (default=False)
#         Whether input graph is directed.

#     dim : int (default: 2)
#         Dimension of layout.
        
#     splits : int or None  optional (default=None)
#         Rounds of splits to use for Barnes-Hut tree building.
#         Number of regions after splitting is 4^barnesHutSplits for 2D
#         and 8^barnesHutSplits for 3D.
        
#     theta : float (default=1.2)
#         Theta of the Barnes Hut optimization.
        
#     update_iter : int (default=1)
#         Update Barnes-Hut tree every update_iter iterations.
        
#     update_center : bool (default=False)
#         Update Barnes-Hut region centers when not rebuilding
#         Barnes-Hut tree.
        
#     jitter_tolerance : float (default=1)
#         How much swinging you allow. Above 1 discouraged.
#         Lower gives less speed and more precision.
        
#     lin_log_mode : bool (default=False)
#         Switch ForceAtlas' model from lin-lin to lin-log. 
#         Makes clusters more tight.
        
#     repulsion : float or None  optional (default: 1)
#         How much repulsion you want. More makes a more sparse graph.
#         None will default to 2.0 if nodes >= 100, otherwise 10.0.
        
#     gravity : float (default=1.0)
#         Attracts nodes to the center.
        
#     strong_gravity_mode : bool (default=False)
#         A stronger gravity law
        
#     outbound_attraction_distribution : bool (default=False)
#         Distributes attraction along outbound edges.
#         Hubs attract less and thus are pushed to the borders.
        
#     n_jobs : int  optional (defaults to all cores)
#         Number of threads to use for parallel computation.
#         If None, defaults to all cores as detected by
#         multiprocessing.cpu_count().

#     seed : int or None  optional (default=None)
#         Seed for random number generation for initial node position.
#         If int, `seed` is the seed used by the random number generator,
#         if None, the random number generator is chosen randomly.

#     Returns
#     -------
#     pos : dict
#         A dictionary of positions keyed by node

#     Examples
#     --------
#     >>> import forceatlas as fa2
#     >>> G = nx.path_graph(4)
#     >>> pos = fa2.fa2_layout(G)
#     """
#     try:
#         if not isinstance(G, nx.Graph):
#             empty_graph = nx.Graph()
#             empty_graph.add_nodes_from(G)
#             G = empty_graph
        
#         mapping = {label: index for index, label in enumerate(G.nodes())}
#         inverse_mapping = {index: label for label, index in mapping.items()}
#         H = nx.relabel_nodes(G, mapping)
            
#         temp_graph_filename = temp_filename() + ".net"
#         nx.write_pajek(H, temp_graph_filename)
        
#         output_filename = temp_filename() + ".coords"
        
#         command = [
#                 "java",
#                 "-Djava.awt.headless=true",
#                 "-Xmx8g",
#                 "-cp",
#                 classpath,
#                 "kco.forceatlas2.Main",
#                 "--input",
#                 temp_graph_filename,
#                 "--output",
#                 output_filename,
#                 "--nthreads",
#                 str(n_jobs),
#                 "--barnesHutTheta",
#                 str(theta),
#                 "--barnesHutUpdateIter",
#                 str(update_iter),
#                 "--jitterTolerance",
#                 str(jitter_tolerance),
#                 "--gravity",
#                 str(gravity),
#         ]
        
#         if dim == 2:
#             command.append("--2d")
            
#         if seed is not None:
#             command.extend(["--seed", str(seed)])
            
#         if splits is not None:
#             command.extend(["--barnesHutSplits", str(splits)])
            
#         if update_center:
#             command.append("--updateCenter")
            
#         if lin_log_mode:
#             command.append("--linLogMode")
            
#         if repulsion is not None:
#             command.extend(["--scalingRatio", str(repulsion)])
            
#         if strong_gravity_mode:
#             command.append("--strongGravityMode")
        
#         if outbound_attraction_distribution:
#             command.append("--outboundAttractionDistribution")
            
#         if directed:
#             command.append("--directed")
            
#         if threshold is None:
#             command.extend(["--nsteps", str(iterations)])
#         else:
#             command.extend([
#                 "--targetChangePerNode",
#                 str(threshold),
#                 "--targetSteps",
#                 str(iterations),
#             ])
            
#         if pos is not None:
#             temp_pos_filename = temp_filename() + ".csv"
#             pos_list = []
#             for label, coords in pos.items():
#                 row = {"id": mapping[label], "x": coords[0], "y": coords[1]}
#                 if dim == 3:
#                     row["z"] = coords[2]
#                 pos_list.append(row)
#             pos_list = pd.DataFrame(pos_list)
#             pos_list.to_csv(temp_pos_filename, sep='\t')
#             command.extend(["--coords", temp_pos_filename])
            
#         subprocess.check_call(command)
        
#         coordinates = pd.read_csv(output_filename + ".txt", header=0, index_col=0, sep="\t").values
        
#         if pos is not None:
#             os.remove(temp_pos_filename)
            
#         os.remove(temp_graph_filename)
#         os.remove(output_filename + ".txt")
        
#         if os.path.exists(output_filename + ".distances.txt"):
#             os.remove(output_filename + ".distances.txt")
        
#         pos = {inverse_mapping[i]: x for i, x in enumerate(coordinates)}
        
#         return pos
#     except Exception as e:
#         raise e
#     finally:
#         for path in [temp_graph_filename, output_filename + ".txt", output_filename + ".distances.txt"]:
#             if os.path.exists(path):
#                 os.remove(path)
                
#         while True:
#             try:
#                 os.remove(temp_pos_filename)
#                 break
#             except KeyboardInterrupt:
#                 continue
#             except:
#                 break
