import const
import networkx     as nx
import numpy        as np
from utils          import cal_iou
from scipy.optimize import linear_sum_assignment


def max_weight_matching(state_list, obs_list):
    """
    The 'Max Weight Matching' Process.

    Args:
        state_list  [List]: A list of each predicted BB position (2D numpy array; "xywh") in the current video frame.
        obs_list    [List]: A list of each  detected BB position (2D numpy array; "xywh") in the current video frame.

    Returns:
        res [Dict]: A dictionary of matched pairs with ('state', 'obs').
    """

    """Create a bipartite graph for the 'Max Weight Matching' Process."""
    # By definition, a Graph is a collection of nodes (vertices) along with identified pairs of nodes (called edges).
    graph = nx.Graph() # Create an empty graph with no nodes and no edges.

    for i, state in enumerate(state_list):
        state_node = 'state_%d' % i
        graph.add_node(state_node, bipartite=0)
        # Use a node attribute named bipartite with values 0 or 1
        # to identify the node set each node belongs to.

        for j, obs in enumerate(obs_list):
            obs_node = 'obs_%d' % j
            graph.add_node(obs_node, bipartite=1)

            """Calculate the IoU between two bounding boxes as edge weight."""
            score = cal_iou(state, obs)
            if score >= const.IOU_MIN: # Exclude the case when the edge score is below a predefined value.
                graph.add_edge(state_node, obs_node, weight=score)

    """Compute a maximum weighted matching of the bipartite graph."""
    match_set = nx.max_weight_matching(graph, maxcardinality=False) # 'match_set': A set for matched node pairs.
    res = dict()
    for (node_1, node_2) in match_set:
        if node_1.split('_')[0] == 'obs':
            node_1, node_2 = node_2, node_1
        res[node_1] = node_2 # 'key': 'state', 'value': 'obs'.

    return res


def hungarian_algorithm(state_list, obs_list):
    """
    The Hungarian Algorithm.

    Args:
        state_list  [List]: A list of each predicted BB position (2D numpy array; "xywh") in the current video frame.
        obs_list    [List]: A list of each  detected BB position (2D numpy array; "xywh") in the current video frame.

    Returns:
        res [Dict]: A dictionary of matched pairs with ('state', 'obs').
    """

    """Initialize a matrix for the Hungarian Algorithm."""
    iou_matrix = np.zeros((len(state_list), len(obs_list)))

    """Process the Hungarian Algorithm."""
    for i, state in enumerate(state_list):
        for j, obs in enumerate(obs_list):
            iou_dist = cal_iou(state, obs)
            if iou_dist >= const.IOU_MIN:
                iou_matrix[i, j] = iou_dist
    cost_matrix = - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    res = dict()
    for i in range(len(row_ind)):
        res["state_{}".format(row_ind[i])] = "obs_{}".format(col_ind[i])

    return res
