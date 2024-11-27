'''
    Visualize the PatchCPG instance.
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def VisualGraph(graph, state=0, options=0, with_labels=True, show_label=False):
    # get each element from graph.
    edgeIndex = graph['edge_index']
    nodeAttr = graph['x']
    edgeAttr = graph['edge_attr']
    label = graph['y']

    # convert to numpy array.
    edgeIndex = edgeIndex.numpy()
    nodeAttr = nodeAttr.numpy()
    label = label.numpy()

    # construct graph.
    G = nx.MultiDiGraph()
    G.add_nodes_from([n for n in range(len(nodeAttr))])
    for i in range(len(edgeIndex[0])):
        G.add_edge(edgeIndex[0][i], edgeIndex[1][i])
    # draw graph.
    if (len(nodeAttr) == len(label)): # y is the node label.
        # draw figure.
        fig = plt.figure()
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G, node_color=label, with_labels=with_labels, pos=nx.spring_layout(G, random_state=state), cmap='Set1')
        if (show_label):
            plt.show()
    elif (1 == len(label)): # y is the graph label.
        # weights0 for node.
        weights0 = np.sum(np.abs(nodeAttr), axis=1)
        weights0 = np.sqrt((weights0 - min(weights0)) / (max(weights0) - min(weights0)))
        # judge if there are edge attributes.
        if (edgeAttr == None):
            # draw figure.
            fig = plt.figure()
            plt.xticks([])
            plt.yticks([])
            nx.draw_networkx(G, pos=nx.spring_layout(G, random_state=state), with_labels=with_labels, font_size=8,
                             node_color=weights0, cmap=plt.cm.YlGn)
            if (show_label):
                plt.show()
        else:
            # edgelist
            edgelist = []
            for i in range(len(edgeIndex[0])):
                edgelist.append((edgeIndex[0][i], edgeIndex[1][i]))
            edgelist = tuple(edgelist)
            edgeAttr = edgeAttr.t().numpy()
            # weights1 for before-after. 1-after 2-before 3-context
            weights1 = 2 * edgeAttr[0] + edgeAttr[1]
            dict1 = {1: 1, 2: 0, 3: 0.5}  # after:red, before:blue, context:grey
            weights1 = tuple([dict1[w] for w in weights1])
            # weights2 for 0-cdg, 1-ddg, 2-ast
            weights2 = np.argmax(edgeAttr[2:], axis=0)
            dict2 = {0: 1, 1: 0, 2: 0.5} # cdg:red, ddg:blue, ast:grey
            weights2 = tuple([dict2[w] for w in weights2])
            # draw figure.
            fig = plt.figure()
            plt.xticks([])
            plt.yticks([])
            if (0 == options):
                nx.draw_networkx(G, pos=nx.spring_layout(G, random_state=state), with_labels=with_labels, font_size=8,
                                 node_color=weights0, cmap=plt.cm.YlGn, node_size=240,
                                 edgelist=edgelist, edge_color=weights1, width=2, edge_cmap=plt.cm.coolwarm)
                blue_line = mlines.Line2D([], [], color='blue', label='before')
                gray_line = mlines.Line2D([], [], color='gray', label='context')
                red_line = mlines.Line2D([], [], color='red', label='after')
                plt.legend(handles=[blue_line, gray_line, red_line])
            elif (1 == options):
                nx.draw_networkx(G, pos=nx.spring_layout(G, random_state=state), with_labels=with_labels, font_size=8,
                                 node_color=weights0, cmap=plt.cm.YlGn, node_size=240,
                                 edgelist=edgelist, edge_color=weights2, width=2, edge_cmap=plt.cm.coolwarm)
                red_line = mlines.Line2D([], [], color='red', label='CDG')
                blue_line = mlines.Line2D([], [], color='blue', label='DDG')
                gray_line = mlines.Line2D([], [], color='gray', label='AST')
                plt.legend(handles=[red_line, blue_line, gray_line])
            else:
                print('[ERROR] <VisualGraph> argument \'options\' should be either 0 or 1!')
            if (show_label):
                plt.show()

    return fig