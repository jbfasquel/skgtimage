import networkx as nx

G = nx.complete_graph(5)

from networkx.drawing.nx_agraph import graphviz_layout

pos = graphviz_layout(G)

nx.draw(G, pos)

a=nx.nx_agraph.to_agraph(G)