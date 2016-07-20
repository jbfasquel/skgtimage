import networkx as nx
import skgtimage as skgti
import matplotlib.pylab as plt

#from skgtimage.core.search_base import recursive_brothers

#p_graph=skgti.core.graph_factory("B=C<D=A")
graph1=skgti.core.IrDiGraph()
graph1.add_node('C')
graph1.add_node('A')
graph1.add_node('B')
graph1.add_node('D')
graph1.add_edge('B','C')
graph1.add_edge('C','B')
graph1.add_edge('D','A')
graph1.add_edge('A','D')
graph1.add_edge('C','D')


print(graph1)

print(skgti.core.find_head(graph1))
result_graph1=skgti.core.decreasing_ordered_nodes(graph1)
print(result_graph1)
result_graph1=skgti.core.decreasing_ordered_nodes(graph1)
print(result_graph1)
quit()

#######################
# TEST GRAPH1
#######################
graph1=skgti.core.IrDiGraph()
graph1.add_node('A')
graph1.add_node('B')
graph1.add_node('C')
graph1.add_edge('A','B')
graph1.add_edge('B','C')
graph1.add_edge('C','B')

#skgti.io.plot_graphs_regions_new([graph1]);plt.show()
#all_graphs=skgti.core.compute_possible_graphs(graph1)
result_graph1=skgti.core.decreasing_ordered_nodes(graph1)
print(result_graph1)
#

#######################
# TEST GRAPH2
#######################
graph2=skgti.core.IrDiGraph()
graph2.add_node('bottom')
graph2.add_node('A')
graph2.add_node('B')
graph2.add_edge('bottom','A')
graph2.add_edge('B','A')
graph2.add_edge('A','B')
graph2.add_node('C')
graph2.add_edge('B','C')
graph2.add_node('D')
graph2.add_edge('C','D')
graph2.add_edge('D','C')
graph2.add_node('E')
graph2.add_edge('D','E')
graph2.add_edge('E','D')
graph2.add_node('F')
graph2.add_edge('E','F')

result_graph1=skgti.core.decreasing_ordered_nodes(graph2)
print(result_graph1)
skgti.io.plot_graphs_regions_new([graph2]);plt.show()

#all_graphs=skgti.core.compute_possible_graphs(graph2)

#skgti.io.plot_graph(graph2);plt.show();quit()
#skgti.io.plot_graphs_regions_new([graph2]+all_graphs[0:5]);plt.show()

