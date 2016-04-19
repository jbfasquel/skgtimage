import networkx as nx
import skgtimage as skgti
import matplotlib.pylab as plt

graph1=skgti.core.IrDiGraph()
graph1.add_node('A')
graph1.add_node('B')
graph1.add_node('C')
#
graph1.add_edge('A','B')
graph1.add_edge('B','C')
graph1.add_edge('C','B')


all_graphs=skgti.core.compute_possible_graphs(graph1)

skgti.io.plot_graphs_regions_new([graph1]+all_graphs);plt.show()

#######################
# TEST GRAPH
#######################
g_ref=skgti.core.IrDiGraph()
g_ref.add_node('bottom') #bug ??
#g_ref.add_node('X')
g_ref.add_node('A')
g_ref.add_node('B')
g_ref.add_edge('bottom','A')
g_ref.add_edge('B','A')
g_ref.add_edge('A','B')
g_ref.add_node('C')
g_ref.add_edge('B','C')
g_ref.add_node('D')
g_ref.add_edge('C','D')
g_ref.add_edge('D','C')
g_ref.add_node('E')
g_ref.add_edge('D','E')
g_ref.add_edge('E','D')

g_ref.add_node('F')
g_ref.add_edge('E','F')

#BEGIN BROTHERS
'''
g_ref.add_node('G')
g_ref.add_edge('F','G')
g_ref.add_node('H')
g_ref.add_edge('G','H')
g_ref.add_edge('H','G')
'''
#END BROTHERS

#skgti.io.plot_graph(g_ref);plt.show();quit()


skgti.io.plot_graphs_regions_new([g_ref]+all_graphs[0:5]);plt.show()


'''
#################
# ANALYZE OF (GROUPS OF) BROTHERS
#################
groups_of_brothers=skgti.core.find_groups_of_brothers(g_ref)
predecessors=skgti.core.predecessors_of_each_groups_of_brothers(g_ref,groups_of_brothers)
successors=skgti.core.successors_of_each_groups_of_brothers(g_ref,groups_of_brothers)
print("Groups",groups_of_brothers)
print("Predecessors: ",predecessors)
print("Successors: ",successors)

#################
# FINDING ALL POSSIBLE ORDERINGS OF BROTHERS
#################
all_orderings=skgti.core.orderings_of_groups_of_brothers(groups_of_brothers)

#################
# GENERATING A SPECIFIC ORDERING
#################
clean_graph=g_ref.copy()
skgti.core.disconnect_brothers(clean_graph,groups_of_brothers)
#skgti.io.plot_graph(c_g_ref);plt.show();quit()

all_graphs=[]
#for i in range(0,len(all_orderings)):
for i in range(0,5):
    reordered_graph=clean_graph.copy()
    current_ordering=all_orderings[i]
    skgti.core.generate_connection(reordered_graph,current_ordering,groups_of_brothers,predecessors,successors)
    all_graphs+=[reordered_graph]
#skgti.io.plot_graph(reordered_graph);plt.show()
'''
