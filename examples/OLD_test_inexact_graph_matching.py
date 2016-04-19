import networkx as nx
import skgtimage as skgti

g_ref=nx.DiGraph()
g_ref.add_node('A')
g_ref.add_node('B')
g_ref.add_edge('B','A')
g_ref.add_edge('A','B')
g_ref.add_node('C')

print(skgti.core.recursive_brothers(g_ref,'A'))

g_query=nx.DiGraph()
g_query.add_node('0')
g_query.add_node('1')
g_query.add_edge('1','0')

def mynodematch(n1):
    return False

matcher=nx.isomorphism.DiGraphMatcher(g_query,g_ref,node_match=mynodematch)
isomorphisms=[i for i in matcher.isomorphisms_iter()]
print(isomorphisms)
quit()
'''
#g_ref.add_edge('C','B')
cc=[i for i in nx.connected_components(g_ref.to_undirected())]
print(cc)
print({'A','B','C'} in nx.connected_components(g_ref.to_undirected()))
print({'A'} in nx.connected_components(g_ref.to_undirected()))
'''
'''
g_query=nx.DiGraph()
g_query.add_node(1)
g_query.add_node(0)
g_query.add_node(2)
g_query.add_node(3)

g_query.add_edge(0,1)
g_query.add_edge(2,1)
g_query.add_edge(3,0)


matcher=nx.isomorphism.DiGraphMatcher(g_query,g_ref)
isomorphisms=[i for i in matcher.isomorphisms_iter()]
print(isomorphisms)

isomorphisms=[i for i in matcher.subgraph_isomorphisms_iter()]
print(isomorphisms)
'''