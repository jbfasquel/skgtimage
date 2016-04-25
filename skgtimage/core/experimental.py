def find_head(graph):
    head=set()
    for n in graph.nodes():
        if len(graph.successors(n)) == 0: head = set([n])
    return head

def ordered_nodes(graph):
    head=find_head(graph)
    top2bottom=[head]
    pred=graph.predecessors(list(head)[0])
    continue_cond = (len(pred) !=0 )
    while continue_cond:
        top2bottom+=[pred]
        pred=graph.predecessors(pred[0])
        continue_cond = (len(pred) !=0 )
    return top2bottom


def compute_energie(p_iso,image,residues,p_graph,ref_p_graph):
    stats=[]
    for r in residues:
        stats+=[region_stat(image,r,fct=np.mean,mc=False)]
    #print(stats)
    #print(ref_p_graph)
    head=find_head(ref_p_graph)
    #print(head)
    grps_b=find_groups_of_brothers(ref_p_graph)
    ordered=ordered_nodes(ref_p_graph)
    #print(ordered)
    oi=oirelationships(p_iso)
    energie=0
    for i in range(1,len(ordered)):
        ref_node_prev=list(ordered[i-1])[0]
        ref_node_current=list(ordered[i])[0]
        query_node_prev=list(oi[ref_node_prev])[0]
        query_node_current=list(oi[ref_node_current])[0]
        diff=stats[query_node_prev]-stats[query_node_current]
        energie-=diff
    return energie
'''
#print(common_isomorphisms)
for p_iso in common_isomorphisms:
    print("ISO: ", p_iso)
    energie=compute_energie(p_iso,image,new_residues,built_p_graph,p_graph)
    print("--> ", energie)
'''