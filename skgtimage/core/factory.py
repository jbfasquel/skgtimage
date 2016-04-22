import re,itertools
import networkx as nx
from skgtimage.core.graph import IrDiGraph,transitive_reduction

def __analyze_sentence__(g,desc) :
    operators=re.findall('<|>|=',desc)
    operands=re.split('<|>|=',desc)
    multioperands=[ re.split(',',o) for o in operands]
    for operands in multioperands:
        for o in operands : g.add_node(o)
    for i in range(0,len(operators)):
        operator=operators[i]
        left_operands=multioperands[i]
        right_operands=multioperands[i+1]
        if operator == '<':
            for l,r in itertools.product(left_operands,right_operands):
                g.add_edge(l,r)
        elif operator == '=':
            for l,r in itertools.product(left_operands,right_operands):
                g.add_edge(l,r)
                g.add_edge(r,l)
        elif operator == '>':
            for l,r in itertools.product(left_operands,right_operands):
                g.add_edge(r,l)

def graph_factory(desc):
    g=IrDiGraph()
    #Remove spaces
    nospace_desc=re.sub(' ','',desc)
    #Remove == -> =
    nospace_desc=re.sub('==','=',nospace_desc)
    #Split into sentences (separator is ';')
    descs=re.split(';',nospace_desc)
    #
    for d in descs : __analyze_sentence__(g,d)
    #
    return g

def fromstring(desc,g=None):
    if g is None: g=nx.DiGraph()
    #Remove spaces
    nospace_desc=re.sub(' ','',desc)
    #Remove == -> =
    nospace_desc=re.sub('==','=',nospace_desc)
    #Split into sentences (separator is ';')
    descs=re.split(';',nospace_desc)
    #
    for d in descs : __analyze_sentence__(g,d)
    #
    return g



