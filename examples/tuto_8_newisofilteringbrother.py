import networkx as nx
import skgtimage as skgti
import numpy as np
import matplotlib.pylab as plt



p_ref=skgti.core.from_string("A<B=C<D")
#print(p_ref.nodes())


image=np.array([[0,1,2,3,4]])
label=np.array([[0,1,2,3,4]])
_,p=skgti.core.from_labelled_image(image,label)
#p=skgti.core.from_string("0<1<2<3<4",skgti.core.IrDiGraph())


#print(p)
#print("Ordered nodes:",p.get_ordered_nodes())

iso1={0:'A',1:'B',2:'C',4:'D'}
#skgti.io.plot_graph_links(p,p_ref,[skgti.io.matching2links(iso1)],['red']);plt.show()
iso2={0:'A',1:'B',2:'D',3:'C'}
#skgti.io.plot_graph_links(p,p_ref,[skgti.io.matching2links(iso2)],['red']);plt.show()

iso3={0:'A',1:'B',3:'C',4:'D'}
#skgti.io.plot_graph_links(p,p_ref,[skgti.io.matching2links(iso3)],['red']);plt.show()

print("ISO:",iso1)
print("Validity ISO1:",skgti.core.check_iso_eligibility(iso1,p.get_ordered_nodes(),skgti.core.find_groups_of_brothers(p_ref)))
skgti.io.plot_graph_links(p,p_ref,[skgti.io.matching2links(iso1)],['red']);plt.show()


print("ISO:",iso2)
print("Validity ISO2:",skgti.core.check_iso_eligibility(iso2,p.get_ordered_nodes(),skgti.core.find_groups_of_brothers(p_ref)))
skgti.io.plot_graph_links(p,p_ref,[skgti.io.matching2links(iso2)],['red']);plt.show()

print("ISO:",iso3)
print("Validity ISO3:",skgti.core.check_iso_eligibility(iso3,p.get_ordered_nodes(),skgti.core.find_groups_of_brothers(p_ref)))
skgti.io.plot_graph_links(p,p_ref,[skgti.io.matching2links(iso3)],['red']);plt.show()