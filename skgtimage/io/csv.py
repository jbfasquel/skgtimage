import os,csv

def save_intensities(graph,directory=None,filename="intensities"):
    if not os.path.exists(directory) : os.mkdir(directory)
    csv_file=open(os.path.join(directory,filename+".csv"), "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    for n in graph.nodes():
        c_writer.writerow([n] + [graph.get_mean_intensity(n)])
    csv_file.close()
