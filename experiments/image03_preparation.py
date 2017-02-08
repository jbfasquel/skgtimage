import os,re
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import skimage as sk;from skimage import exposure
import matplotlib.pyplot as plt
import skgtimage as skgti

########################################
# SCRIPT CONFIGURATION
########################################
#root_dir="../../Database/image03/pose_1/"
#root_dir="../../Database/image03/pose_2/"
#root_dir="../../Database/image03/pose_3/"
root_dir="../../Database/image03/pose_4/"
#root_dir="../../Database/image03/pose_5/"
#root_dir="../../Database/image03/pose_6/"
#root_dir="../../Database/image03/pose_7/" #For 7, comment "clear_truth_dir"


t_desc="1C,1D<1B<1A;1F<1C;1G,1E<1D;1H<1E;2C<2B<2A;2D,2H,2F,2I<2C;2E<2D;2G<2F"
p_desc="1G=1B=1E=1F=2B=2F=2H=2D<1H=1D=2E=2I<1A=1C=2G=2A=2C"




########################################
# HELPER FUNCTIONS
########################################
def clear_truth_dir(t_desc,p_desc,input_dir,output_dir):
    '''
    Applying matching method to clear unperfect truth due to downsampling

    :param t_desc:
    :param p_desc:
    :param input_dir:
    :param output_dir:
    :return:
    '''
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    #Save image
    image = sp.misc.imread(os.path.join(input_dir, "image.png"))
    sp.misc.imsave(os.path.join(output_dir, "image.png"), image)

    #Applying matching method to clear
    t, p = skgti.core.from_dir(input_dir, True)
    label = np.zeros(t.get_image().shape,dtype=np.uint8)
    value = 1
    for n in t.nodes():
        region = t.get_region(n)
        label = np.ma.masked_array(label, mask=region).filled(value)
        value += 1

    roi = skgti.core.fill_region(t.get_region('1A')) + skgti.core.fill_region(t.get_region('2A'))
    id2r, matcher = skgti.core.recognize_regions(t.get_image(), label, t_desc, p_desc, roi=roi, verbose=True,mc=False)  # 7 labels
    for i in id2r:
        sp.misc.imsave(os.path.join(output_dir, "region_"+ i +".png"), id2r[i])

    skgti.io.save_graph(matcher.relabelled_final_t_graph, name="graph_t", directory=output_dir)
    skgti.io.save_graph(matcher.relabelled_final_p_graph, name="graph_p", directory=output_dir)


def show(image,roi):
    roied_image = np.ma.masked_array(image,mask=np.logical_not(np.dstack(tuple([roi for i in range(0, 3)])))).filled(0)
    plt.imshow(roied_image)
    plt.show()

def holes(region):
    region=255*(region/np.max(region)).astype(np.uint8)
    res = 255 * skgti.core.fill_region(region).astype(np.uint8) - region
    return res

def inter(A,B):
    return 255*np.logical_and(A, B).astype(np.uint8)

def fill(A):
    return 255 * skgti.core.fill_region(A).astype(np.uint8)

def save_graph_from_truth(input_dir,output_dir):
    t, p = skgti.core.from_dir(input_dir, True)
    skgti.io.save_graph(t, name="graph_t",directory=output_dir);skgti.io.save_graph(p, name="graph_p",directory=output_dir)
    skgti.io.save_graphregions(t, output_dir)
    return t,p

def correctBvsA(region_2A,region_2B):
    region_2B -= inter(region_2A, region_2B)  # removal pixel of B covering A
    region_2B += holes(region_2B + region_2A) - holes(region_2B)  # outer boundary of B connected to inner boundary of A
    return region_2B

def get_largest_cc(region):
    labelled_image, nb_labels = sp.ndimage.measurements.label(region)
    # Calcul histo des etiquettes
    histo, bins = sk.exposure.histogram(labelled_image)
    # Calcul de l'indice du maximum, en ignorant le fond (indice 0)
    # In short: "indice_of_max=bins[np.argmax(histo[1:])+1]"
    indice_of_max, value_of_max = 0, 0
    for i in range(1, len(bins)):
        if histo[i] > value_of_max:
            value_of_max = histo[i]
            indice_of_max = bins[i]
    # Extraction de la plus grand composante connexe
    largest_cc = np.where(labelled_image == indice_of_max, 255, 0)
    return largest_cc

def correct2H2I(region_2F,region_2H,region_2I):
    tmp_region_2F = fill(region_2F)
    region_2H -= inter(tmp_region_2F, region_2H)
    region_2H_filled2F = region_2H + tmp_region_2F
    region_2H += holes(region_2H_filled2F)
    tmp_region_2H=fill(region_2H)
    region_2I-=inter(tmp_region_2H,region_2I)
    region_2I_filled2H=region_2I+tmp_region_2H
    region_2I+=holes(region_2I_filled2H)
    return region_2H,region_2I


def model_merging(input_dir,output_dir):
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    image = sp.misc.imread(os.path.join(input_dir, "image.png"))
    sp.misc.imsave(os.path.join(output_dir, "image.png"), image)



    # Simplified model
    t, p = skgti.core.from_dir(input_dir, True)
    # 1A+2A
    skgti.core.merge_nodes_topology(t, '2A', '1A')
    skgti.core.merge_nodes_photometry(p, '2A', '1A')
    (t, p) = skgti.core.rename_nodes([t, p], {'1A': 'A'})
    # 1B+2B
    skgti.core.merge_nodes_topology(t, '2B', '1B')
    skgti.core.merge_nodes_photometry(p, '2B', '1B')
    (t, p) = skgti.core.rename_nodes([t, p], {'1B': 'B'})
    # 1C+2C
    skgti.core.merge_nodes_topology(t, '2C', '1C')
    skgti.core.merge_nodes_photometry(p, '2C', '1C')
    (t, p) = skgti.core.rename_nodes([t, p], {'1C': 'C'})
    # 1F+2F
    skgti.core.merge_nodes_topology(t, '2F', '1F')
    skgti.core.merge_nodes_photometry(p, '2F', '1F')
    (t, p) = skgti.core.rename_nodes([t, p], {'1F': 'F'})
    # Export
    skgti.io.save_graph(t, name="graph_t", directory=output_dir);
    skgti.io.save_graph(p, name="graph_p", directory=output_dir)
    skgti.io.save_graphregions(t, output_dir)




manual_dir=root_dir+"full_resolution/manual/"
truth_dir=root_dir+"full_resolution/truth/"
if not os.path.exists(truth_dir) : os.mkdir(truth_dir)
image=sp.misc.imread(os.path.join(manual_dir,"image.png"))
sp.misc.imsave(os.path.join(truth_dir,"image.png"),image)

############################################################################################################
#
#           TOP REGIONS 1A,1B, .... 1H
#
############################################################################################################
region_1A=sp.misc.imread(os.path.join(manual_dir,"region_1A.png"))
region_1A=get_largest_cc(region_1A)
region_1B=sp.misc.imread(os.path.join(manual_dir,"region_1B.png"))
region_1B=get_largest_cc(region_1B)
region_1G=sp.misc.imread(os.path.join(manual_dir,"region_1G.png"))
region_1G=fill(region_1G)
region_2B=correctBvsA(region_1A,region_1B)
sp.misc.imsave(os.path.join(truth_dir,"region_1A.png"),region_1A)
sp.misc.imsave(os.path.join(truth_dir,"region_1B.png"),region_2B)
sp.misc.imsave(os.path.join(truth_dir,"region_1G.png"),region_1G)

region_1E=sp.misc.imread(os.path.join(manual_dir,"region_1E.png"))
sp.misc.imsave(os.path.join(truth_dir,"region_1E.png"),region_1E)
region_1H=holes(region_1E)
sp.misc.imsave(os.path.join(truth_dir,"region_1H.png"),region_1H)


region_1F=sp.misc.imread(os.path.join(manual_dir,"region_1F.png"))
region_1F=fill(region_1F)
sp.misc.imsave(os.path.join(truth_dir,"region_1F.png"),region_1F)


background_region_1D=holes(region_1B)
background_region_1D=get_largest_cc(background_region_1D)

region_1C=holes(region_1B+region_1F)-background_region_1D
sp.misc.imsave(os.path.join(truth_dir,"region_1C.png"),region_1C)

region_1D=background_region_1D-region_1G-fill(region_1E)
sp.misc.imsave(os.path.join(truth_dir,"region_1D.png"),region_1D)

############################################################################################################
#
#           BOTTOM REGIONS 2A,2B,...,2I
#
############################################################################################################
#INPUTS 2A, 2B
region_2A=sp.misc.imread(os.path.join(manual_dir,"region_2A.png"))
region_2A=get_largest_cc(region_2A)
sp.misc.imsave(os.path.join(truth_dir,"region_2A.png"),region_2A)
region_2B=sp.misc.imread(os.path.join(manual_dir,"region_2B.png"))
region_2B=get_largest_cc(region_2B)
region_2B=correctBvsA(region_2A,region_2B)
sp.misc.imsave(os.path.join(truth_dir,"region_2B.png"),region_2B)

region_2D=sp.misc.imread(os.path.join(manual_dir,"region_2D.png"))
sp.misc.imsave(os.path.join(truth_dir,"region_2D.png"),region_2D)
region_2E=holes(region_2D)
region_2E=fill(region_2E)
sp.misc.imsave(os.path.join(truth_dir,"region_2E.png"),region_2E)
region_2D-=inter(region_2E,region_2D)
sp.misc.imsave(os.path.join(truth_dir,"region_2D.png"),region_2D)

region_2F=sp.misc.imread(os.path.join(manual_dir,"region_2F.png"))
sp.misc.imsave(os.path.join(truth_dir,"region_2F.png"),region_2F)
region_2G=holes(region_2F)
sp.misc.imsave(os.path.join(truth_dir,"region_2G.png"),region_2G)


#Manage bandeau 2H and 2I versus 2F (to avoid overlapping and holes at boundaries)
region_2H=sp.misc.imread(os.path.join(manual_dir,"region_2H.png"))
region_2H=fill(region_2H)
region_2I=sp.misc.imread(os.path.join(manual_dir,"region_2I.png"))
region_2I=fill(region_2I)
region_2H,region_2I=correct2H2I(region_2F,region_2H,region_2I)
sp.misc.imsave(os.path.join(truth_dir,"region_2H.png"),region_2H)
sp.misc.imsave(os.path.join(truth_dir,"region_2I.png"),region_2I)

#Region C
region_2C=holes(region_2B)-fill(region_2D)-fill(region_2F)-region_2H-region_2I
sp.misc.imsave(os.path.join(truth_dir,"region_2C.png"),region_2C)
#Save auto generated graphs
save_graph_from_truth(truth_dir,truth_dir+"control/")

#skgti.io.plot_graph_histogram(t)
############################################################################################################
#
#           SIMPLIFIED MODEL TRUTH BY MERGING: A=1A+2A, B=1B+2B, C=1C+2C, F=1F+2F
#
############################################################################################################
model_merging(truth_dir,root_dir+"full_resolution/truth_simplified/")

############################################################################################################
#
#           DOWNSAMPLING
#
############################################################################################################
down=3
tmp_dir = root_dir + "downsampled" + str(down) + "/"
if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)


tmp_truth_dir_down = tmp_dir + "tmp_truth/"
if not os.path.exists(tmp_truth_dir_down): os.mkdir(tmp_truth_dir_down)
image = sp.misc.imread(truth_dir + "image.png")
image = image[::down, ::down, :]
sp.misc.imsave(tmp_truth_dir_down + "image.png", image)
for f in os.listdir(truth_dir):
    print(f)
    if re.match('^region.*png$', f) is not None:
        region = sp.misc.imread(os.path.join(truth_dir, f))
        region = region[::down, ::down]
        #Hack
        '''
        if (root_dir == "../../Database/image03/pose_4/") and (f == "region_1G.png"):
            region=fill(region)
        '''
        sp.misc.imsave(tmp_truth_dir_down + f, region)
save_graph_from_truth(tmp_truth_dir_down, tmp_truth_dir_down+"control/")


truth_dir_down = tmp_dir + "truth/"
clear_truth_dir(t_desc,p_desc,tmp_truth_dir_down,truth_dir_down)
save_graph_from_truth(truth_dir_down, truth_dir_down+"control/")


############################################################################################################
#
#           DOWNSAMPLED SIMPLIFIED MODEL TRUTH BY MERGING: A=1A+2A, B=1B+2B, C=1C+2C, F=1F+2F
#
############################################################################################################
model_merging(truth_dir_down,tmp_dir+"truth_simplified/")
save_graph_from_truth(tmp_dir+"truth_simplified", tmp_dir+"truth_simplified/"+"control/")