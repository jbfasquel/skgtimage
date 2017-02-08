import os,csv,pickle,time
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti
from skgtimage.utils import grey_levels
from skgtimage.core.graph import rename_nodes,transitive_closure
from skgtimage.core.filtering import remove_smallest_regions,size_filtering,merge_filtering,rag_merge_until_commonisomorphism,merge_photometry_gray
from skgtimage.core.subisomorphism import find_subgraph_isomorphims,best_common_subgraphisomorphism,common_subgraphisomorphisms,common_subgraphisomorphisms_optimized,common_subgraphisomorphisms_optimized_v2
from skgtimage.core.factory import from_labelled_image



def save_context(save_dir,image,label,t_desc,p_desc,t,p):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    skgti.io.save_graphregions(t, save_dir)
    skgti.io.save_graph(t, directory=save_dir)
    ref_t = skgti.core.from_string(t_desc)
    skgti.io.save_graph(ref_t, "graph_ref", directory=save_dir)
    skgti.io.with_graphviz.__save_image2d__(label.astype(np.uint8),save_dir + "label.png", True)

def concatenate(root_save_dir,filename,all_tests=None):
    ###################
    # IF ALL TESTS IS NONE
    ###################
    if all_tests is None:
        all_tests=[]
        for f in os.listdir(root_save_dir):
            if os.path.isdir(root_save_dir+f): all_tests+=[f+"/"]
    ###################
    #Prepare final csv
    ###################
    csv_file=open(root_save_dir+filename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    #First file
    test=all_tests[0]
    tmp_csv_file=open(root_save_dir+test+filename, 'r')
    tmp_c_reader = csv.reader(tmp_csv_file, dialect='excel')
    c_writer.writerow(next(tmp_c_reader))
    c_writer.writerow(next(tmp_c_reader))
    tmp_csv_file.close()
    for i in range(1,len(all_tests)):
        test = all_tests[i]
        if not os.path.exists(root_save_dir + test + filename):
            c_writer.writerow([test,"Err"])
            continue

        tmp_csv_file = open(root_save_dir + test + filename, 'r')
        tmp_c_reader = csv.reader(tmp_csv_file, dialect='excel')
        next(tmp_c_reader)
        c_writer.writerow(next(tmp_c_reader))
    #
    csv_file.close()

def influence_of_commonisos(matcher,t_desc,p_desc,truth_dir,save_dir,slices=None):
    image=matcher.image

    result_dir=save_dir+"commoniso_influence/"
    if not os.path.exists(result_dir) : os.mkdir(result_dir)

    truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
    res2int=skgti.io.compute_intensitymap(truth_t_graph,do_round=True)
    truth_image=skgti.io.generate_single_image(truth_t_graph,res2int)
    head=list(skgti.core.find_head(truth_t_graph))[0]
    roi=skgti.core.fill_region(truth_t_graph.get_region(head))
    l_truth_image=np.ma.array(truth_image, mask=np.logical_not(roi))

    #Reference result
    matcher.relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
    result_image=skgti.io.generate_single_image(matcher.relabelled_final_t_graph,res2int)
    l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))
    ref_classif=skgti.utils.goodclassification_rate(l_result_image,l_truth_image)
    ref_classif=np.round(ref_classif,3)


    performances=[]
    for i in range(0,len(matcher.common_isomorphisms)):
        print("Testing iso number ",i,"/",len(matcher.common_isomorphisms))
        current_matching=matcher.common_isomorphisms[i]
        #tmp_directory
        tmp_dir=result_dir+"iso_"+str(i)+"/" #;if not os.path.exists(tmp_dir) : os.mkdir(tmp_dir)
        matching_links=skgti.io.matching2links(current_matching)
        skgti.io.save_graph_links(matcher.t_graph,matcher.ref_t_graph,[matching_links],['red'],name="common_iso_t",directory=tmp_dir,tree=True)
        skgti.io.save_graph_links(matcher.p_graph,matcher.ref_p_graph,[matching_links],['red'],name="common_iso_p",directory=tmp_dir,tree=True)

        final_t_graph,final_p_graph,ordered_merges=skgti.core.propagate(matcher.t_graph,matcher.p_graph,matcher.ref_t_graph,matcher.ref_p_graph,current_matching,verbose=True)
        skgti.io.save_graph_links(matcher.t_graph,matcher.ref_t_graph,[matching_links,ordered_merges],['red','green'],label_lists=[[],range(0,len(ordered_merges)+1)],name="common_iso_merge_t",directory=tmp_dir,tree=True)
        skgti.io.save_graph_links(matcher.p_graph,matcher.ref_p_graph,[matching_links,ordered_merges],['red','green'],label_lists=[[],range(0,len(ordered_merges)+1)],name="common_iso_merge_p",directory=tmp_dir,tree=True)


        (relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],current_matching)
        relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
        # GENERATING IMAGES COMBINING REGIONS RESIDUES WITH SPECIFICS INTENSITIES
        result_image=skgti.io.generate_single_image(relabelled_final_t_graph,res2int)
        l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))
        #####
        # SAVING IMAGES
        if len(result_image.shape)==2:
            sp.misc.imsave(tmp_dir+"_image.png",image.astype(np.uint8))
            sp.misc.imsave(tmp_dir+"_truth.png",truth_image.astype(np.uint8))
            sp.misc.imsave(tmp_dir+"_result.png",result_image.astype(np.uint8))
            skgti.io.with_graphviz.__save_image2d__(result_image.astype(np.uint8),tmp_dir + "_result_rescaled.png", True)
            sp.misc.imsave(tmp_dir+"_diff.png",generate_absdiff_inunint8(result_image,truth_image))
            #Crop
            result_image_cropped = skgti.utils.extract_subarray(result_image, roi=roi)
            skgti.io.with_graphviz.__save_image2d__(result_image_cropped.astype(np.uint8), tmp_dir + "_result_rescaled_crop.png",True)
            diffe = generate_absdiff_inunint8(result_image,truth_image)
            diffe_cropped = skgti.utils.extract_subarray(diffe, roi=roi)
            skgti.io.with_graphviz.__save_image2d__(diffe_cropped.astype(np.uint8),tmp_dir + "_diff_crop.png", True)

        if len(result_image.shape)==3:
            for s in slices:
                sp.misc.imsave(tmp_dir+"_image_"+str(s)+".png",slice2png(image,s))
                sp.misc.imsave(tmp_dir+"_truth_"+str(s)+".png",slice2png(truth_image,s))
                sp.misc.imsave(tmp_dir+"_result_"+str(s)+".png",slice2png(result_image,s))

        #####
        # COMPUTING THE CLASSIFICATION RATE
        classif=skgti.utils.goodclassification_rate(l_result_image,l_truth_image)
        classif=np.round(classif,3)
        performances+=[classif]
        #####
        # COMPUTING THE SIMILARITY INDEX FOR EACH REGION
        region2sim=skgti.utils.compute_sim_between_graph_regions(relabelled_final_t_graph,truth_t_graph)
        nodepersim=[]
        related_sims=[]
        for n in sorted(region2sim):
            sim=np.round(region2sim[n],3)
            nodepersim+=[n]
            related_sims+=[sim]
        #####
        # SAVING THE RESULT AS A .CSV FILE
        skgti.utils.save_to_csv(tmp_dir,"result_cmpvs_truth",classif,related_sims,nodepersim)


    #####
    # SAVING TO CSV
    is_ref_best=True
    for p in performances:
        if p > ref_classif: is_ref_best=False

    nb_worst=0
    for p in performances:
       if p != ref_classif:  nb_worst+=1
    percentage_of_good = 100*(len(performances)-nb_worst)/len(performances)

    best_gcr=max(performances)
    worst_gcr = min(performances)

    import csv
    fullfilename=os.path.join(result_dir,"Classif_vs_commoniso.csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    if is_ref_best is True:
        c_writer.writerow(["Reference result (min/max eie): "]+[ref_classif]+["Best one"])
    else:
        c_writer.writerow(["Reference result (min/max eie): "] + [ref_classif] + ["NOT the best one"])
    c_writer.writerow(["Percentage of good result"]+[np.round(percentage_of_good,1)])
    c_writer.writerow(["Result for each commoniso"])
    c_writer.writerow(['Eie']+[np.round(i,2) for i in matcher.eies])
    c_writer.writerow(['GCR']+[np.round(i,3) for i in performances])
    csv_file.close()

    return is_ref_best,np.round(percentage_of_good,1),len(performances),best_gcr,worst_gcr

def generate_absdiff_inunint8(result,ref):
    diff_result=np.abs(result.astype(np.float)-ref.astype(np.float))
    diff_result=np.round(diff_result,2)
    diff_result=np.where(diff_result!=0.0,255,0).astype(np.uint8)
    #diff_result=255*diff_result/np.max(diff_result)
    return diff_result.astype(np.uint8)

def slice2png(image,slice):
    mini=np.min(image)
    maxi=np.max(image)
    image2D=image[:,:,slice].astype(np.float)
    image2D=np.rot90(255*(image2D+mini)/(maxi-mini))
    return image2D.astype(np.uint8)


def compared_with_rawsegmentation(segmentation_filename,t_desc,p_desc,image,region2segmentintensities,result_dir,truth_dir,comparison_dir):
    #Preparing data
    segmentation=sp.misc.imread(segmentation_filename)
    truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
    related_truth=skgti.utils.combine_refactorying(truth_t_graph,region2segmentintensities)
    result_t_graph,result_p_graph=skgti.io.from_dir(t_desc,p_desc,image,result_dir)
    related_result=skgti.utils.combine_refactorying(result_t_graph,region2segmentintensities)

    #Comparison
    if not os.path.exists(comparison_dir) : os.mkdir(comparison_dir)

    head=list(skgti.core.find_head(truth_t_graph))[0]
    roi=skgti.core.fill_region(truth_t_graph.get_region(head))
    l_related_truth=np.ma.array(related_truth, mask=np.logical_not(roi))
    l_related_result=np.ma.array(related_result, mask=np.logical_not(roi))
    l_segmentation=np.ma.array(segmentation, mask=np.logical_not(roi))

    #####
    # COMPUTING THE CLASSIFICATION RATE
    classif_result=skgti.utils.goodclassification_rate(l_related_result,l_related_truth)
    classif_result=np.round(classif_result,3)
    classif_rawsegmentation=skgti.utils.goodclassification_rate(l_segmentation,l_related_truth)
    classif_rawsegmentation=np.round(classif_rawsegmentation,3)

    #####
    # COMPUTING SIMILARITIES
    intensities2regions={}
    for id in region2segmentintensities:
        intensity=region2segmentintensities[id]
        if intensity in intensities2regions: intensities2regions[intensity]+="+"+id
        else: intensities2regions[intensity]=id

    regions=[]
    sims_raw=[]
    sims_result=[]
    for l in intensities2regions:
            regions+=[intensities2regions[l]]
            true_region=np.where(related_truth==l,1,0)
            seg_region=np.where(segmentation==l,1,0)
            sims_raw+=[np.round(skgti.utils.similarity_index(seg_region,true_region),3)]
            seg_result=np.where(related_result==l,1,0)
            sims_result+=[np.round(skgti.utils.similarity_index(seg_result,true_region),3)]

    #####
    # SAVING TO CSV
    fullfilename=os.path.join(comparison_dir,"classifrates.csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow(["Comparison type","good classification rate (1.0==100%)"])
    c_writer.writerow(['From result (our method)']+[classif_result])
    c_writer.writerow(['From raw segmentation']+[classif_rawsegmentation])
    c_writer.writerow(['related ids']+[i for i in regions])
    c_writer.writerow(['similarities (raw vs truth)']+[i for i in sims_raw])
    c_writer.writerow(['similarities (our result vs truth)']+[i for i in sims_result])

    csv_file.close()
    #####
    # SAVING IMAGES
    sp.misc.imsave(comparison_dir+"raw_segmentation.png",segmentation.astype(np.uint8))
    c_segmentation=skgti.utils.extract_subarray(segmentation,roi)
    sp.misc.imsave(comparison_dir+"raw_segmentation_cropped.png",c_segmentation.astype(np.uint8))

    sp.misc.imsave(comparison_dir+"obtained_result.png",related_result.astype(np.uint8))
    c_related_result=skgti.utils.extract_subarray(related_result,roi)
    sp.misc.imsave(comparison_dir+"obtained_result_cropped.png",c_related_result.astype(np.uint8))

    sp.misc.imsave(comparison_dir+"truth.png",related_truth.astype(np.uint8))
    c_related_truth=skgti.utils.extract_subarray(related_truth,roi)
    sp.misc.imsave(comparison_dir+"truth_cropped.png",c_related_truth.astype(np.uint8))

    #diff
    sp.misc.imsave(comparison_dir+"diff_obtained_result_vs_truth.png",generate_absdiff_inunint8(c_related_result,c_related_truth))
    sp.misc.imsave(comparison_dir+"diff_rawsegmentation_vs_truth.png",generate_absdiff_inunint8(c_segmentation,c_related_truth))
    sp.misc.imsave(comparison_dir+"diff_obtained_result_vs_rawsegmentation.png",generate_absdiff_inunint8(c_segmentation,c_related_result))

    return classif_result,classif_rawsegmentation

def save_stats(recognizer,result,save_dir,name,runtime):
    csv_file = open(save_dir+"stats.csv", "w")
    c_writer = csv.writer(csv_file, dialect='excel')
    #### DIRECTORY
    row_title=["Directory"]
    row_values=[name]
    #### NAME ANALYSIS
    import re
    split_name=re.split("_",name)
    row_title+=["Image"];
    row_values+=[split_name[0]]
    row_title+= ["ROI"];row_values += [split_name[2]]
    row_title+= ["Seg."];row_values += [split_name[3]]
    seg_param = split_name[4].replace("/", "")
    ###################################
    ##### #Label, #Nodes, #Isos...
    #nb_init_labels = len(grey_levels(recognizer.label, roi=recognizer.roi))
    nb_init_labels = len(grey_levels(recognizer.intermediate_labels[0], roi=recognizer.roi))
    row_title += ["#Labels"];row_values += [nb_init_labels]
    nb_nodes=len(recognizer.t_graph.nodes())
    row_title += ["#Nodes"];row_values += [nb_nodes]
    t_isomorphisms_candidates = find_subgraph_isomorphims(transitive_closure(recognizer.t_graph),transitive_closure(recognizer.ref_t_graph))
    nb_t_isos = len(t_isomorphisms_candidates)
    row_title += ["#I-isos"];
    row_values += [nb_t_isos]
    nb_c_isos=len(recognizer.common_isomorphisms)
    row_title += ["#C-isos"];
    row_values += [nb_c_isos]

    ###################################
    #### GCR, SIMILARITIES
    if "GCR" in result:
        #GCR
        row_title += ["GCR"];row_values+=[result["GCR"]]
        #Similarities
        region2sim = result["Similarities"]
        ids_sim = [n for n in sorted(region2sim)]
        val_sims = [round(region2sim[n], 2) for n in sorted(region2sim)]
        string_val_ids=""
        for i in val_sims: string_val_ids+=str(i)+","
        string_val_ids=string_val_ids[0:-1] #remove last ','
        string_val_ids+='\n('
        for i in ids_sim: string_val_ids += i+","
        string_val_ids = string_val_ids[0:-1]  # remove last ','
        string_val_ids+=")"
        row_title += ["Similarities"];row_values += [string_val_ids]
        #Min/Max similarities
        row_title += ["Similarities (max,min)"]
        maxi=max(val_sims)
        maxi_ids=[]
        for m in range(0,len(val_sims)):
            if maxi == val_sims[m]: maxi_ids+=[ids_sim[m]]
        string_min_max=str(maxi)+" ("
        for id in maxi_ids:
            string_min_max+=(id+",")
        string_min_max=string_min_max[0:-1]
        string_min_max+=") "
        #
        mini=min(val_sims)
        mini_ids=[]
        for m in range(0,len(val_sims)):
            if mini == val_sims[m]: mini_ids+=[ids_sim[m]]
        string_min_max+=str(mini)+" ("
        for id in mini_ids:
            string_min_max+=(id+",")
        string_min_max=string_min_max[0:-1]
        string_min_max+=")"

        row_values+=[string_min_max]
    ###################################
    #### ISO INFLUENCE
    if "Is best iso" in result: #"Worst GCR"
        is_best=result["Is best iso"]
        if is_best:
            row_title += ["Best iso ?"];
            row_values += ["Y ("+str(result["Best GCR"])+"-"+str(result["Worst GCR"])+")"]
            row_title += ["Percentage"];
            row_values += [result["Percentage of good iso"]]
        else:
            row_values += ["N ("+str(result["Best GCR"])+"-"+str(result["Worst GCR"])+")"]
            row_title += ["Percentage"];
            row_values += ["-"]

    ###################################
    ##### MISC
    #rag_merging=(recognizer.t_graph_before_rag is not None)
    #prefil= (recognizer.label_pre_rag is not None) or (recognizer.label_pre_photomerge is not None)
    prefil=""
    for i in recognizer.intermediate_operations:
        prefil+=i+" - "
    filtered_size=recognizer.size_min
    bg_removal=recognizer.remove_background
    row_title+=["Seg param","Prefilt", "min size ?", "rm bg ?"]
    row_values+=[seg_param,prefil,filtered_size,bg_removal]

    ###################################
    ##### WRITE ROW
    c_writer.writerow(row_title)
    c_writer.writerow(row_values)
    csv_file.close()

def save_stats_runtime(recognizer,result,save_dir,name,seg_runtime):
    ###################
    #Runtimes
    ###################
    #Hack: rerunning graph building after preprocessing steps
    image,labelled,roi=recognizer.image,recognizer.t_graph.get_labelled(),recognizer.roi
    t0=time.clock()
    t,p=from_labelled_image(image,labelled,roi)
    t1 = time.clock()
    #CVS file
    csv_file = open(save_dir + "stats_runtime.csv", "w")
    c_writer = csv.writer(csv_file, dialect='excel')
    title_row=[]
    value_row=[]
    #### DIRECTORY
    title_row=["Directory"]
    value_row=[name]
    #### NAME ANALYSIS
    import re
    split_name=re.split("_",name)
    title_row+=["Image"];
    value_row+=[split_name[0]]
    title_row+= ["ROI"];value_row += [split_name[2]]
    title_row+= ["Seg."];value_row += [split_name[3]]
    seg_param = split_name[4].replace("/", "")

    if seg_runtime is not None:
        title_row += ["Seg"]
        value_row += [np.round(seg_runtime,2)]
    title_row+=["Build"]
    value_row+=[np.round(t1-t0,2)]
    for r in sorted(recognizer.action2runtime):
        title_row+=[r]
        runtime=recognizer.action2runtime[r]
        value_row+=[np.round(runtime,2)]
    c_writer.writerow(title_row)
    c_writer.writerow(value_row)
    csv_file.close()




def analyze(matcher,save_dir,name="",runtime=None,truth_dir=None,iso_influence=False,slices=[],full=False):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    #skgti.utils.save_recognizer_report(matcher,save_dir,name,runtime)
    skgti.utils.save_recognizer_details(matcher,save_dir,full,slices)
    result=None
    if truth_dir is not None:
        result = evaluate(truth_dir, save_dir + "Evaluation/", iso_influence=iso_influence, matcher=matcher,result_dir=save_dir,slices=slices)
        '''
        #Write to cvs file
        csv_file = open(os.path.join(save_dir, "stats_performances.csv"), "w")
        c_writer = csv.writer(csv_file, dialect='excel')
        region2sim=result["Similarities"]
        ids_sim=[n for n in sorted(region2sim)]
        val_sims=[round(region2sim[n], 2) for n in sorted(region2sim)]
        if iso_influence:
            c_writer.writerow(["Context"]+["GCR"]+ids_sim+["Is best iso ?"]+["Percentage good iso"]+["Nb iso"]+["Best GCR"]+["Worst GCR"])
            c_writer.writerow([name]+[result["GCR"]]+val_sims+[result["Is best iso"],result["Percentage of good iso"],result["Number of isos"],result["Best GCR"],result["Worst GCR"]])
        else:
            c_writer.writerow(["Context"]+["GCR"]+ids_sim)
            c_writer.writerow([name]+[result["GCR"]]+val_sims)

        csv_file.close()
        #New: save all in a single file
        '''
    save_stats(matcher,result,save_dir,name,runtime)
    save_stats_runtime(matcher,result,save_dir,name,runtime)


    return result


def evaluate(truth_dir,comparison_dir,iso_influence=False,matcher=None,result_dir=None,image=None,mc=False,t_desc=None,p_desc=None,slices=[]):
    """
    Examples of invokations:
    result = helper.evaluate(truth_dir, save_dir+"Evaluation/",iso_influence = True, matcher = recognizer, result_dir=save_dir, image = image, mc = False, t_desc = t_desc, p_desc = p_desc, slices = [])
    result = helper.evaluate(truth_dir, save_dir + "Evaluation/", iso_influence=True, matcher=recognizer,result_dir=save_dir)
    :param truth_dir:
    :param comparison_dir:
    :param iso_influence:
    :param matcher:
    :param result_dir:
    :param image:
    :param mc:
    :param t_desc:
    :param p_desc:
    :param slices:
    :return:
    """
    if not os.path.exists(comparison_dir) : os.mkdir(comparison_dir)
    skgti.utils.clear_dir_content(comparison_dir)
    if (matcher is not None) and (image is None):
        return evaluate(truth_dir, comparison_dir, iso_influence, matcher, result_dir, image=matcher.image, mc = False, t_desc = matcher.t_desc, p_desc = matcher.p_desc , slices = slices)
    if (image is not None) and (mc==True):
        tmp=0.2125*image[:,:,0]+ 0.7154*image[:,:,1]+0.0721*image[:,:,2]
        return evaluate(truth_dir, comparison_dir, iso_influence, matcher, result_dir, image=tmp, mc = False, t_desc = t_desc, p_desc = p_desc , slices = slices)
        #return evaluate(tmp,t_desc,p_desc,truth_dir,result_dir,comparison_dir,slices,mc=False,iso_influence=iso_influence,matcher=matcher)

    #####
    # LOADING TRUTH AND OBTAINED GRAPHS
    truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
    result_t_graph,result_p_graph=skgti.io.from_dir(t_desc,p_desc,image,result_dir)

    #####
    # GENERATING IMAGES COMBINING REGIONS RESIDUES WITH SPECIFICS INTENSITIES
    res2int=skgti.io.compute_intensitymap(truth_t_graph,do_round=True)
    truth_image=skgti.io.generate_single_image(truth_t_graph,res2int)
    result_image=skgti.io.generate_single_image(result_t_graph,res2int)

    if len(image.shape)==2:
        sp.misc.imsave(comparison_dir+"full_truth.png",truth_image.astype(np.uint8))
        sp.misc.imsave(comparison_dir+"full_result.png",result_image.astype(np.uint8))
        skgti.io.with_graphviz.__save_image2d__(result_image.astype(np.uint8),comparison_dir+"full_result_rescaled.png",True)
        sp.misc.imsave(comparison_dir + "full_diff.png",generate_absdiff_inunint8(result_image, truth_image))
    if len(image.shape)==3:
        for s in slices:
            sp.misc.imsave(comparison_dir+"full_truth_"+str(s)+".png",slice2png(truth_image,s))
            sp.misc.imsave(comparison_dir+"full_result_"+str(s)+".png",slice2png(result_image,s))
            sp.misc.imsave(comparison_dir+"full_diff.png",generate_absdiff_inunint8(slice2png(result_image,s), slice2png(truth_image,s)))

    #####
    # GENERATING MASKED IMAGES
    head=list(skgti.core.find_head(truth_t_graph))[0]
    roi=skgti.core.fill_region(truth_t_graph.get_region(head))
    l_truth_image=np.ma.array(truth_image, mask=np.logical_not(roi))
    l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))

    if len(image.shape)==2:
        c_truth_image=skgti.utils.extract_subarray(truth_image,roi)
        sp.misc.imsave(comparison_dir+"crop_truth.png",c_truth_image.astype(np.uint8))
        c_result_image=skgti.utils.extract_subarray(result_image,roi)
        sp.misc.imsave(comparison_dir+"crop_result.png",c_result_image.astype(np.uint8))
        sp.misc.imsave(comparison_dir+"crop_diff.png",generate_absdiff_inunint8(c_result_image,c_truth_image))
    #####
    # COMPUTING THE CLASSIFICATION RATE
    classif=skgti.utils.goodclassification_rate(l_result_image,l_truth_image)
    classif=np.round(classif,3)

    #####
    # COMPUTING THE SIMILARITY INDEX FOR EACH REGION
    region2sim=skgti.utils.compute_sim_between_graph_regions(result_t_graph,truth_t_graph)
    nodepersim=[]
    related_sims=[]
    for n in sorted(region2sim):
        sim=np.round(region2sim[n],3)
        nodepersim+=[n]
        related_sims+=[sim]

    #####
    # SAVING THE RESULT AS A .CSV FILE
    skgti.utils.save_to_csv(comparison_dir,"gcr_similarities",classif,related_sims,nodepersim)

    #####
    # RETURN RESULTS
    result={"GCR":classif,"Similarities":region2sim}
    if iso_influence and (matcher is not None):
        is_ref_best, percentage_of_good, nb_isos,best_gcr,worst_gcr = influence_of_commonisos(matcher, t_desc, p_desc, truth_dir,comparison_dir, slices=slices)
        result["Is best iso"] = is_ref_best
        result["Percentage of good iso"] = percentage_of_good
        result["Number of isos"] = nb_isos
        result["Best GCR"] = best_gcr
        result["Worst GCR"] = worst_gcr

    return result


def pickle_isos(save_dir,prefixe,matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist):
    tmp=[matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist]
    matching_file=open(save_dir+prefixe+'all_isos.pkl', 'wb')
    pickle.dump(tmp,matching_file)
    matching_file.close()

def unpickle_isos(save_dir,prefixe):
    matching_file=open(save_dir+prefixe+'all_isos.pkl', 'rb')
    [matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist]=pickle.load(matching_file)
    return matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist

def pickle_isos2(save_dir,prefixe,common_isomorphisms):
    tmp=common_isomorphisms
    matching_file=open(save_dir+prefixe+'all_isos.pkl', 'wb')
    pickle.dump(tmp,matching_file)
    matching_file.close()

def unpickle_isos2(save_dir,prefixe):
    matching_file=open(save_dir+prefixe+'all_isos.pkl', 'rb')
    common_isomorphisms=pickle.load(matching_file)
    return common_isomorphisms

def save_refinement_historization(save_dir,historization,ref_t_graph,ref_p_graph,matching):
    for i in range(0,len(historization)):
        context=historization[i]
        current_t_graph=context[0]
        current_p_graph=context[1]
        current_matching=context[2]
        save_matching_details(save_dir,"04_t_refinement_step_"+str(i),current_t_graph,ref_t_graph,matching=matching)
        save_matching_details(save_dir,"04_p_refinement_step_"+str(i),current_p_graph,ref_p_graph,matching=matching)

def save_built_graphs(save_dir,prefixe,t_graph,p_graph,residues=None,slices=None):
    if not os.path.exists(save_dir) : os.mkdir(save_dir)
    if residues is not None:
        nodes=t_graph.nodes()
        for i in range(0,len(nodes)):
            filled_r=skgti.core.fill_region(residues[i])
            t_graph.set_region(nodes[i],filled_r)
            p_graph.set_region(nodes[i],filled_r)
    skgti.io.save_graph("topo",t_graph,nodes=None,tree=True,directory=save_dir+prefixe+"built_t_graph",save_regions=True)
    skgti.io.save_graph("photo",p_graph,nodes=None,tree=True,directory=save_dir+prefixe+"built_p_graph",save_regions=True)

    if residues is not None:
        if len(residues[0].shape) > 2:
            for n in t_graph.nodes():
                region=t_graph.get_region(n)
                np.save(save_dir+prefixe+"built_t_graph/"+"region_"+str(n)+".npy",region)
                save_3d_slices(save_dir,prefixe+"built_t_graph/"+str(n),region,slices=slices)
    if residues is None:
        any_node=t_graph.nodes()[0]
        any_region=t_graph.get_region(any_node)
        if any_region is not None:
            if len(any_region.shape) > 2:
                for n in t_graph.nodes():
                    region=t_graph.get_region(n)
                    np.save(save_dir+prefixe+"built_t_graph/"+"region_"+str(n)+".npy",region)
                    save_3d_slices(save_dir,prefixe+"built_t_graph/"+str(n),region,slices=slices)

    if t_graph.get_image() is not None:
        t_graph.update_intensities(t_graph.get_image())
        res2int=skgti.io.compute_intensitymap(t_graph,do_round=True)
        image_with_allregions=skgti.io.generate_single_image(t_graph,res2int)
        if len(image_with_allregions.shape) == 2:
            sp.misc.imsave(save_dir+prefixe+"built_t_graph"+"/all_regions.png",image_with_allregions.astype(np.uint8))
        else:
            np.save(save_dir+prefixe+"built_t_graph/"+"all_regions.npy",image_with_allregions)
            save_3d_slices(save_dir,prefixe+"built_t_graph/all_regions",image_with_allregions,slices=slices)

def save_3d_slices(save_dir,prefixe,image,slices=None):
    mini=np.min(image)
    maxi=np.max(image)
    output_dir=save_dir+prefixe+"/"
    if not os.path.exists(output_dir) : os.mkdir(output_dir)
    if type(image)==np.ma.MaskedArray:
        roi=np.logical_not(image.mask)
        for s in slices:
            image2D=image[:,:,s].astype(np.float)
            roi2D=roi[:,:,s]
            image2D=skgti.utils.extract_subarray(image2D,roi=roi2D)
            image2D=np.rot90(255*(image2D+mini)/(maxi-mini))
            sp.misc.imsave(output_dir+"slice_"+str(s)+".png",image2D.astype(np.uint8))
    else:
        for s in slices:
            image2D=image[:,:,s].astype(np.float)
            image2D=np.rot90(255*(image2D+mini)/(maxi-mini))
            sp.misc.imsave(output_dir+"slice_"+str(s)+".png",image2D.astype(np.uint8))

def save_initial_context(save_dir,prefixe,image,segmentation,t_graph,p_graph,slice_index=0):
    if not os.path.exists(save_dir) : os.mkdir(save_dir)
    if len(image.shape) <= 2:
        max_seg=np.max(segmentation)
        sp.misc.imsave(save_dir+prefixe+"_segmentation.png",round(255.0/max_seg)*(segmentation.astype(np.uint8)))
        max_img=np.max(image)
        sp.misc.imsave(save_dir+prefixe+"_image.png",round(255.0/max_img)*(image.astype(np.uint8)))
        plt.imshow(image,cmap="gray",vmin=np.min(image),vmax=np.max(image),interpolation="nearest");plt.axis('off');
        plt.savefig(save_dir+prefixe+"_matplotlib_image.png");plt.gcf().clear()
        plt.imshow(segmentation,cmap="gray",vmin=np.min(segmentation),vmax=np.max(segmentation),interpolation="nearest");plt.axis('off');
        plt.savefig(save_dir+prefixe+"_matplotlib_segmentation.png");plt.gcf().clear()

    skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+prefixe+'_a_priori_topo.png');plt.gcf().clear()
    skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+prefixe+'_a_priori_photo.png');plt.gcf().clear()


def save_matching_details(save_dir,prefixe,built_graph,ref_graph,matching=None,common_isomorphisms=None,all_isomorphisms=None,energies=None):
    if not os.path.exists(save_dir) : os.mkdir(save_dir)
    #All isomorphisms
    if all_isomorphisms is not None:
        for i in range(0,len(all_isomorphisms)):
            iso=all_isomorphisms[i]
            skgti.io.plot_graphs_matching([ref_graph],[built_graph],iso,titles=None)
            plt.savefig(save_dir+prefixe+'_iso'+str(i)+'.png');plt.gcf().clear()

    #Only common isomorphism
    if common_isomorphisms is not None:
        for i in range(0,len(common_isomorphisms)):
            iso=common_isomorphisms[i]
            skgti.io.plot_graphs_matching([ref_graph],[built_graph],iso,titles=None)
            plt.savefig(save_dir+prefixe+'_common_iso'+str(i)+'.png');plt.gcf().clear()

    #Matching
    if matching is not None:
        skgti.io.plot_graphs_matching([ref_graph],[built_graph],matching,titles=None)
        plt.savefig(save_dir+prefixe+'_matching.png');plt.gcf().clear()

    #Energies: [sim,dist]
    if energies is not None:
        eie_sim=energies[0]
        eie_dist=energies[1]
        csv_file=open(save_dir+prefixe+"_energies.csv", "w")
        c_writer = csv.writer(csv_file,dialect='excel')
        c_writer.writerow(["Isomorphism"]+["Sim"]+["dist"])
        for i in range(0,len(eie_sim)):
            c_writer.writerow(["iso "+str(i)]+[eie_sim[i]]+[eie_dist[i]])
        csv_file.close()


class Slicer:
    def __init__(self,image_3d,slice_index=0):
        self.slice_index=slice_index
        self.image_3d=image_3d
    def mouse(self,event):
        if event.button == "up": ds=1
        elif event.button == "down": ds=-1
        else: ds=0
        self.slice_index+=ds
        if self.slice_index < 0: self.slice_index = 0
        if self.slice_index == self.image_3d.shape[2]: self.slice_index=self.image_3d.shape[2]-1

        image2D=self.image_3d[:,:,self.slice_index]

        plt.imshow(np.rot90(image2D),'gray')
        plt.title("Slice "+str(self.slice_index))
        plt.gcf().canvas.draw()

def plot(image3D,slice_index=0):
    slicer=Slicer(image3D,slice_index=slice_index)
    plt.gcf().canvas.mpl_connect('scroll_event', slicer.mouse)
    slice=image3D[:,:,slice_index]
    plt.imshow(np.rot90(slice),'gray');plt.axis('off')
    plt.title("Slice "+str(slice_index))
    plt.show()
