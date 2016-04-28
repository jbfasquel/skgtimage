import os,csv,pickle
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti


def influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir):
    result_dir=save_dir+"08_eie_influence/"
    if not os.path.exists(result_dir) : os.mkdir(result_dir)

    truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
    res2int=skgti.io.compute_intensitymap(truth_t_graph,do_round=True)
    truth_image=skgti.io.generate_single_image(truth_t_graph,res2int)
    head=list(skgti.core.find_head(truth_t_graph))[0]
    roi=truth_t_graph.get_region(head)
    l_truth_image=np.ma.array(truth_image, mask=np.logical_not(roi))

    #Reference result
    matcher.relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
    result_image=skgti.io.generate_single_image(matcher.relabelled_final_t_graph,res2int)
    l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))
    ref_classif=skgti.utils.goodclassification_rate(l_result_image,l_truth_image)
    ref_classif=np.round(ref_classif,3)


    performances=[]
    for i in range(0,len(matcher.common_isomorphisms)):
        current_matching=matcher.common_isomorphisms[i]
        #tmp_directory
        tmp_dir=result_dir+"iso_"+str(i)+"/" #;if not os.path.exists(tmp_dir) : os.mkdir(tmp_dir)
        matching_links=skgti.io.matching2links(current_matching)
        skgti.io.save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="common_iso_t",directory=tmp_dir,tree=True)
        skgti.io.save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="common_iso_p",directory=tmp_dir,tree=True)

        final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(matcher.query_t_graph,
                                                                      matcher.query_p_graph,
                                                                      matcher.ref_t_graph,
                                                                      matcher.ref_p_graph,current_matching)

        ordered_merges=[i[2] for i in histo]
        skgti.io.save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links,ordered_merges],['red','green'],name="common_iso_merge_t",directory=tmp_dir,tree=True)
        skgti.io.save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links,ordered_merges],['red','green'],name="common_iso_merge_p",directory=tmp_dir,tree=True)


        (relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],current_matching)
        relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
        # GENERATING IMAGES COMBINING REGIONS RESIDUES WITH SPECIFICS INTENSITIES
        result_image=skgti.io.generate_single_image(relabelled_final_t_graph,res2int)
        l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))
        #####
        # COMPUTING THE CLASSIFICATION RATE
        classif=skgti.utils.goodclassification_rate(l_result_image,l_truth_image)
        classif=np.round(classif,3)
        performances+=[classif]

        '''
        try :
            #tmp_directory
            tmp_dir=result_dir+"iso_"+str(i)+"/" #;if not os.path.exists(tmp_dir) : os.mkdir(tmp_dir)
            matching_links=skgti.io.matching2links(current_matching)
            skgti.io.save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="common_iso_t",directory=tmp_dir,tree=True)
            skgti.io.save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="common_iso_p",directory=tmp_dir,tree=True)

            final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(matcher.query_t_graph,
                                                                          matcher.query_p_graph,
                                                                          matcher.ref_t_graph,
                                                                          matcher.ref_p_graph,current_matching)

            ordered_merges=[i[2] for i in histo]
            skgti.io.save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links,ordered_merges],['red','green'],name="common_iso_merge_t",directory=tmp_dir,tree=True)
            skgti.io.save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links,ordered_merges],['red','green'],name="common_iso_merge_p",directory=tmp_dir,tree=True)


            (relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],current_matching)
            relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
            # GENERATING IMAGES COMBINING REGIONS RESIDUES WITH SPECIFICS INTENSITIES
            result_image=skgti.io.generate_single_image(relabelled_final_t_graph,res2int)
            l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))
            #####
            # COMPUTING THE CLASSIFICATION RATE
            classif=skgti.utils.goodclassification_rate(l_result_image,l_truth_image)
            classif=np.round(classif,3)
            performances+=[classif]
        except Exception as e:
            print("exception ",e)
            performances+=["Failed"]
        '''
    #####
    # SAVING TO CSV
    import csv
    fullfilename=os.path.join(result_dir,"08_classif_vs_commoniso.csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow(["Reference result (min/max eie): "]+[ref_classif])
    c_writer.writerow(["Result for each commoniso"])
    c_writer.writerow(['Eie dist']+[i for i in matcher.eie_dist])
    c_writer.writerow(['Eie sim']+[i for i in matcher.eie_sim])
    c_writer.writerow(['GCR']+[i for i in performances])
    csv_file.close()

    print("Eies dis: ",matcher.eie_dist)
    print("Eies sim: ",matcher.eie_sim)
    print("performances: ",performances)


def influence_of_commonisos(image,common_isomorphisms,eie_dist,eie_sim,built_t_graph,built_p_graph,t_graph,p_graph,t_desc,p_desc,input_dir,save_dir):
    performances=[]
    for i in range(0,len(common_isomorphisms)):
        current_matching=common_isomorphisms[i]
        try :
            final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(built_t_graph,built_p_graph,t_graph,p_graph,current_matching)
            (relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],current_matching)
            relabelled_final_t_graph.set_image(image) #hack to save mixed region residues

            save_built_graphs(save_dir,"06_relabelled_"+str(i)+"_",relabelled_final_t_graph,relabelled_final_p_graph)
            classif,region2sim=compared_with_truth(image,t_desc,p_desc,input_dir,save_dir+"06_relabelled_"+str(i)+"_built_t_graph",save_dir+"07_eval_classif_"+str(i)+"/")
            #print("Eie dis: ",eie_dist[i]," - Eie sim: ", eie_sim[i], " --> classif: " , classif)
            performances+=[classif]
        except Exception as e:
            print("exception ",e)
            performances+=["Failed"]

    #####
    # SAVING TO CSV
    import csv
    fullfilename=os.path.join(save_dir,"06_classif_vs_commoniso.csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow(["Result for each commoniso"])
    c_writer.writerow(['Eie dist']+[i for i in eie_dist])
    c_writer.writerow(['Eie sim']+[i for i in eie_sim])
    c_writer.writerow(['GCR']+[i for i in performances])
    csv_file.close()

    print("Eies dis: ",eie_dist)
    print("Eies sim: ",eie_sim)
    print("performances: ",performances)


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


'''
def compared_with_rawsegmentation(segmentation,related_result,related_truth,truth_t_graph,comparison_dir,levels=None,ids=None):
    if not os.path.exists(comparison_dir) : os.mkdir(comparison_dir)

    head=list(skgti.core.find_head(truth_t_graph))[0]
    roi=truth_t_graph.get_region(head)
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
    sims=None
    if levels is not None:
        sims=[]
        for i in range(0,len(levels)):
            l=levels[i]
            true_region=np.where(related_truth==l,1,0)
            seg_region=np.where(segmentation==l,1,0)
            sim=skgti.utils.similarity_index(seg_region,true_region)
            sim=np.round(sim,3)
            sims+=[sim]



    #####
    # SAVING TO CSV
    fullfilename=os.path.join(comparison_dir,"classifrates.csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow(["Comparison type","good classification rate (1.0==100%)"])
    c_writer.writerow(['From result (our method)']+[classif_result])
    c_writer.writerow(['From raw segmentation']+[classif_rawsegmentation])
    if sims is not None:
        c_writer.writerow(['similarities']+[i for i in sims])
        c_writer.writerow(['related ids']+[i for i in ids])
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
'''
def compared_with_rawsegmentation_refactorying(segmentation_filename,t_desc,p_desc,image,region2segmentintensities,result_dir,truth_dir,comparison_dir):
    #Preparing data
    segmentation=sp.misc.imread(segmentation_filename)
    truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
    related_truth=skgti.utils.combine_refactorying(truth_t_graph,region2segmentintensities)
    result_t_graph,result_p_graph=skgti.io.from_dir(t_desc,p_desc,image,result_dir)
    related_result=skgti.utils.combine_refactorying(result_t_graph,region2segmentintensities)



    #Comparison
    if not os.path.exists(comparison_dir) : os.mkdir(comparison_dir)

    head=list(skgti.core.find_head(truth_t_graph))[0]
    roi=truth_t_graph.get_region(head)
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
            sims_raw+=[skgti.utils.similarity_index(seg_region,true_region)]
            seg_result=np.where(related_result==l,1,0)
            sims_result+=[skgti.utils.similarity_index(seg_result,true_region)]

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


def compared_with_truth(image_gray,t_desc,p_desc,truth_dir,result_dir,comparison_dir,slices=None):
    if not os.path.exists(comparison_dir) : os.mkdir(comparison_dir)
    #####
    # LOADING TRUTH AND OBTAINED GRAPHS
    truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image_gray,truth_dir)
    result_t_graph,result_p_graph=skgti.io.from_dir(t_desc,p_desc,image_gray,result_dir)

    #####
    # GENERATING IMAGES COMBINING REGIONS RESIDUES WITH SPECIFICS INTENSITIES
    res2int=skgti.io.compute_intensitymap(truth_t_graph,do_round=True)
    truth_image=skgti.io.generate_single_image(truth_t_graph,res2int)
    result_image=skgti.io.generate_single_image(result_t_graph,res2int)

    if len(image_gray.shape)==2:
        sp.misc.imsave(comparison_dir+"_image.png",image_gray.astype(np.uint8))
        sp.misc.imsave(comparison_dir+"_truth.png",truth_image.astype(np.uint8))
        sp.misc.imsave(comparison_dir+"_result.png",result_image.astype(np.uint8))
    if len(image_gray.shape)==3:
        for s in slices:
            sp.misc.imsave(comparison_dir+"_image_"+str(s)+".png",slice2png(image_gray,s))
            sp.misc.imsave(comparison_dir+"_truth_"+str(s)+".png",slice2png(truth_image,s))
            sp.misc.imsave(comparison_dir+"_result_"+str(s)+".png",slice2png(result_image,s))





    #####
    # GENERATING MASKED IMAGES
    head=list(skgti.core.find_head(truth_t_graph))[0]
    roi=truth_t_graph.get_region(head)
    l_truth_image=np.ma.array(truth_image, mask=np.logical_not(roi))
    l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))

    if len(image_gray.shape)==2:
        c_image=skgti.utils.extract_subarray(image_gray,roi)
        sp.misc.imsave(comparison_dir+"_image_crop.png",c_image.astype(np.uint8))
        c_truth_image=skgti.utils.extract_subarray(truth_image,roi)
        sp.misc.imsave(comparison_dir+"_truth_crop.png",c_truth_image.astype(np.uint8))
        c_result_image=skgti.utils.extract_subarray(result_image,roi)
        sp.misc.imsave(comparison_dir+"_result_crop.png",c_result_image.astype(np.uint8))

        sp.misc.imsave(comparison_dir+"diff_obtained_result_vs_truth.png",generate_absdiff_inunint8(c_result_image,c_truth_image))
    #####
    # COMPUTING THE CLASSIFICATION RATE
    classif=skgti.utils.goodclassification_rate(l_result_image,l_truth_image)
    classif=np.round(classif,3)

    #####
    # COMPUTING THE SIMILARITY INDEX FOR EACH REGION
    region2sim=skgti.utils.compute_sim_between_graph_regions(result_t_graph,truth_t_graph)
    nodepersim=[]
    related_sims=[]
    for n in region2sim:
        sim=np.round(region2sim[n],3)
        nodepersim+=[n]
        related_sims+=[sim]

    #####
    # SAVING THE RESULT AS A .CSV FILE
    skgti.utils.save_to_csv(comparison_dir,"result_cmpvs_truth",classif,related_sims,nodepersim)

    #####
    # RETURN
    return classif,region2sim


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
        print("Merge:",current_matching)



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
