import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import skimage; from skimage import filters
import helper


#########
# A PRIORI KNOWLEDGE
#########
t_desc="2E<2D;2G<F;2D,F,2H,2I<C<B<A;C,1D<B<A;1E,1G<1D;1H<1E"
p_desc="B=F=2D=2H=1E=1G<2I=2E=1H=1D<C=2G=A"
t_ref=skgti.core.from_string(t_desc)
p_ref=skgti.core.from_string(p_desc)

#########
# GLOBAL INFORMATIONS
#########
ref_dir="../../Database/image03/"
algo_name="MS_ROI_None"
seg_para=0.1 #0.09 good for [1-5]


#########
# CSV PREPARATION
#########
tmp=skgti.core.from_string(p_desc)
import csv
csv_file = open(os.path.join(ref_dir, algo_name+".csv"), "w")
c_writer = csv.writer(csv_file, dialect='excel')
c_writer.writerow([algo_name,str(seg_para)])
c_writer.writerow(['Pose','GCR']+sorted(tmp.nodes())+["is best iso"]+['% good c_iso'])

#########
# MISC INFORMATIONS
#########
for pose_id in range(1,8):
    print("PROCESSING POSE", pose_id)
    root_dir=ref_dir+"pose_"+str(pose_id)+"/downsampled3/"
    truth_dir=root_dir+"truth_simplified/"
    save_dir=root_dir+"results/"+algo_name+str(seg_para)+"/"
    if not os.path.exists(root_dir+"results/"): os.mkdir(root_dir+"results/")
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    #########
    # IMAGE + ROI
    #########
    image=sp.misc.imread(os.path.join(truth_dir,"image.png"))
    roi=None #skgti.core.fill_region(sp.misc.imread(os.path.join(truth_dir,"region_A.png")))
    #########
    # OVERSEGMENTATION
    #########
    label=skgti.utils.mean_shift(image,seg_para,roi,True,True,sigma=0.5,rgb_convert=True)
    skgti.io.save_image2d_boundaries(image,label,save_dir,"0_init_")

    #########
    # RAG MERGE IF NECESSARY
    #########
    try:
        post_fix_rag=""
        t,p=skgti.core.from_labelled_image(skgti.utils.rgb2gray(image),label,roi)
        common_isomorphisms = skgti.core.common_subgraphisomorphisms_optimized_v2([t, p], [t_ref, p_ref])
        if len(common_isomorphisms) == 0:
            post_fix_rag="(rag)"
            t,p=skgti.core.rag_merge_until_commonisomorphism(t,p,t_ref,p_ref)
            label=t.get_labelled()
            skgti.io.save_image2d_boundaries(image,label,save_dir,"0_after_rag")

    except:
        c_writer.writerow(['Pose ' + str(pose_id), 'Err'])
        continue

    #########
    # RECOGNITION
    #########
    try :
        # RECOGNITION
        id2r, matcher = skgti.core.recognize_regions(image, label, t_desc, p_desc, roi=roi, verbose=True,filter_size=20,background=True, mc=True)  # 7 labels
        skgti.io.save_matcher_result(matcher,image,label,roi,save_dir,mc=True)
        # EVALUATION VS TRUTH
        classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/",mc=True)
        print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)

        #ISO INFLUENCE
        is_best,perc=helper.influence_of_commonisos(matcher, image, t_desc, p_desc, truth_dir, save_dir, mc=True)

        sims = [round(region2sim[n], 2) for n in sorted(region2sim)]
        c_writer.writerow(['Pose ' + str(round(pose_id, 2)) + post_fix_rag, classif] + sims+[is_best]+[perc])

    except skgti.core.matcher_exception as m_e:
        print('Matcher exception:', m_e.matcher)
        print("impossible to finalize")

        c_writer.writerow(['Pose ' + str(pose_id), 'Err'])

#########
# CLOSING FILE
#########
csv_file.close()
