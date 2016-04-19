#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import numpy as np
from sklearn import cluster
from skgtimage.utils.color import hsv2chsv


###########################################################################################################################################
# GENERIC CONSTRAINED SEED GENERATOR
###########################################################################################################################################
class KMeansSeeder:
    def __init__(self,data,n_clusters,constraints=None,mc=False,projection_functor=None):
        """
        MC: Multicomponent image
        Seeder
        data: masked image or not, multidimensional (e.g. grayscale (n_features=1) rgb (n_features=3))
        #Example:
        seeder=skit.KMeansSeeder(l_image_hsv.astype(np.float),nb_classes,n_features=3)
        seeder.seed_projection_functor=skit.cartesian_hsv_points
        #Example with constraints
        np.random.seed()
        seeder=skit.KMeansSeeder(l_image_hsv.astype(np.float),nb_classes,n_features=3)
        seeder.seed_projection_functor=skit.cartesian_hsv_points
        tmp_image_hsv=scipy.misc.imread(os.path.join(dir,root_name+".jpeg"))

        r_white,residue_data=build_residue(dir,tmp_image_hsv,[root_name+"_region1.png"],sub=[root_name+"_region2.png",root_name+"_region3.png",root_name+"_region6.png"],n_features=3,factor=factor,stat=np.mean)
        r_black,residue_data=build_residue(dir,tmp_image_hsv,[root_name+"_region2.png",root_name+"_region3.png"],sub=[root_name+"_region4.png",root_name+"_region5.png"],n_features=3,factor=factor,stat=np.mean)
        r_colors,residue_data=build_residue(dir,tmp_image_hsv,[root_name+"_region4.png",root_name+"_region5.png",root_name+"_region6.png"],n_features=3,factor=factor,stat=np.mean)
        r_colors_and_white,residue_data=build_residue(dir,tmp_image_hsv,[root_name+"_region1.png"],sub=[root_name+"_region2.png",root_name+"_region3.png"],re_add=[root_name+"_region4.png",root_name+"_region5.png"],n_features=3,factor=factor,stat=np.mean)
        r_yellow,residue_data=build_residue(dir,tmp_image_hsv,[root_name+"_region5.png",root_name+"_region7.png"],n_features=3,factor=factor,stat=np.mean)
        r_red,residue_data=build_residue(dir,tmp_image_hsv,[root_name+"_region4.png"],n_features=3,factor=factor,stat=np.mean)

        #Constraint for class 0,9 (blanc) = r1
        seeder.add_constraint([r_white[0],r_white[1],r_colors_and_white[2]])
        #Constraint for class 8,12 (noir) = r2+r33
        seeder.add_constraint(r_black)
        #Constraint for class 10 (yellow)
        seeder.add_constraint([r_yellow[0],r_colors[1],r_colors_and_white[2]])
        #Constraint for class 10 (red)
        seeder.add_constraint([r_red[0],r_colors[1],r_colors_and_white[2]])
        """
        if mc :
            n_features=data.shape[-1] #the array size along the last dimension is the number of features.
        else:
            n_features=1
        #PREPARE DATA
        if type(data) == np.ma.masked_array:
            self.data=data.compressed().reshape(-1,n_features)
        else:
            self.data=data.flatten().reshape(-1,n_features)
        #clusters
        self.n_clusters=n_clusters
        #seeding: constraints, static seeds, random seed taken from a subdataset
        self.constraints={}
        self.subdataset_randomseed={}
        #functor allowing to transform seeds, e.g. from hsv (space within which contraints are defined) to cartesian_hsv
        self.seed_projection_functor=projection_functor
        if constraints is not None:
            self.set_constraints(constraints)
        #FEATURES
        #self.n_features=n_features
        #self.static_seed={}

    def detail_info(self):
        answer=str(self)
        for s in self.subdataset_randomseed:
            data=self.subdataset_randomseed[s]
            answer+=str(np.transpose(data))+"\n"
        return answer

    def __str__(self):
        answer="A priori number of clusters: " + str(self.n_clusters) + "\n"
        answer+="Number of points: " + str(self.data.shape[0]) + "\n"
        n_features=self.data.shape[1]
        answer+="Number of features: " + str(n_features) + "\n"
        answer+="Constraints:\n"
        for c in sorted(self.constraints.keys()):
            answer+= "Index " + str(c) + " --> " + str(self.constraints[c])+"\n"
        #answer+="Static seeds:\n"
        #for c in sorted(self.static_seed.keys()):
        #    answer+= "Index " + str(c) + " --> " + str(self.static_seed[c])+"\n"
        answer+="Random seeds from subdataset:\n"
        for c in sorted(self.subdataset_randomseed.keys()):
            answer+= "Index " + str(c) + " --> Number of points : " + str(self.subdataset_randomseed[c].shape[0])+"\n"
        return answer

    def set_constraints(self,constraints):
        for s in constraints.values(): self.add_constraint(s)

    def add_constraint(self,constraint):
        """
        [value1,value2,...,value_n_features]: the seed will the closest point to constraint in self.data
        Note:  [value1,None,...,None], any point whose feature correspond to a "None" constraint is candidate for seeding if other features match "UnNone" constraints
        Note: for ndg image (n_features=1), interval-based seeding must be specified by add_constraint([[min,max]])
        Note: add_constraint([[min,max]]) with min > max -> candidate will verify value > min or value < max
        e.g. add_constraint([[10,5]]) -> all point being either higher than 10 or lower than 5
        """
        #Store the constraint
        #self.n_constraints+=1
        n_constraints=len(self.constraints)+1
        self.constraints[n_constraints]=constraint
        #Analyze the constraint is order to define either this constraint leads to a
        #1- a static seed (if constraint contains at least a value) -> self.static_seed[n_constraints]= "this seed"
        #2- a random seed taken from the reduced data (subdataset, if constraint contains at least an interval) -> self.subdataset_randomseed[n_constraints]= "the subdataset"
        subdata=np.copy(self.data)
        dimensions_with_static_cst=[]
        #FIRST CONSTRAINT PARSING TO REDUCE DE SUBDATASET
        for dimension in range(0,len(constraint)):
            #print "sub data set"
            min_boundary=constraint[dimension][0]
            max_boundary=constraint[dimension][1]
            #[a,b] with a < b
            if min_boundary < max_boundary :
                subdata=subdata[np.where(subdata[:,dimension] >= min_boundary)]
                subdata=subdata[np.where(subdata[:,dimension] <= max_boundary)]
            #[a,b] with b < a : we keep values > a and values < b
            elif min_boundary > max_boundary:
                first_subset=subdata[np.where(subdata[:,dimension] > min_boundary)]
                second_subset=subdata[np.where(subdata[:,dimension] < max_boundary)]
                subdata=np.concatenate((first_subset,second_subset))
            #[a,b] with a=B : "static seed" -> the closest point will be kept
            elif min_boundary == max_boundary:
                dimensions_with_static_cst+=[dimension]

        #STATIC SEED (MAY BE FROM THE SUBDATA)
        if dimensions_with_static_cst != []:
            cumulative_dist=None
            for dimension in dimensions_with_static_cst:
                static_value=constraint[dimension][0]
                if cumulative_dist is None:
                    cumulative_dist=np.abs(subdata[:,dimension]-static_value)
                else:
                    cumulative_dist+=np.abs(subdata[:,dimension]-static_value)
                #if type(current_constraint[dimension])!=list :
            seed=subdata[np.argmin(cumulative_dist),:]
            self.subdataset_randomseed[n_constraints]=np.array([seed])
            #self.static_seed[n_constraints]=seed
            #print "seed : " , seed
            #return seed
        #SUBDATASET RANDOM SEED
        else :
            self.subdataset_randomseed[n_constraints]=subdata


    def random_seed(self):
        """
        return an sample (a row) of self.data, being a [n_samples,n_features] array
        """
        indice=np.random.randint(0,self.data.shape[0])
        seed=np.array(self.data[indice])
        return seed
    """
    def generate_static(self):
        list_of_seeds=[]
        for i in self.static_seed.keys():
            list_of_seeds+=[self.static_seed[i]]
        return np.asarray(list_of_seeds).reshape(-1,self.n_features)
    def generate_random(self):
        list_of_seeds=[]
        for i in self.subdataset_randomseed.keys():
            subdata=self.subdataset_randomseed[i]
            indice=np.random.randint(0,subdata.shape[0])
            seed=np.array(subdata[indice])
            list_of_seeds+=[seed]
        return np.asarray(list_of_seeds).reshape(-1,self.n_features)
    """
    def generate(self):
        """
        doc
        """
        n_features=self.data.shape[1]
        if len(self.constraints)==0 :
            list_of_seeds=[]
            for s in range(0,self.n_clusters):
                list_of_seeds+=[self.random_seed()]
            seeds=np.asarray(list_of_seeds).reshape(-1,n_features)
        else :
            if len(self.constraints) != self.n_clusters :
                raise Exception("The number of constraints differ from the number of clusters")
            list_of_seeds=[]
            '''
            for i in self.static_seed.keys():
                list_of_seeds+=[self.static_seed[i]]
            '''
            for i in self.subdataset_randomseed.keys():
                subdata=self.subdataset_randomseed[i]
                indice=np.random.randint(0,subdata.shape[0])
                seed=np.array(subdata[indice])
                list_of_seeds+=[seed]
            seeds=np.asarray(list_of_seeds).reshape(-1,n_features)
        #Eviter de generate 2 similar seeds
        '''
        if n_features ==1 :
            are_similar_values=False
            list_of_value=seeds.flatten().tolist()
            for p in list_of_value:
                if list_of_value.count(p) > 1 : are_similar_values=True
            if are_similar_values == True:
                return self.generate()
        '''
        #Projection is relevant
        if self.seed_projection_functor is not None:
            return self.seed_projection_functor(seeds)
        else:
            return seeds



def kmeans(image,nb_clusters,n_seedings=100,seeder=None,intervals=None,mc=False,tol=0.0001,fct=None,verbose=False):
    """
    Performs n_seedings clusterings, keeping as result the one minimizing the moment of inertia
    If seeder is None, the default "random" seeding is considered (scikits learn default method)
    If seeder is not None, kmeans is initialized each one with centroids returned by seeder.generate() method
    Manage nD image (spatial_dim), begin grayscale (n_features=1) or rgb (n_features=3)

    """
    #Multi-component image
    if mc : nb_components=image.shape[-1] #the array size along the last dimension is the number of features.
    else:   nb_components=1

    if type(image) == np.ma.masked_array :
        reshaped_data=image.compressed()
        reshaped_data=reshaped_data.reshape(-1,nb_components).astype(np.float)
    else:
        data=image.flatten()
        reshaped_data=data.reshape(len(data),nb_components).astype(np.float)

    if fct is not None: reshaped_data=hsv2chsv(reshaped_data)


    if (seeder is None) and (intervals is not None): seeder=KMeansSeeder(image,nb_clusters,intervals,mc=mc,projection_functor=fct)
    inertia=None
    labels=None
    for seeding in range(0,n_seedings):
        if verbose : print("random seeding " , seeding+1 , " / " , n_seedings)
        if seeder is not None:
            centroids=seeder.generate()
            #KMEANS SINGLE RUN
            k=cluster.KMeans(n_clusters=nb_clusters,init=centroids,n_init=1,verbose=0,max_iter=300,tol=tol) #single random centroid initialization
        if seeder is None:
            k=cluster.KMeans(n_clusters=nb_clusters,init="random",n_init=1,verbose=0,max_iter=300,tol=tol) #single random centroid initialization

        k.fit(reshaped_data)
        if inertia is None : inertia = k.inertia_ #for the first step
        if k.inertia_ <= inertia :
            labels=k.labels_

    #### Writing labels to appropriate pixels: Version 1
    spatial_dim=len(image.shape)
    if mc : spatial_dim-=1
    result = np.zeros(image.shape[0:spatial_dim],dtype=np.uint8) #ne marche pas avec color -> vu comme 3D

    if type(image) == np.ma.masked_array :
        roi=np.logical_not(image.mask)
    else: roi=np.ones(image.shape,dtype=np.bool)
    if (spatial_dim == 2) & (nb_components>1): #cas 2D RGB
        roi=roi[:,:,0]
    elif (spatial_dim == 2) & (nb_components==1): #cas 2D grayscale
        roi=roi
    elif (spatial_dim == 3) & (nb_components==1): #cas 2D grayscale
        roi=roi

    #roi=np.dsplit(roi, 2)
    result[roi] = labels
    result=np.ma.masked_array(result, mask=np.logical_not(roi))

    return result


