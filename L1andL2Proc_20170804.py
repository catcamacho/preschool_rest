
# coding: utf-8

# In[1]:


#import packages
from os import listdir
from nipype.interfaces.io import DataSink, SelectFiles, DataGrabber # Data i/o
from nipype.interfaces.utility import IdentityInterface, Function     # utility
from nipype.pipeline.engine import Node, Workflow, JoinNode        # pypeline engine

from nipype.interfaces.fsl.model import Randomise, GLM, Cluster
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.fsl.utils import ImageMeants, Merge, Split
from nipype.interfaces.fsl.maths import ApplyMask

#set output file type for FSL to NIFTI
from nipype.interfaces.fsl.preprocess import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Set study variables
#studyhome = '/Users/catcamacho/Box/FFnHK-Oddball/RESTINGSTATE'
studyhome = '/home/camachocm2/Box_home/CARS_rest'
raw_data = studyhome + '/raw'
preproc_dir = studyhome + '/proc/preproc'
output_dir = studyhome + '/proc/analysis'
workflow_dir = studyhome + '/workflows'
roi_dir = studyhome + '/ROIs'
group_con = studyhome + '/misc/tcon.con'
group_mat = studyhome + '/misc/design.mat'
proc_cores = 2

#subjects_list = ['101']
subjects_list = open(studyhome + '/misc/subjects.txt').read().splitlines()

template_brain = studyhome + '/templates/MNI152_T1_2mm_brain.nii'

# ROIs for connectivity analysis
Lamyg = roi_dir + '/L_amyg_anatomical.nii'
Ramyg = roi_dir + '/R_amyg_anatomical.nii'

ROIs = [Lamyg, Ramyg]
rois = ['L_amyg','R_amyg']

min_clust_size = 25


# In[2]:


## File handling
# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id','ROIs']),
                     name='infosource')
infosource.iterables = [('subject_id', subjects_list),('ROIs',ROIs)]


# Data grabber- select fMRI and ROIs
templates = {'orig_func': preproc_dir + '/smoothed_filt_func/{subject_id}/func_filtered_smooth.nii'}
selectfiles = Node(SelectFiles(templates), name='selectfiles')

# Datasink- where our select outputs will go
datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir
substitutions = [('_subject_id_', ''),
                ('_ROIs_..home..camachocm2..Box_home..CARS_rest..ROIs..','')]
datasink.inputs.substitutions = substitutions


# In[3]:


## Seed-based level 1

# Extract ROI timeseries
ROI_timeseries = Node(ImageMeants(), name='ROI_timeseries', iterfield='mask')

def converthex(orig):
    from numpy import genfromtxt, savetxt
    from os.path import abspath
    
    orig = genfromtxt(orig, delimiter='  ', dtype=None, skip_header=0)
    new = 'func_roi_ts.txt'
    savetxt(new, orig, delimiter='  ')
    
    new_file = abspath(new)
    return(new_file)

converthex = Node(name='converthex', 
                  interface=Function(input_names=['orig'], 
                                     output_names=['new_file'], 
                                     function=converthex))

# model ROI connectivity
glm = Node(GLM(out_file='betas.nii',out_cope='cope.nii'), name='glm', iterfield='design')


# In[4]:


sbc1_workflow = Workflow(name='sbc1_workflow')
sbc1_workflow.connect([(infosource,selectfiles,[('subject_id','subject_id')]),
                       (selectfiles,ROI_timeseries,[('orig_func','in_file')]),
                       (infosource,ROI_timeseries,[('ROIs','mask')]),
                       (ROI_timeseries,converthex,[('out_file','orig')]),
                       (converthex,glm,[('new_file','design')]),
                       (selectfiles,glm,[('orig_func','in_file')]),
                       (converthex, datasink, [('new_file','roi_ts')]),
                       (glm,datasink,[('out_cope','glm_seed_copes')]),
                       (glm,datasink,[('out_file','glm_betas')])
                      ])
sbc1_workflow.base_dir = workflow_dir
sbc1_workflow.write_graph(graph2use='flat')
sbc1_workflow.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:


infosource2 = Node(IdentityInterface(fields=['roi']),
                   name='infosource2')
infosource2.iterables = ('roi',rois)


# Data grabber- select fMRI and ROIs
templates = {'roi': 'glm_seed_copes/%s_*/cope.nii'}

datagrabber = Node(DataGrabber(infields=['roi'], 
                               outfields=['roi'],
                               sort_filelist=True,
                               base_directory=output_dir,
                               template='glm_seed_copes/%s_*/cope.nii',
                               field_template=templates,
                               template_args=dict(roi=[['roi']])),
                   name='datagrabber')


# In[ ]:


## Level 2

# merge param estimates across all subjects per seed
merge = Node(Merge(dimension='t'),
             name='merge')

# FSL randomise for higher level analysis
highermodel = Node(Randomise(tfce=True,
                             raw_stats_imgs= True,
                             design_mat=group_mat,
                             tcon=group_con),
                   name = 'highermodel')

## Cluster results

# make binary masks of sig clusters
binarize = Node(Binarize(min=0.95, max=1.0), 
                name='binarize', 
                iterfield='in_file')

# mask T-map before clustering
mask_tmaps = Node(ApplyMask(), name='mask_tmaps')

# clusterize and extract cluster stats/peaks
clusterize = Node(Cluster(threshold=2.3, 
                          out_index_file='outindex.nii', 
                          out_localmax_txt_file='localmax.txt'), 
                  name='clusterize')

# make pictures if time


# In[ ]:


sbc2_workflow = Workflow(name='sbc2_workflow')
sbc2_workflow.connect([(infosource2,datagrabber,[('roi','roi')]),
                       (datagrabber,merge,[('roi','in_files')]),
                       (merge,highermodel,[('merged_file','in_file')]),

                       (highermodel,datasink,[('t_corrected_p_files','rand_corrp_files')]),
                       (highermodel,datasink,[('tstat_files','rand_tstat_files')])
                      ])
sbc2_workflow.base_dir = workflow_dir
sbc2_workflow.write_graph(graph2use='flat')
sbc2_workflow.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:


# Identity node- select subjects
infosource3 = Node(IdentityInterface(fields=['roi']),
                   name='infosource3')
infosource3.iterables = [('roi',rois)]


# Data grabber- select fMRI and ROIs
templates = {'pcorrT': output_dir + '/rand_corrp_files/_roi_{roi}/tbss__tfce_corrp_tstat1.nii', 
             'tstat': output_dir + '/rand_tstat_files/_roi_{roi}/tbss__tstat1.nii'}
selectfiles2 = Node(SelectFiles(templates), name='selectfiles2')


# In[ ]:


sbc3_workflow = Workflow(name='sbc3_workflow')
sbc3_workflow.connect([(infosource3,selectfiles2, [('roi','roi')]),
                       (selectfiles2, binarize, [('pcorrT','in_file')]),
                       (binarize, mask_tmaps, [('binary_file','mask_file')]),
                       (selectfiles2, mask_tmaps, [('tstat','in_file')]),
                       (mask_tmaps, clusterize, [('out_file','in_file')]),
                       
                       (binarize,datasink,[('binary_file','binary_pval')]),
                       (mask_tmaps,datasink,[('out_file','masked_tmaps')]),
                       (clusterize,datasink,[('index_file','cluster_index_file')]),
                       (clusterize,datasink,[('localmax_txt_file','localmax_txt_file')])
                      ])
sbc3_workflow.base_dir = workflow_dir
sbc3_workflow.write_graph(graph2use='flat')
sbc3_workflow.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:


# Identity node- select subjects
infosource4 = Node(IdentityInterface(fields=['roi']),
                   name='infosource4')
infosource4.iterables = [('roi',rois)]


# Data grabber- select fMRI and ROIs
templates = {'clusters': output_dir + '/cluster_index_file/_roi_{roi}/outindex.nii', 
             'cluster_table': output_dir + '/localmax_txt_file/_roi_{roi}/cluster_table.txt'}
selectfiles3 = Node(SelectFiles(templates), name='selectfiles3')

# Grab betas for stuff
templates2 = {'roi': 'glm_betas/%s_4mm.nii*-BABIES-T1/betas.nii'}

betagrabber = Node(DataGrabber(infields=['roi'], 
                               outfields=['roi'],
                               sort_filelist=True,
                               base_directory=output_dir,
                               template='glm_betas/%s_4mm.nii*-BABIES-T1/betas.nii',
                               field_template=templates2,
                               template_args=dict(roi=[['roi']])),
                   name='betagrabber')


# In[ ]:


# parse clusters table
def determine_clusters(clusters_table, min_clust_size):
    from os import path
    from numpy import genfromtxt
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    clusters = genfromtxt(clusters_table, delimiter='\t', dtype=None, skip_header=1)
    clusters_to_extract = []
    
    for t in clusters:
        if clusters[t][1] >= min_clust_size:
            clusters_to_extract.append(clusters[t][0])
    
    
    return(cluster_index)

det_clust = Node(name='det_clust', 
                 interface=Function(input_names=['clusters_table','min_clust_size'],
                                    output_names=['cluster_index'], 
                                    function=determine_clusters))
det_clust.inputs.min_clust_size=min_clust_size

# separate cluster volumes
split_clusters = Node(Split(dimension='t'), name='split_clusters')

# merge betas together
merge_betas = Node(Merge(dimension='t'), name='merge_betas')

# extract betas for each subject/roi clusters and put in table as fisher's Z scores
def extract_fisherZ(subj_betas, clusters, cluster_table):
    from os import path
    from numpy import genfromtxt, savetxt
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    header = []
    clusters = genfromtxt(clusters_table, delimiter='\t', dtype=None, skip_header=1)
    
    savetxt(file, matrix, delimiter='\t', header=header)

    return(table_path)

extract_fisherZ = Node(name='extract_fisherZ', 
                       interface=Function(input_names=['subj_betas','clusters','cluster_table'],
                                          output_names=['table_path'], 
                                          function=extract_fisherZ))


# In[ ]:


sbc4_workflow = Workflow(name='sbc4_workflow')
sbc4_workflow.connect([(infosource4, selectfiles3, [('roi','roi')]),
                       (infosource4, betagrabber, [('roi','roi')]),
                       (selectfiles3, split_clusters, [('clusters','in_file')]),
                       (betagrabber, merge_betas, [('roi','in_files')]),
                       
                       (merge_betas, datasink, [('merged_file','merged_betas')]),
                       (split_clusters, datasink, [('out_files','split_clusters')])
                      ])
sbc4_workflow.base_dir = workflow_dir
sbc4_workflow.write_graph(graph2use='flat')
#sbc4_workflow.run('MultiProc', plugin_args={'n_procs': proc_cores})

