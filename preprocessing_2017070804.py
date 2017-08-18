
# coding: utf-8

# In[1]:


#import packages
from os import listdir
from nipype.interfaces.io import DataSink, SelectFiles, FreeSurferSource # Data i/o
from nipype.interfaces.utility import IdentityInterface, Function     # utility
from nipype.pipeline.engine import Node, Workflow, MapNode        # pypeline engine
from nipype.interfaces.nipy.preprocess import Trim

from nipype.algorithms.rapidart import ArtifactDetect 
from nipype.interfaces.fsl.preprocess import SliceTimer, MCFLIRT, FLIRT, SUSAN, FAST
from nipype.interfaces.fsl.utils import Reorient2Std
from nipype.interfaces.fsl.model import GLM
from nipype.interfaces.fsl.maths import ApplyMask, TemporalFilter
from nipype.interfaces.freesurfer import Resample, Binarize, FSCommand, MRIConvert
from nipype.algorithms.confounds import CompCor


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
output_dir = studyhome + '/proc/preproc'
workflow_dir = studyhome + '/workflows'
subjects_list = open(studyhome + '/misc/subjects.txt').read().splitlines()
#subjects_list = ['101']

template_brain = studyhome + '/templates/MNI152_T1_2mm_brain.nii'

#freesurfer setup
fs_dir = studyhome + '/freesurfer'
FSCommand.set_default_subjects_dir(fs_dir)

proc_cores = 2 # number of cores of processing for the workflows

#vols_to_trim = 4
interleave = True
TR = 2 # in seconds
slice_dir = 3 # 1=x, 2=y, 3=z
resampled_voxel_size = (2,2,2)
fwhm = 4 #fwhm for smoothing with SUSAN

highpass_freq = 0.008 #in Hz
lowpass_freq = 0.09 #in Hz

mask_erosion = 1
mask_dilation = 2


# In[2]:


## File handling Nodes

# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id']),
                     name='infosource')
infosource.iterables = ('subject_id', subjects_list)


# Data grabber- select fMRI and sMRI
templates = {'func': raw_data + '/{subject_id}/rest/rest_raw.nii'}
selectfiles = Node(SelectFiles(templates), name='selectfiles')

# FreeSurferSource - Data grabber specific for FreeSurfer data
fssource = Node(FreeSurferSource(subjects_dir=fs_dir),
                run_without_submitting=True,
                name='fssource')

# Datasink- where our select outputs will go
substitutions = [('_subject_id_', '')] #output file name substitutions
datasink = Node(DataSink(substitutions = substitutions), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir


# In[3]:


## Nodes for preprocessing

# Reorient to standard space using FSL
reorientfunc = Node(Reorient2Std(), name='reorientfunc')
reorientstruct = Node(Reorient2Std(), name='reorientstruct')

# Reslice- using MRI_convert 
reslice = Node(MRIConvert(vox_size=resampled_voxel_size, out_type='nii'), 
               name='reslice')

# Segment structural scan
#segment = Node(Segment(affine_regularization='none'), name='segment')
segment = Node(FAST(no_bias=True, 
                    segments=True, 
                    number_classes=3), 
               name='segment')

#Slice timing correction based on interleaved acquisition using FSL
slicetime_correct = Node(SliceTimer(interleaved=interleave, 
                                    slice_direction=slice_dir,
                                   time_repetition=TR),
                            name='slicetime_correct')

# Motion correction
motion_correct = Node(MCFLIRT(save_plots=True, 
                              mean_vol=True), 
                      name='motion_correct')

# Registration- using FLIRT
# The BOLD image is 'in_file', the anat is 'reference', the output is 'out_file'
coreg1 = Node(FLIRT(), name='coreg1')
coreg2 = Node(FLIRT(apply_xfm=True), name = 'coreg2')

# make binary mask 
# structural is the 'in_file', output is 'binary_file'
binarize_struct = Node(Binarize(dilate=mask_dilation, 
                                erode=mask_erosion, 
                                min=1), 
                       name='binarize_struct')

# apply the binary mask to the functional data
# functional is 'in_file', binary mask is 'mask_file', output is 'out_file'
mask_func = Node(ApplyMask(), name='mask_func')


# Artifact detection for scrubbing/motion assessment
art = Node(ArtifactDetect(mask_type='file',
                          parameter_source='FSL',
                          norm_threshold=0.5, #mutually exclusive with rotation and translation thresh
                          zintensity_threshold=3,
                          use_differences=[True, False]),
           name='art')

def converthex_xform(orig_xform):
    from numpy import genfromtxt, savetxt
    from os.path import abspath
    
    orig_matrix = genfromtxt(orig_xform, delimiter='  ',
                             dtype=None, skip_header=0)
    new_xform = 'brainmask_out_flirt.mat'
    savetxt(new_xform, orig_matrix, delimiter='  ')
    
    xform_file = abspath(new_xform)
    return(xform_file)

converthex = Node(name='converthex', 
                  interface=Function(input_names=['orig_xform'], 
                                     output_names=['xform_file'], 
                                     function=converthex_xform))
converthex2 = Node(name='converthex2', 
                   interface=Function(input_names=['orig_xform'], 
                                      output_names=['xform_file'], 
                                      function=converthex_xform))


# In[4]:


# Data QC nodes
def create_coreg_plot(epi,anat):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    coreg_filename='coregistration.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'coregistration to anatomy')
    display.add_edges(anat)
    display.savefig(coreg_filename) 
    display.close()
    coreg_file = os.path.abspath(coreg_filename)
    
    return(coreg_file)

def check_mask_coverage(epi,brainmask):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    maskcheck_filename='maskcheck.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'brainmask coverage')
    display.add_contours(brainmask,levels=[.5], colors='r')
    display.savefig(maskcheck_filename)
    display.close()
    maskcheck_file = os.path.abspath(maskcheck_filename)

    return(maskcheck_file)

make_coreg_img = Node(name='make_coreg_img',
                      interface=Function(input_names=['epi','anat'],
                                         output_names=['coreg_file'],
                                         function=create_coreg_plot))

make_checkmask_img = Node(name='make_checkmask_img',
                      interface=Function(input_names=['epi','brainmask'],
                                         output_names=['maskcheck_file'],
                                         function=check_mask_coverage))


# In[1]:


# Normalization
register_template = Node(FLIRT(reference=template_brain), 
                         name='register_template')
xfmTissue = MapNode(FLIRT(reference=template_brain,apply_xfm=True), 
                 name='xfmTissue', 
                    iterfield=['in_file'])
xfmFUNC = Node(FLIRT(reference=template_brain,apply_xfm=True), 
               name='xfmFUNC')

def adjust_masks(masks):
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from nipype.interfaces.freesurfer.model import Binarize
    #pve0 = csf, pve1 = gm, pve2 = wm
    
    origvols = sorted(masks)
    csf = origvols[0]
    wm = origvols[2]
    
    erode = Binarize()
    erode.inputs.in_file = wm
    erode.inputs.erode = 1
    erode.inputs.min = 0.5
    erode.inputs.max = 1000
    erode.inputs.binary_file = 'WM_seg.nii'
    erode.run()
    
    wm_new = abspath(erode.inputs.binary_file)
    
    vols = []
    vols.append(wm_new)
    vols.append(csf)
    
    return(vols)
    
fix_confs = Node(name='fix_confs',
                 interface=Function(input_names=['masks'], 
                                    output_names=['vols'],
                                    function=adjust_masks))

compcor = Node(CompCor(merge_method='none'), 
               name='compcor')


# Remove all noise (GLM with noise params)
def create_noise_matrix(vols_to_censor,motion_params,comp_noise):
    from numpy import genfromtxt, zeros,concatenate, savetxt
    from os import path
    
    motion = genfromtxt(motion_params, delimiter='  ', dtype=None, skip_header=0)
    comp_noise = genfromtxt(comp_noise, delimiter='\t', dtype=None, skip_header=1)
    censor_vol_list = genfromtxt(vols_to_censor, delimiter='\t', dtype=None, skip_header=0)
    
    try:
        c = censor_vol_list.size
    except:
        c = 0
    
    d=len(comp_noise)

    if c > 1:
        scrubbing = zeros((d,c),dtype=int)
        for t in range(c):
            scrubbing[censor_vol_list[t],t] = 1
        noise_matrix = concatenate((motion,comp_noise,scrubbing),axis=1)
    elif c == 1:
        scrubbing = zeros((d,c),dtype=int)
        scrubbing[censor_vol_list] = 1
        noise_matrix = concatenate((motion,comp_noise,scrubbing),axis=1)
    else:
        noise_matrix = concatenate((motion,comp_noise),axis=1)
    
    noise_file = 'noise_matrix.txt'
    savetxt(noise_file, noise_matrix, delimiter='\t')
    noise_filepath = path.abspath(noise_file)
    
    return(noise_filepath)

noise_mat = Node(name='noise_mat', interface=Function(input_names=['vols_to_censor','motion_params','comp_noise'],
                                                      output_names=['noise_filepath'], 
                                                      function=create_noise_matrix))

denoise = Node(GLM(out_res_name='denoised_residuals.nii', 
                   out_data_name='denoised_func.nii'), 
               name='denoise')

# band pass filtering- all rates are in Hz (1/TR or samples/second)
def bandpass_filter(in_file, lowpass, highpass, TR):
    import numpy as np
    import nibabel as nb
    from os import path
    from nipype.interfaces.afni.preprocess import Bandpass
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    out_file = 'func_filtered.nii'
    bp = Bandpass()
    bp.inputs.highpass = highpass
    bp.inputs.lowpass = lowpass
    bp.inputs.in_file = in_file
    bp.inputs.tr = TR
    bp.inputs.out_file = out_file
    bp.inputs.outputtype = 'NIFTI'
    bp.run()
    
    out_file = path.abspath(out_file)
    return(out_file)

bandpass = Node(name='bandpass', 
                interface=Function(input_names=['in_file','lowpass','highpass','TR'], 
                                   output_names=['out_file'],
                                   function=bandpass_filter))
bandpass.inputs.lowpass = lowpass_freq
bandpass.inputs.highpass = highpass_freq
bandpass.inputs.TR = TR


# Spatial smoothing using FSL
# Brightness threshold should be 0.75 * the contrast between the median brain intensity and the background
def brightthresh(func):
    import nibabel as nib
    from numpy import median, where
    
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    func_nifti1 = nib.load(func)
    func_data = func_nifti1.get_data()
    func_data = func_data.astype(float)
    
    brain_values = where(func_data > 0)
    median_thresh = median(brain_values)
    bright_thresh = 0.75 * median_thresh
    
    return(bright_thresh)

brightthresh_filt = Node(name='brightthresh_filt',
                         interface=Function(input_names=['func'], 
                                            output_names=['bright_thresh'], 
                                            function=brightthresh))    
    
smooth_filt = Node(SUSAN(fwhm=fwhm), name='smooth_filt')


brightthresh_orig = Node(name='brightthresh_orig',
                         interface=Function(input_names=['func'], 
                                            output_names=['bright_thresh'], 
                                            function=brightthresh))    
    
smooth_orig = Node(SUSAN(fwhm=fwhm), name='smooth_orig')


# In[8]:


## Preprocessing Workflow

# workflowname.connect([(node1,node2,[('node1output','node2input')]),
#                    (node2,node3,[('node2output','node3input')])
#                    ])

preprocwf = Workflow(name='preprocwf')
preprocwf.connect([(infosource,selectfiles,[('subject_id','subject_id')]), 
                   (infosource, fssource, [('subject_id','subject_id')]),
                   (fssource,reslice, [('brainmask','in_file')]),
                   (reslice, reorientstruct,[('out_file','in_file')]),
                   (selectfiles,reorientfunc,[('func','in_file')]),
                   (reorientstruct,coreg1,[('out_file','reference')]),
                   (reorientstruct,coreg2,[('out_file','reference')]),
                   (reorientstruct,segment,[('out_file','in_files')]),
                   (reorientfunc,slicetime_correct,[('out_file','in_file')]),
                   (slicetime_correct,motion_correct,[('slice_time_corrected_file','in_file')]),
                   (motion_correct,coreg1,[('out_file','in_file')]),
                   (motion_correct,coreg2,[('out_file','in_file')]),
                   (coreg1, converthex,[('out_matrix_file','orig_xform')]),
                   (converthex, coreg2,[('xform_file', 'in_matrix_file')]),
                   (reorientstruct, binarize_struct, [('out_file','in_file')]),
                   (binarize_struct,mask_func,[('binary_file','mask_file')]),
                   (coreg2,mask_func,[('out_file','in_file')]),
                   (mask_func,art,[('out_file','realigned_files')]),
                   (binarize_struct,art,[('binary_file','mask_file')]),
                   (motion_correct,art,[('par_file','realignment_parameters')]),
                   (coreg1,make_coreg_img,[('out_file','epi')]),
                   (reorientstruct,make_coreg_img,[('out_file','anat')]),
                   (binarize_struct,make_checkmask_img,[('binary_file','brainmask')]),
                   (coreg1,make_checkmask_img,[('out_file','epi')]),
                   
                   (reorientstruct,register_template,[('out_file','in_file')]),
                   (mask_func,xfmFUNC,[('out_file','in_file')]),
                   (segment,xfmTissue,[('tissue_class_files','in_file')]),
                   (register_template,converthex2,[('out_matrix_file','orig_xform')]),
                   (converthex2,xfmFUNC,[('xform_file','in_matrix_file')]),
                   (converthex2,xfmTissue,[('xform_file','in_matrix_file')]),
                   (xfmTissue,fix_confs,[('out_file','masks')]),
                   (fix_confs,compcor,[('vols','mask_files')]),
                   (xfmFUNC,compcor,[('out_file','realigned_file')]),
                   (compcor,noise_mat,[('components_file','comp_noise')]),
                   (art,noise_mat,[('outlier_files','vols_to_censor')]),
                   (motion_correct,noise_mat,[('par_file','motion_params')]),
                   (noise_mat,denoise,[('noise_filepath','design')]),
                   (xfmFUNC,denoise,[('out_file','in_file')]),
                   (denoise,bandpass,[('out_data','in_file')]),
                   (bandpass,brightthresh_filt,[('out_file','func')]),
                   (brightthresh_filt,smooth_filt,[('bright_thresh','brightness_threshold')]),
                   (bandpass,smooth_filt,[('out_file','in_file')]), 
                   (denoise,brightthresh_orig,[('out_file','func')]),
                   (brightthresh_orig,smooth_orig,[('bright_thresh','brightness_threshold')]),
                   (denoise,smooth_orig,[('out_data','in_file')]),
                   
                   #(motion_correct,datasink,[('par_file','motion_params')]),
                   (reorientstruct,datasink,[('out_file','resliced_struct')]),
                   #(mask_func,datasink,[('out_file','masked_func')]),
                   (segment,datasink,[('tissue_class_files','tissue_class_files')]),
                   (art,datasink, [('plot_files','art_plot_files')]),
                   #(art,datasink, [('outlier_files','vols_to_censor')]),
                   (make_checkmask_img,datasink,[('maskcheck_file','maskcheck_image')]),
                   (make_coreg_img,datasink,[('coreg_file','coreg_image')]),
                   #(compcor,datasink,[('components_file','components_file')]),
                   (smooth_filt,datasink,[('smoothed_file','smoothed_filt_func')]),
                   (smooth_orig,datasink,[('smoothed_file','smoothed_orig_func')]),
                   (bandpass,datasink,[('out_file','bp_filtered_func')]),
                   (denoise,datasink,[('out_data','denoised_func')])
                  ])
preprocwf.base_dir = workflow_dir
preprocwf.write_graph(graph2use='flat')
preprocwf.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:




