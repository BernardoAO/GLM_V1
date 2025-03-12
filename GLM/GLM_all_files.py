import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import Helper_functions as hf

# data file names
cam_files_name = 'cam0_'
working_dir = '\\\\wsl.localhost\\Ubuntu\\home\\ag-schmitz-nwfz\\yota\\project'#'/home/ag_schmitz_nwfz/yota/project'

# Session information
sampling_rate = 30000 # Hz
camara_sampling_rate = 200 # Hz
time_bin_width = 0.005 # s
diff_TTL_usual_value = 150 # sampling_rate / camara_sampling_rate
n_cam_switch = 9 # number of camara switches for each session
n_ROIs = 6
ROI_names = ["Wheel", "Vibrissae", "Nose", "Pupil", "Mouth", "Paw"]

# GLM type and files
GLM_type = "binary_binary"
coef_save_path =  \
    os.path.join(working_dir, 'GLM', 'output_files', 'GLM_coef.npy')
all_GLM_coef = np.load(coef_save_path, allow_pickle=True).item() #{}

sig_save_path =  \
    os.path.join(working_dir, 'GLM', 'output_files', 'GLM_sig.npy')
sig_GLM_coef = {}

# Get valid periods and thresholds
all_awake_periods = np.load(os.path.join(working_dir,'GLM',
                                         'output_files', 'valid_periods.npy'), 
                            allow_pickle=True).item()

""" Old threshold code
all_thresholds = np.load(os.path.join(working_dir,'GLM',
                                      'output_files','manual_thresholds.npy'),
                         allow_pickle=True).item()
    and not all_thresholds[exp[:10]][1] == 0
        thresholds = [th / 3 for th in all_thresholds[exp[:10]]]
"""

# Get data
path_2_spike_bundle =  \
    os.path.join(working_dir, 'data-single-unit')

experiments = [exp for exp in os.listdir(path_2_spike_bundle)
               if exp[:2] == "20" and exp[:10] in all_awake_periods]
experiments.sort()

# file loop
for exp in tqdm(experiments, desc="Files processed"):
    if  True: # exp == '2023-03-21_16-17-18': #
        try:
            
            ## IMPORT SPIKE DATA
            Spke_Bundle, spiketimes, camera_change_times, SIN_data = \
                hf.import_spike_data(exp, working_dir, path_2_spike_bundle)
            
            ## IMPORT BEHAVIOR DATA
            Behavior = hf.import_behavior(exp, working_dir)
            
            ## GLM PIPELINE
            awake_periods = all_awake_periods[exp[:10]]
            
            # Get valid units and unit type
            valid_cluster_indx, cluster_type, unit_colors = \
                hf.get_valid_cluster(Spke_Bundle, SIN_data)            
            valid_spiketimes = [spiketimes[i] for i in valid_cluster_indx]
            
            # Get times for each valid period in the session
            start_behavior, duration_of_period = \
                hf.get_period_times(Behavior, awake_periods, camera_change_times)
            
            # Get spiketrain of each period
            spike_counts_all_periods = []
            behavior_array = []
            for i, period in enumerate(awake_periods):
            
                _, spike_counts = hf.get_spike_counts(valid_spiketimes, 
                                                   start_behavior[i], 
                                                   duration_of_period[i])
                spike_counts_all_periods.append(spike_counts)
            
                behavior_array.append(np.array(Behavior[period]['motion'][1:]).T) #0 indx empty
            
            behavior_array = np.concatenate(behavior_array, axis = 0)
            spike_counts_all_periods = np.concatenate(spike_counts_all_periods, axis = 1)
            thresholds = hf.get_thresholds(behavior_array)
            
            """
            # GLM
            
            GLM_coefs = hf.GLM(behavior_array, spike_counts_all_periods,
                            GLM_type, thresholds)
            all_GLM_coef[exp[:10]] = GLM_coefs
            
            # plot
            hf.plot_activity(behavior_array, spike_counts_all_periods, unit_colors,
                          ROI_names, exp, thresholds)
            costume_bins = np.arange(-0.55, 0.65, 0.1)
            hf.plot_histogram_of_GLM(GLM_coefs, ROI_names, unit_colors,
                                  bins = costume_bins, exp = exp)
            
            """
            # Significance
            GLM_coefs = all_GLM_coef[exp[:10]]
            sig_GLM_coef[exp[:10]] = \
                hf.get_significance(GLM_coefs, behavior_array, spike_counts_all_periods,
                                 GLM_type, thresholds, n_perm = 50)
            
            
        except:
            tqdm.write(" Error when processing file " + exp[:10])


#np.save(coef_save_path, all_GLM_coef)
np.save(sig_save_path, sig_GLM_coef)