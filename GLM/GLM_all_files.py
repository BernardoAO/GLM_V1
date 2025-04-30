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
rate_ratio = 150 # sampling_rate / camara_sampling_rate
time_bin_width = 0.005 # s
n_ROIs = 6
ROI_names = ["Wheel", "Vibrissae", "Nose", "Pupil", "Mouth", "Paw"]

# GLM type and files
GLM_type = "binary_binary"
coef_save_path =  \
    os.path.join(working_dir, 'GLM', 'output_files', 'GLM_coef.npy')
all_GLM_coef = np.load(coef_save_path, allow_pickle=True).item() #{}
all_GLM_coef = {}

sig_save_path =  \
    os.path.join(working_dir, 'GLM', 'output_files', 'GLM_sig.npy')
sig_GLM_coef = {}

# Get valid periods and thresholds
all_awake_periods = np.load(os.path.join(working_dir,'GLM',
                                         'output_files', 'valid_periods.npy'), 
                            allow_pickle=True).item()

# # Vision
# visual_stim_types = ["Sl36x22_d_3",  "Sd36x22_l_3","chirp",
#                     "csd","cm_18x11_1", "mb","mg_sq"]
# visual_windows = np.array([[0.05, 0.1],[0.05, 0.1],
#                            [0.05, 35],[0.05, 1],
#                            [0.05, 0.1],[0.05, 1.7],
#                            [0.05,1.7]])



# Get data
path_2_spike_bundle =  \
    os.path.join(working_dir, 'data-single-unit')

experiments = [exp for exp in os.listdir(path_2_spike_bundle)
               if exp[:2] == "20" and exp[:10] in all_awake_periods]
experiments.sort()

"""experiments =  [#'2021-11-18_15-39-45',
                #'2021-11-19_15-11-14',
                # '2022-11-08_11-35-42', SC
                # '2022-11-09_15-24-08', SC
                '2022-12-20_15-08-10',
                '2022-12-21_13-09-10',
                # '2023-02-23_08-57-20',
                # '2023-03-15_11-05-00',
                # '2023-03-15_15-23-14',
                # '2023-03-16_12-16-07',
                '2023-03-21_16-17-18',
                '2023-03-22_12-22-12',
                '2023-04-13_12-35-02',
                '2023-04-14_11-48-04',
                '2023-04-17_12-26-07',
                '2023-04-18_12-10-34',
                #'2023-08-10_13-07-52',
                #'2023-08-10_16-32-27',
                #'2023-08-11_12-23-01'
                ]"""

# file loop
unit_colors_storage = dict()
for exp in tqdm(experiments, desc="Files processed"):
    try:
        
        ## IMPORT 
        
        visual_stim_types = ["Sd36x22_l_3","Sl36x22_d_3", "chirp",
                            "cm_18x11_1","csd", "mb","mg","mg_sq"]
        visual_windows = np.array([[0.05, 0.1],[0.05, 0.1],
                                   [0.05, 35],[0.05, 0.1],
                                   [0.05, 1],[0.05, 1.7],
                                   [0.05, 1.7],[0.05, 1.7]
                                   ])
        
        # spike data
        Spke_Bundle, spiketimes, camera_change_times, SIN_data = \
            hf.import_spike_data(exp, working_dir, path_2_spike_bundle)
        
        ## behavior
        Behavior, running_band = hf.import_behavior(exp, working_dir)
        
        # vision
        
        #we here prepare a alternative visual_stim_types based on the stim present
        vis_stim_local = Spke_Bundle["events"].keys()
        
        visual_stim_types_local = list(set(visual_stim_types) & set(vis_stim_local))
        visual_stim_types_local = sorted(visual_stim_types_local)
        
        visual_windows = visual_windows[0:len(visual_stim_types_local)]
        
        visual_times = [(Spke_Bundle["events"][type] - 
                         Spke_Bundle["Synchronization_TTLs"]["Sync_cam"][0]) 
                        / sampling_rate
                        for type in visual_stim_types_local]
    
        ## GLM PIPELINE
        
        # Get valid units and unit type
        valid_cluster_indx, cluster_type, unit_colors = \
            hf.get_valid_cluster(Spke_Bundle, SIN_data)            
        valid_spiketimes = [spiketimes[i] for i in valid_cluster_indx]
        
        # Get times for each valid period in the session
        awake_periods = all_awake_periods[exp[:10]]
        start_behavior, duration_of_period = \
            hf.get_period_times(Behavior, awake_periods, camera_change_times)
        
        # Get spiketrain, behavior, and visual input of each period
        spike_counts_all_periods, behavior_array, thresholds, visual_array = \
            hf.extract_periods(valid_spiketimes, awake_periods, start_behavior, 
                            duration_of_period, Behavior, visual_times, 
                            visual_windows)
        
        # GLM

        binary_behavior = hf.get_binary_behavior(behavior_array)
        if exp == '2023-04-13_12-35-02':
            visual_array = visual_array[:,0:binary_behavior.shape[0]]
            
        GLM_regresors = np.concatenate((binary_behavior, visual_array.T), 
                                       axis = 1)
        GLM_coefs = hf.GLM(GLM_regresors, spike_counts_all_periods,
                        GLM_type, thresholds, n_core = 4)
        all_GLM_coef[exp[:10]] = GLM_coefs
        
        ## PLOT
        
        hf.plot_activity(behavior_array, spike_counts_all_periods, unit_colors,
                      ROI_names, exp, thresholds)
        
        costume_bins = np.arange(-0.7, 0.7, 0.02)
        
        hf.plot_histogram_of_GLM(GLM_coefs, ROI_names, unit_colors,
                              bins = costume_bins, exp = exp)
        
        hf.plot_histogram_of_GLM(GLM_coefs[:,6:], visual_stim_types_local, unit_colors,
                              bins = costume_bins, exp = exp)#, n_ROIs=7)
        
        unit_colors_storage[exp[0:10]] = unit_colors
        
        
        # # Significance
        # GLM_coefs = all_GLM_coef[exp[:10]]
        # sig_GLM_coef[exp[:10]] = \
        #     hf.get_significance(GLM_coefs, GLM_regresors, spike_counts_all_periods,
        #                       GLM_type, thresholds, n_perm = 50)
    
        
    except:
        tqdm.write(" Error when processing file " + exp[:10])


np.save(coef_save_path, all_GLM_coef)
np.save(coef_save_path[0:-4] + '_color.npy', unit_colors_storage)
np.save(sig_save_path, sig_GLM_coef)

loc_storage_GLM = []
loc_storage_GLM_stat = []
loc_storage_color = []

for i,exp_loc in enumerate(all_GLM_coef.keys()):
    if exp_loc == '2023-08-11':
        continue
    if i == 0:
        loc_storage_GLM = np.array(all_GLM_coef[exp_loc])
        # loc_storage_GLM_stat = np.array(sig_GLM_coef[exp_loc])
        loc_storage_color = np.array(unit_colors_storage[exp_loc])
    else:
        loc_storage_GLM = np.append(loc_storage_GLM,np.array(all_GLM_coef[exp_loc]),axis = 0)
        loc_storage_GLM_stat = np.append(loc_storage_GLM_stat,np.array(sig_GLM_coef[exp_loc]),axis = 0)
        loc_storage_color = np.append(loc_storage_color,unit_colors_storage[exp_loc])
        
exp = 'All recordings vision'
hf.plot_histogram_of_GLM(loc_storage_GLM[:,6:], visual_stim_types, loc_storage_color,
                      bins = costume_bins, exp = exp)#, n_ROIs=7)

exp = 'All recordings behavior'
hf.plot_histogram_of_GLM(loc_storage_GLM[:,:], visual_stim_types, loc_storage_color,
                      bins = costume_bins, exp = exp)#, n_ROIs=7)

# exp = 'All recordings vision stat'
# hf.plot_histogram_of_GLM(loc_storage_GLM_stat[:,6:], visual_stim_types, loc_storage_color,
#                       bins = costume_bins, exp = exp)#, n_ROIs=7)

# exp = 'All recordings behavior stat'
# hf.plot_histogram_of_GLM(loc_storage_GLM_stat[:,:], visual_stim_types, loc_storage_color,
#                       bins = costume_bins, exp = exp)#, n_ROIs=7)

#now we do the related scatter plot



""" Old threshold code
all_thresholds = np.load(os.path.join(working_dir,'GLM',
                                      'output_files','manual_thresholds.npy'),
                         allow_pickle=True).item()
    and not all_thresholds[exp[:10]][1] == 0
        thresholds = [th / 3 for th in all_thresholds[exp[:10]]]
"""
