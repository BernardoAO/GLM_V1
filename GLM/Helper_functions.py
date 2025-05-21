import numpy as np
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def import_spike_data(exp, working_dir, path_2_spike_bundle, 
                      sampling_rate = 30000, time_bin_width = 0.005, diff_TTL_usual_value = 150):
    """
    Imports spike bundle data from the _Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.npy files,
    and supposedly inhibitory neuron data from the '_local_storage_SIN.npy' files
    """
    # load spike data for each unit
    Spke_Bundle_name = os.path.join(path_2_spike_bundle, exp,
                                    exp + '_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.npy')
    Spke_Bundle = \
        np.load(Spke_Bundle_name, allow_pickle=True,encoding='latin1').item()
    spiketimes = [(unit_times - Spke_Bundle["Synchronization_TTLs"]["Sync_cam"][0]) / sampling_rate
                  for unit_times in Spke_Bundle['spiketimes_aligned']]  
    
    # Get how much time was spended changing cameras
    diff_TTLs = np.diff(Spke_Bundle["Synchronization_TTLs"]["Sync_cam"])
    camera_change_indx = np.argwhere(diff_TTLs > 1000)
    camera_change_times = diff_TTLs[camera_change_indx] * time_bin_width / diff_TTL_usual_value
    camera_change_times = np.insert(camera_change_times, 0, 0)
    
    # load Supposedly inhibitory neurons file
    path_SIN_data =  \
        os.path.join(working_dir, 'data-single-unit', 
                        exp, 
                        exp + '_local_storage_SIN.npy')
    SIN_data =  \
        np.load(path_SIN_data,allow_pickle=True,encoding='latin1').item()

    return Spke_Bundle, spiketimes, camera_change_times, SIN_data
    
def import_behavior(exp, working_dir, Spke_Bundle, rate_ratio=150):
    """
    Imports behavioral data from the cam0_ files and the running band data.

    Parameters:
    - exp: session name
    - working_dir
    - Spke_Bundle array
    - rate_ratio: ratio between the running band and cam signals

    """
    # get all cam files of the day
    cam_files_name = 'cam0_' + exp[:10]

    path_2_behavior =  \
        os.path.join(working_dir, 'Facemap/facemap_motion_data')
    cam_files = [file for file in os.listdir(path_2_behavior) 
                        if file.startswith(cam_files_name)]
    cam_files.sort()
    
    Behavior = []
    for file in cam_files:
        Behavior.append(np.load(os.path.join(path_2_behavior,file)
                ,allow_pickle=True,encoding='latin1').item())
    
    # get running band data
    path_2_run =  \
        os.path.join(working_dir, 'storage_run_simple')
    file = exp + "_local_simple_run.npy"

    if file in os.listdir(path_2_run):
        running_band = np.load(os.path.join(path_2_run,file), allow_pickle=True,encoding='latin1')
        
        rb_start = Spke_Bundle["Synchronization_TTLs"]["Sync_cam"][0] - \
                    Spke_Bundle["Synchronization_TTLs"]["Sync_NIDAQmx"][0]
        running_band = running_band[rb_start:]
        running_band = np.abs(running_band - running_band[0])
        running_band = running_band[::rate_ratio]
    else:
        running_band = []

    return Behavior, running_band

def get_valid_cluster(Spke_Bundle, SIN_data):
    """
    Gets valid units for GLM study.
    Parameters:
    - Spke_Bundle: Spike bundle dictionary
    - SIN_data: Supposedly inhibitory neuron dictionary

    Returns:
    - valid_cluster_indx: indices where valid_cluster is True
    - cluster_type: Name of each valid cluster (TCA, NW, BW)
    - unit_colors: color of each valid cluster

    """

    valid_cluster_indx = []
    cluster_type = []

    for neu_indx, GUI_name in enumerate(Spke_Bundle["classif_from_GUI"]["Classification"]):

        if neu_indx in SIN_data["Classif_SIN_indx"]:
            cluster_type.append("NW")
            valid_cluster_indx.append(neu_indx)

        elif neu_indx in SIN_data["Classif_SUR_indx"]:
            cluster_type.append("BW")
            valid_cluster_indx.append(neu_indx)

        elif GUI_name == 'MPW-Axon':
            cluster_type.append("TCA")
            valid_cluster_indx.append(neu_indx)

    # color the units 
    unit_colors = ["purple" if type == "TCA" 
                           else "red" if type == "NW" 
                           else "black" for type in cluster_type]
    
    return valid_cluster_indx, cluster_type, unit_colors

def get_period_times(Behavior, awake_periods, camera_change_times_duration, camara_sampling_rate = 200):
    """    
    Parameters:
    - Behavior: Dictionary with all cam0 files of the day
    - awake_periods: periods to analize (from 0 to n)
    - camera_change_times_duration: time between each cam0 file [seconds] 

    Returns:
    - start_of_period: first time in the period relative to the Spike times [seconds]
    - duration_of_period: duration of the period [seconds] 
    """

    start_of_period = np.zeros(len(awake_periods))
    duration_of_period = np.zeros(len(awake_periods))
    for period in awake_periods:

        duration_of_period[period] = len(Behavior[period]['motion'][1]) / camara_sampling_rate
        start_of_period[period] = start_of_period[period - 1] + duration_of_period[period - 1] + camera_change_times_duration[period]
    
    return start_of_period, duration_of_period

def get_spike_counts(spike_times, start = 0, duration = 1000, time_bin_width = 0.005, GLM_type = "binary_binary"):
    """
    Gets a matrix of n_units x period_time aligned to the start of the period with the spike counts for fixed windows.
    
    Parameters:
    - spike_times: List of lenght n_units containing a list of the spike times
    - start: first time in the period relative to the Spike times [seconds]
    - duration: duration of the period [seconds] 
    - time_bin_width: width of the time bin (step size)

    Returns:
    - tv: time vector
    - binary_st: Matrix of binarized spikes n_units x time
    """

    tv = np.arange(0, duration, time_bin_width)
    binary_st = np.zeros((len(spike_times), len(tv)))
    for n in range(len(spike_times)):

        aligned_spike_times = np.array(spike_times[n]) - start
        all_spikes_indx = aligned_spike_times // time_bin_width
        indx = all_spikes_indx[(all_spikes_indx > 0) & 
                               (all_spikes_indx < len(tv))].astype("uint64")
        if GLM_type == "binary_binary" or GLM_type == "continious_binary" :
            binary_st[n, indx] = 1
        else:
            for i in indx:
                binary_st[n, i] += 1 
    
    return tv, binary_st

def get_visual_input(visual_times, windows, start = 0, duration = 1000, time_bin_width = 0.005):
    """
    Gets a matrix of n_stimtypes x period_time aligned to the start of the period with the spike counts for fixed windows.
    
    Parameters:
    - visual_times: List of input types containing a list of the event times.
    - windows: start and end [seconds] of a time window to use for each stim type. shape: n_types x 2
    - start: first time in the period relative to the Spike times [seconds]
    - duration: duration of the period [seconds] 
    - time_bin_width: width of the time bin (step size)

    Returns:
    - tv: time vector
    - binary_st: Matrix of binarized spikes n_units x time
    """

    tv = np.arange(0, duration, time_bin_width)
    binary_stim = np.zeros((len(visual_times), len(tv)))
    
    for n in range(len(visual_times)):

        aligned_visual_times = np.array(visual_times[n]) - start
        all_times_indx = aligned_visual_times // time_bin_width
        indx = all_times_indx[(all_times_indx > 0) & 
                               (all_times_indx < len(tv))]
        window_indx = windows[n,:] // time_bin_width
        joint_indx = np.array([np.arange(i + window_indx[0], i + window_indx[1])
                                   for i in indx]).flatten().astype("uint64")
        binary_stim[n, joint_indx[joint_indx < len(tv)]] = 1
    
    return tv, binary_stim

def join_running_band(behavior_no_rb, running_band, sync):
    """
    Substitutes the Wheel cam channel (first channel of the input array) for the running band signal,
    in the case there is one
    
    Parameters:
    - behavior_no_rb: original behavior array, shape time x n_ROIs
    - running_band: shape t
    - sync: syncronization index for each period
    Returns:
    - behavior_rb: behavior array with the running band signal, shape time x n_ROIs
    """
    # Import
    
    if len(running_band) > 0:
        lenb = behavior_no_rb.shape[0]
        behavior_rb = np.copy(behavior_no_rb)

        behavior_rb[:,0] = running_band[sync:sync+lenb]

        ''' #Debug running band
        
        # plot current
        fig, ax1 = plt.subplots()
        t = np.arange(0,lenb/200,1/200)
        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('cam', color=color)
        ax1.plot(t, behavior_no_rb[:,0], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('running_band', color=color) 
        ax2.plot(t, behavior_rb[:,0], color=color, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.show()'''

        return behavior_rb

    else:
        return behavior_no_rb
    
    
    

def extract_periods(valid_spiketimes, awake_periods, start_behavior, duration_of_period,
                    Behavior, running_band, visual_times, visual_windows, camara_sampling_rate = 200):
    """
    Aligns and merges the spike, behavior, and visual time signals into periods defined by the times
    the camara was active, respectively.
    """
    spike_counts_all_periods = []
    behavior_array = []
    visual_array = []
    for i, period in enumerate(awake_periods):
        # spikes
        _, spike_counts = get_spike_counts(valid_spiketimes, 
                                            start_behavior[i], 
                                            duration_of_period[i])
        spike_counts_all_periods.append(spike_counts)
        
        # behavior
        behavior_no_rb = np.array(Behavior[period]['motion'][1:]).T #0 indx empty
        sync = int(start_behavior[i]*camara_sampling_rate) # 6666
        behavior_rb = join_running_band(behavior_no_rb, running_band, sync)
        behavior_array.append(behavior_rb) 
        
        # visual
        _, binary_stim = get_visual_input(visual_times, visual_windows,
                                                start_behavior[i], 
                                                duration_of_period[i])
        visual_array.append(binary_stim)
    
    spike_counts_all_periods = np.concatenate(spike_counts_all_periods, axis = 1)
    
    behavior_array = np.concatenate(behavior_array, axis = 0)
    thresholds = get_thresholds(behavior_array)
    
    visual_array = np.concatenate(visual_array, axis = 1)
    return spike_counts_all_periods, behavior_array, thresholds, visual_array

def get_binary_behavior(B, thresholds=[]):
    """
    Turns a SVd matrix to a binary matrix (move or not move) given a threshold
    
    Parameters:
    - B: Matrix of SVd's, shape: time x n_ROIs
    - thresholds: Threshold for each ROI, 
        if non given the function calculates the mean plus two standard deviations 

    Returns:
    - Binarization of the behavior, shape time x n_ROIs
    """

    if not thresholds:
        thresholds = np.mean(B, axis = 0) + 2 * np.std(B, axis = 0)
    
    return np.where(B < thresholds, 0, 1)

def get_thresholds(behavior_array, proportion = 0.85):
    """
    Gets the thresholds to binarize the behavior so that it follows a given proportion.
    Parameters:
    - behavior_array: Matrix of SVd's, shape: time x n_ROIs
    - proportion: Proportion of the non-behaving time.

    Returns:
    - thresholds: Threshold for each ROI, shape: n_ROIs
    """
    min_indx = int(behavior_array.shape[0] * proportion)
    sorted_array = np.sort(behavior_array, axis = 0)
    thresholds = sorted_array[min_indx, :]
    
    return thresholds.tolist()

def clean_behavior(behavior_array, n_ROIs = 6, ratio_for_outlayer = 1e-3, stds_for_outlayer = 5):
    """
    Cleans behavior array by removing outlayers and normalizing.
    """
    clean_behavior = np.empty_like(behavior_array)
    
    for roi in range(n_ROIs):
        # Remove outlayers
        b = np.copy(behavior_array[:, roi])
        outlayer_threshold = np.mean(b) + stds_for_outlayer * np.std(b)
        if sum(b > outlayer_threshold) / len(b) < ratio_for_outlayer:
            b[b > outlayer_threshold] = 0
        clean_behavior[:, roi] = b

        # Normalize each roi to 1
        clean_behavior[:, roi] /= np.max(clean_behavior[:, roi])

    return clean_behavior

def GLM(behavior_array, spike_counts, GLM_type, thresholds = [], n_core = -1):
    """
    Fits a Generalized Linear Model (GLM).
    
    Parameters:
    - behavior_array: Regressors (behavioral responses), shape: time x number_of_regressors.
    - spike_counts: Outcome variable (neural responses), shape: n_neurons x time.
    - GLM_type: "binary_binary": binary behavior to binary neural responses.
                "continious_binary": continious behavior to binary neural responses.
                "binary_continious": binary behavior to continious neural responses.
                "continious_continious": continious behavior to continious neural responses.
    - thresholds: thresholds for binary behavior

    Returns:
    - GLM_coefs: array of GLM coeffitiens, shape n_neurons x number_of_regressors.
    """
    n_neu = spike_counts.shape[0]
    GLM_coefs = np.zeros((n_neu, behavior_array.shape[1]))

    if GLM_type == "binary_binary":
        clf = linear_model.LogisticRegression()
        X = behavior_array
        y = spike_counts #np.where(spike_counts >= 1, 1, 0)  

    elif GLM_type == "continious_binary":
        clf = linear_model.LogisticRegression()
        X = clean_behavior(behavior_array)
        y = np.where(spike_counts >= 1, 1, 0) 
        
    elif GLM_type == "continious_continious":
        clf = linear_model.PoissonRegressor()
        X = clean_behavior(behavior_array)
        y = spike_counts

    # Define function to fit the model for each neuron
    def fit_model_for_neuron(n):
        clf.fit(X, y[n])
        return clf.coef_

    # Parallelize the model fitting across neurons
    GLM_coefs = Parallel(n_jobs=n_core)(delayed(fit_model_for_neuron)(n) for n in range(n_neu))

    return np.concatenate(GLM_coefs, axis = 0)

def get_significance(GLM_coefs, behavior_array, spike_counts, GLM_type, thresholds,
                     n_perm = 1000, p_value = 0.05):
    """
    Calculates significance with a permutation test.
    
    Parameters:
    - GLM_coefs: array of GLM coeffitiens, shape n_neurons x number_of_regressors.
    - behavior_array: Regressors (behavioral responses), shape: time x number_of_regressors.
    - spike_counts: Outcome variable (neural responses), shape: n_neurons x time.
    - GLM_type: "binary_binary": binary behavior to binary neural responses.
                "continious_binary": continious behavior to binary neural responses.
                "binary_continious": binary behavior to continious neural responses.
                "continious_continious": continious behavior to continious neural responses.
    - thresholds: thresholds for binary behavior
    - n_perm: number of permutations
    - p_value: p value

    Returns:
    - boolean array of significant coeficients, shape n_neurons x number_of_regressors.
    """
    n_neu = spike_counts.shape[0]
    perm_coef = np.zeros((GLM_coefs.shape[0], GLM_coefs.shape[1], n_perm))
    
    print("Permutations: ")
    for p in range(n_perm):
        
        # permute
        permuted_spike_counts = spike_counts.copy()
        for n in range(n_neu):
            permuted_spike_counts[n] = np.random.permutation(permuted_spike_counts[n])
        
        # GLM
        perm_coef[:, :, p] = GLM(behavior_array, permuted_spike_counts, 
                                GLM_type, thresholds)
        
        if p % 10 == 0:
            print(p, "% ")
            
    
    # Calculate proportion bigger than randomnes
    abs_GLM_coefs = np.abs(GLM_coefs)
    comparison = abs_GLM_coefs[:, :, np.newaxis] < np.abs(perm_coef)
    proportion = np.sum(comparison, axis=2) / n_perm

    return proportion < p_value


## Plot functions

def plot_SVD_thresholds(behavior_array, thresholds, exp, ROI_names, n_ROIs = 6, camara_sampling_rate = 200):
    binary_behavior = get_binary_behavior(behavior_array, thresholds)
    tv_behavior = np.arange(0, behavior_array.shape[0] / camara_sampling_rate, 1 / camara_sampling_rate)
    plt.figure(figsize=(10,7))
    for roi in range(n_ROIs):
        plt.subplot(6, 1, roi+1)
        plt.plot(tv_behavior, behavior_array[:, roi] / np.max(behavior_array[:, roi]), color="navy")
        plt.plot(tv_behavior, binary_behavior[:, roi], color="red", alpha=0.2)
        plt.title(ROI_names[roi])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if roi < 5:
            plt.xticks([])
    plt.xlabel("time [s]")
    plt.title(exp[:10])
    plt.tight_layout()
    plt.show()

def plot_raster(spiketimes, valid_cluster_indx, unit_colors, n_spikes = 2000, xlims = []):
    n_spikes = 2000

    spikes_to_plot = np.zeros((len(valid_cluster_indx), n_spikes))
    for n, n_val in enumerate(valid_cluster_indx): 
        spikes_to_plot[n,:] = spiketimes[n_val][:n_spikes] 
    
    plt.figure(figsize=(5,7))
    plt.eventplot(spikes_to_plot, colors=unit_colors, linewidths=0.5)
    plt.ylabel("unit n")
    plt.xlabel("time [s]")
    if xlims:
        plt.xlim(xlims)
    plt.show()

def plot_activity(behavior_array, spike_counts, colors, ROI_names, exp, thresholds, n_ROIs = 6):
    colors = np.array(colors)
    binary_behavior = get_binary_behavior(behavior_array, thresholds)

    plt.figure(figsize=(10,6))
    for roi in range(n_ROIs):
        
        spike_counts_behavior = np.zeros((spike_counts.shape[0], 2))
        spike_counts_behavior[:, 0] = np.mean(spike_counts[:, binary_behavior[:, roi] == 0], axis = 1)
        spike_counts_behavior[:, 1] = np.mean(spike_counts[:, binary_behavior[:, roi] == 1], axis = 1)
        behaving_percentage = sum(binary_behavior[:, roi] == 1) / len(binary_behavior[:, roi]) * 100

        plt.subplot(2, 3, roi + 1)
        plt.scatter(spike_counts_behavior[:, 0], spike_counts_behavior[:, 1], 
                    edgecolors="none",c=colors, marker="o", alpha=0.5)

        plot_lim = 0.5
        plt.plot([0, plot_lim], [0, plot_lim], linestyle="--", color="gray")
        plt.legend([str(np.round(behaving_percentage, 2)) + "%"])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlim([1e-3, plot_lim])
        plt.ylim([1e-3, plot_lim])
        plt.xscale("log")
        plt.yscale("log")
        if roi in [3, 4, 5]:
            plt.xlabel("Not behaving")
        if roi in [0, 3]:
            plt.ylabel("Behaving")
        plt.title(ROI_names[roi])
        #plt.yticks([])

    plt.suptitle("Experiment day " + exp[:10])
    plt.tight_layout()
    plt.show()

def plot_scatter_of_GLM(x, y, colors, coefs_to_plot):
    plt.scatter(x, y, c = colors, alpha = 0.4, edgecolors="none")

    plt.xlim(-max(abs(min(x)), abs(max(x))), max(abs(min(x)), abs(max(x))))
    plt.ylim(-max(abs(min(y)), abs(max(y))), max(abs(min(y)), abs(max(y))))

    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)

    plt.gca().spines['left'].set_position('zero')
    plt.gca().spines['bottom'].set_position('zero')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')

    plt.xlabel(f"{coefs_to_plot[0]} $\\beta$", labelpad=120)
    plt.ylabel(f"{coefs_to_plot[1]} $\\beta$", labelpad=150)
    plt.savefig('scatter' + str(coefs_to_plot)+'.pdf',dpi = 200)
    plt.show()

def plot_histogram_of_GLM(GLM_coefs, coef_names, colors, bins = np.array([]), 
                          n_coef = 6, exp = '2023-03-15_11-05-00', save = True):
    colors = np.array(colors)
    if bins.size == 0:
        bins = np.linspace(np.min(GLM_coefs), np.max(GLM_coefs), 50)

    plt.figure(figsize=(6,10))
    for roi in range(n_coef):
        for color in np.unique(colors):
            indx =  np.argwhere(colors == color).flatten()
            
            plt.subplot(n_coef, 1, roi + 1)
            plt.hist(GLM_coefs[indx, roi], bins = bins, histtype='step', 
                     stacked=True, fill=False, edgecolor=color, density=True)
            plt.title(coef_names[roi])

        plt.vlines(0, 0, plt.gca().get_ylim()[1], linestyles="--", color = "black", alpha=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if roi < n_coef - 1:
            plt.xticks([])
        if roi == 0:
            plt.legend(["n_BW = " + str(np.argwhere(colors == 'black').flatten().shape[0]),
                        "n_TCA = " + str(np.argwhere(colors == 'purple').flatten().shape[0]),
                        "n_NW = " + str(np.argwhere(colors == 'red').flatten().shape[0])])
    
    plt.suptitle("Experiment day " + exp[:10])
    plt.xlabel(f"$\\beta$")
    plt.tight_layout()

    if save:
        save_name = 'histogram ' + str(exp) + '.pdf'
        plt.savefig(os.path.join('plots', save_name),dpi = 200)

    plt.show()

def plot_visual(visual_times, visual_stim_types, sampling_rate = 30000):
    colors = {
    "chirp": "#e6194B",       # strong red
    "Sd36x22_l_3": "#3cb44b", # green
    "Sl36x22_d_3": "#4363d8", # blue
    "mb": "#f58231",          # orange
    "csd": "#911eb4",         # purple
    "cm_18x11_1": "#46f0f0",  # cyan
    "mg_sq": "#f032e6"        # magenta
    }
    for i, stim in enumerate(visual_stim_types):
        plt.eventplot(visual_times[i] / sampling_rate, color=colors[stim],label=stim)
    plt.legend()
    plt.show()
    """
    for i,vtype in enumerate(visual_stim_types):
        plt.plot(visual_array[i,:],label=vtype)
    plt.legend()
    plt.show()
    """
    