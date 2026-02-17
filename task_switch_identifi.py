"""
task_switch_identifi.py

A file loading dataset after preprocessing. Identifying task switching points for all files under the floder path.
Save those time points into a json/numpy file with dataset file name.

Author: Jinpeng Deng
"""


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, find_peaks, welch
from scipy.ndimage import generic_filter
from joblib import Parallel, delayed
import os
import mne
from mne_bids import BIDSPath, read_raw_bids
import json


"""
Step 1: Use Bartholomew's distance to identify potential task switching points.

Calculate whether the Bhattacharyya distance between 500 milliseconds before and after a given time point exceeds the average of the previous 10 seconds plus 3SD, using 40-millisecond intervals.  
Save the time point into the list

"""


def bhattacharyya_distance(psd1, psd2):
    """
    Calculate the Bhattacharyya distance between two discrete distributions (power spectra).
    psd1 and psd2 should be two power spectral density arrays
    """
    
    # Normalization to probability distribution
    p = psd1 / (np.sum(psd1) + 1e-10)
    q = psd2 / (np.sum(psd2) + 1e-10)
    #bc = 1 mean totally same, bc = 0 mean totally different
    bc = np.sum(np.sqrt(p * q))
    #bhattacharyya distance=-ln(bc)
    return -np.log(bc + 1e-10)


#Traverse the entire dataset, save the bd_scores with time points

def worker(i, eeg_data_numpy, window_len, sfreq):
    front_window = eeg_data_numpy[:, i-window_len:i]
    back_window = eeg_data_numpy[:, i:i+window_len]
    
    _, psd_front = welch(front_window, fs=sfreq, nperseg=window_len)
    _, psd_back = welch(back_window, fs=sfreq, nperseg=window_len)
    
    avg_psd_front = np.mean(psd_front, axis=0)
    avg_psd_back = np.mean(psd_back, axis=0)
    
    return bhattacharyya_distance(avg_psd_front, avg_psd_back)


def detect_task_switch_by_bhattacharyya_with_better_CPU(eeg_data_frame, sfreq=256):
    """
    Args:
        eeg_data_frame (_type_): pandas dataframe (Row: Time point, Column: EEG channel)
        gfp (_type_): numpy array of every time point's gfp
        sfreq (int, optional): 256hz. Defaults to 256.
    return:
        A list of time points that may be task switch time point
    """
    entire_time_len = len(eeg_data_frame)
    
    #use 40ms as the step length of window
    step_len = int(0.040 * sfreq)
    #use 500 ms as the window length
    window_len = int(0.5 * sfreq)
    
    #Use 10s as the baseline length as filter
    #Later use 10s (mean + 3SD) to filt
    baseline_len = int(10 * sfreq)
    
    
    #Trans pandas dataframe to numpy array to calculate quicker
    #Filp to row is channels, column is time points
    eeg_data_numpy = eeg_data_frame.values.T #Now the shape is (channel, time)
    
    #Do not use for loop anymore. 
    task_indices = range(window_len, entire_time_len - window_len, step_len)
    
    bd_results = Parallel(n_jobs=-1)(
        delayed(worker)(i, eeg_data_numpy, window_len, sfreq) 
        for i in task_indices
    )
    
    df_results = pd.DataFrame({
        'time_point': list(task_indices),
        'bd': bd_results
    })
    
    window_count = baseline_len // step_len
    
    
    df_results['prev_mean'] = df_results['bd'].rolling(window=window_count).mean().shift(1)
    df_results['prev_std'] = df_results['bd'].rolling(window=window_count).std().shift(1)
    df_results['threshold'] = df_results['prev_mean'] + 3 * df_results['prev_std']

    # Filtering: 
    # 1. Time greater than 10 seconds; 
    # 2. bd value exceeds the threshold.
    mask = (df_results['time_point'] >= baseline_len) & (df_results['bd'] > df_results['threshold'])
    candidate_time_points = df_results.loc[mask, 'time_point'].tolist()
    
    return candidate_time_points



"""
Step 2: Alpha and theta check
Check 800 ms before and after the time points from step 1. Check average power of envelope for alpha and theta. 
If average of aplha increase/decrease more than 30%, theta increase/decrease more than 20% and increase and decrease are opposite. 
The it detect as an task switch. (In compare 200ms before and after time points) Then return the change range. 
"""

#Calculate the average power for a specific frequency(alpha/theta) band using the Welch method.
def get_multi_channel_band_power(eeg_data, sfreq, band):

    f, psd = welch(eeg_data, fs=sfreq, nperseg=eeg_data.shape[1], axis=-1)
    idx = np.logical_and(f >= band[0], f <= band[1])
    band_psd = psd[:, idx]
    return np.mean(band_psd)

#Get the envelope of alpha and theta band
def butter_bandpass_filter(data, low, high, sfreq=256, order=4):
    nyq = 0.5 * sfreq
    low_cut = low / nyq
    high_cut = high / nyq
    b, a = butter(order, [low_cut, high_cut], btype='band')
    return filtfilt(b, a, data, axis=0)

#Calculate the envelope of each channel.
def get_combined_envelope_sq(df, low, high, fs):
        data = df.values.T # (channels, time)
        
        # Bandpass filtering is performed on each channel.
        filtered = butter_bandpass_filter(data.T, low, high, sfreq=fs).T
        
        # Calculate the Hilbert envelope for each channel.
        envelopes = np.abs(hilbert(filtered))
        
        #Since power is V^2 and envelope is V, we need to square the envelope if we want to use the same standard value in paper.
        # Take the average along the channel dimension, then square it to obtain the energy trend.
        return np.mean(envelopes, axis=0)**2


def verify_alpha_theta_2(eeg_data_frame, candidate_time_points, sfreq=256):
    """
    Args:
        candidate_time_points (list): A list of possible task switch times point
        eeg_data_frame: A pandas dataframe of EEG data
        sfreq (int): Sampling frequency of the EEG data(256hz)
    """
    
    entire_time_len = len(eeg_data_frame)
    
    
    #First get alpha and theta envelope square
    alpha_envelope_sq = get_combined_envelope_sq(eeg_data_frame, 8, 13, sfreq)
    theta_envelope_sq = get_combined_envelope_sq(eeg_data_frame, 4, 8, sfreq)
    
    #Set list save verified time points
    verified_segments = []
    #The window size is 200ms
    check_window = int(0.2 * sfreq)
    #The range is 800ms before and after the candidate time point
    half_range = int(0.8 * sfreq)
    #use 40ms as the step length of window
    step_len = int(0.040 * sfreq)
    
    for time_point in candidate_time_points:
        search_start = max(0, time_point - half_range)
        search_end = min(entire_time_len, time_point + half_range)
        
        #Set the variable
        coarse_start = None
        
        for time in range(search_start, search_end - check_window, step_len):
            
            alpha_pre = np.mean(alpha_envelope_sq[time : time+check_window])
            alpha_post = np.mean(alpha_envelope_sq[time+check_window : time+2*check_window])
            theta_pre = np.mean(theta_envelope_sq[time : time+check_window])
            theta_post = np.mean(theta_envelope_sq[time+check_window : time+2*check_window])
            
            alpha_ratio = (alpha_post - alpha_pre) / (alpha_pre + 1e-10)
            theta_ratio = (theta_post - theta_pre) / (theta_pre + 1e-10)
            
            if abs(alpha_ratio) >= 0.3 and abs(theta_ratio) >= 0.2:
                if ((alpha_ratio * theta_ratio) < 0):
                    coarse_start = time+check_window
                    
            
            #If found the coarse start. We want to find more details of when the task swich start and end
            #Try to find the peak of alpha and theta gradient
            if coarse_start is not None:
                #return this 200 range of alpha and theta change range
                verified_segments.append((time, time+check_window))
            
                break 
                # #Set search range be 400 ms for when alpha and theta gradient peak
                # search_start = max(0, coarse_start - int(0.4 * sfreq))
                # search_end = min(entire_time_len, coarse_start + int(0.4 * sfreq))
                
                # alpha_grad = np.abs(np.diff(np.sqrt(alpha_envelope_sq[search_start: search_end])))
                # theta_grad = np.abs(np.diff(np.sqrt(theta_envelope_sq[search_start: search_end])))
                
                # if len(alpha_grad) > 0 and len(theta_grad) > 0:
                #     precise_alpha = np.argmax(alpha_grad) + search_start
                #     precise_theta = np.argmax(theta_grad) + search_start
                #     verified_segments.append((min(precise_alpha,precise_theta), max(precise_alpha,precise_theta)))
    return verified_segments


"""
## Step 3:Check by GFP

3 Part in this function:  
Part A: 
    There exist lowest GFP in the range of (300ms before task switch start) to (100ms after task switch end) compare to other time. 
    The reason behaind is before task switch brand will shut down most part of brain. The range base on alpha and theta not 100% accurate, so use more time to make sure. 
    Therefore check the lowest GFP in the range of (1500ms before task switch start) to (300ms after task switch end) also in (300ms before task switch start) to (100ms after task switch end).  
Part B: 
    Check the mean of lowest GFP in the range of (300ms before task switch start) to (task switch end). 
    The GFP decrease before task switch start and increase after task switch end.  
    So the mean of 1500ms before task switch start to 300ms should be less than then mean of (300ms before task switch start)'s GFP
Part C: 
    From paper, before the task switch, there will be a 20hz to 50hz GFP decrease trend,so try to find the decrease trend exist in (300ms before task switch start) to (task switch end).
"""

#Calculates the GFP (Global Field Power) of an EEG signal.
def calculate_GFP(eeg_data_frame, sfreq=256):
    """
        Calculate the Global Field Power (GFP) of an EEG data.
        Assume data structure is Pandas DataFrame with columns as channels and rows as time points.
    """
    #GFP is the SD of every time point across all channels(column)
    #Count and trans to numpy array
    gfp = eeg_data_frame.std(axis=1).values
    
    #Use 30ms as the smoothing range
    smooth_range = int(0.03 * sfreq)
    # Simple moving about 43ms smoothing removes extremely high frequency spikes
    #WHY: 
        #Assume noise can't 100% remove, so noise can make spikes happen.
        #Make spikes more obvious by averaging them out.
    smooth_gfp = np.convolve(gfp, np.ones(smooth_range)/smooth_range, mode='same')
    
    return smooth_gfp

def gfp_check(candidate_task_switch, smooth_gfp, sfreq=256):
    #Set variable
    verified_segments = []
    
    #Set task switch range before and after. 300ms before start time and 100ms after end time.
    before_check_range = int(0.3 * sfreq)
    after_check_range = int(0.1 * sfreq)
    
    
#Can change later after discussing 
    #The check lowest GFP range. 1500 ms before task switch until 300ms after task switch end.
        #For alpha and theta, it may change in 200ms. But consider the task swich is happened in undreds to thousands of milliseconds,
            #So we set the check range to 1500ms before and 300ms later.
    check_lowest_range_before = int(1.5*sfreq)
    check_lowest_range_after = int(0.3*sfreq)
    
    #Check average GFP in the check range is decrease(350hs)
    pre_avg_smp = int(0.350 * sfreq)
    
    #use 25ms as the minumum decrease trend scope
    decrease_scope = int(0.025 * sfreq)
    
    
    for start, end in candidate_task_switch:
        #Assume large task switch not happen in the first 1500ms
        if start < check_lowest_range_before:
            continue
        
        #Part A:
        #Set the actuall range of evaluation lowest GFP
        lowest_GFP_check_start = start - check_lowest_range_before
        lowest_GFP_check_end = min(len(smooth_gfp),end + check_lowest_range_after)

        #Check the lowest GFP in the range
        gfp_check_data = smooth_gfp[lowest_GFP_check_start : lowest_GFP_check_end]
        #Find the lowest GFP in the range
        lowest_gfp = np.argmin(gfp_check_data) + lowest_GFP_check_start
        
        
        start_check_point = start - before_check_range
        end_check_point = min(len(smooth_gfp), end + after_check_range)
        #Check the GFP lowest point of (1500ms+start) to (300ms+end) is also in (300ms+start) to (100ms+end)
        is_lowest_in_task_switch = (start_check_point <= lowest_gfp <= end_check_point)
        
        #If the lowest GFP is not in the task switch range, continue to next candidate
        if not is_lowest_in_task_switch:
            continue
        
        
        #Part B:
        #Then, check the average GFP near to start are less than far way to start.
        #In this function, compare 1500ms to 350ms before start time and 350ms before start time
        average_futher_GFP = np.mean(smooth_gfp[start - check_lowest_range_before : start - pre_avg_smp])
        average_near_GFP = np.mean(smooth_gfp[start - pre_avg_smp : start])
        
        if average_near_GFP > average_futher_GFP:
            continue
    

        #Part C:
        #Check is there exits 25ms GFP decrease trend in 300hs before start to end time.
        
        #Consider the range is same with find lowest GFP, so just use the variable before
        search_trend_data = smooth_gfp[start_check_point : end]
        diffs = np.diff(search_trend_data)
        
        find_decrease_trend = False
        is_decrease = (diffs <= 0).astype(int)
        
        #Find data decrease trend in 25ms
        if np.max(np.convolve(is_decrease, np.ones(decrease_scope), mode='valid')) >= decrease_scope:
            find_decrease_trend = True
        
        if find_decrease_trend:
            verified_segments.append((start, end))
    
    return verified_segments
        

"""
Step 4:Merge back to back and overlap task switch time point
"""
#A function for merge all back to back task switch
def merge_back_to_back(task_switch):
    #Set variable
    merged_segments = []
    current_start, current_end = task_switch[0]
    total = len(task_switch)
    i=1
    
    while i < total:
        if task_switch[i][0] <= current_end:
            current_end = max(task_switch[i][1], current_end)
            i+=1
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = task_switch[i]
            i += 1
    if merged_segments[len(merged_segments)-1][0] != current_start:
        merged_segments.append((current_start, current_end))
    
    return merged_segments
        

"""
read all the dataset file under the path.
"""

def generate_task_switch_file(path='./on', sfreq=256, BIDS_export_path = "task_switch_labels.json"):
    
    """
    path: The path of the dataset floder
    sfreq: The sampling frequency of the dataset(default is 256Hz)
    BIDS_export_path: The path of the exported task switch file(default is "task_switch_labels.json")
    
    """

    #First getting the absolute path of the dataset file
    base_directory = os.getcwd()
    search_target = os.path.abspath(os.path.join(base_directory, path))
    
    all_results = []
    
    #If want to use absolute path, use the following code, if want to use relative path, use the "path "instead of "search_target"
    for root, directory, files in os.walk(search_target):
        for each_file in files:
            if each_file.endswith((".edf", ".bdf", ".set", ".fif")):
                #Get the full path of the dataset file
                abs_file_path = os.path.join(root, each_file)

                #Get the relative path of the dataset file for saving the "name"
                relative_path = os.path.relpath(abs_file_path, base_directory)
                
                try: 
                    #Loding data
                        #Assume data are already been preprocessing
                    raw_data = mne.io.read_raw(abs_file_path, preload=True, verbose=False)
                    
                    #Resample if the sfreq isn't the number we want
                    if raw_data.info['sfreq'] != sfreq:
                        raw_data = raw_data.resample(sfreq)
                        
                    #Trans eeg data to pandasdataframe
                    df_eeg = raw_data.to_data_frame(picks='eeg')
                    
                    #The index of eeg data in the list already contain the time information
                    if 'time' in df_eeg.columns:
                        df_eeg = df_eeg.drop(columns=['time'])
                    
                    #Using the function. 
                    candidates = detect_task_switch_by_bhattacharyya_with_better_CPU(df_eeg, sfreq=sfreq)
                    smooth_gfp = calculate_GFP(df_eeg, sfreq=sfreq)
                    verified_at_segments = verify_alpha_theta_2(df_eeg, candidates, sfreq=sfreq)
                    final_segments = gfp_check(verified_at_segments, smooth_gfp, sfreq=sfreq)
                    after_merge = merge_back_to_back(final_segments)
                    
                    #Export the task switch file, save in json file. 
                    clean_task_switch = [list(item) if isinstance(item, tuple) else item for item in after_merge]
                    
                    all_results.append({
                        "name": relative_path, 
                        "task_switch": clean_task_switch
                    })
                    
                except Exception as e:
                    print(f"Error in {relative_path}: {e}")
    with open(BIDS_export_path, 'w') as f:
        json.dump(all_results, f, indent=4)

generate_task_switch_file("./data/sleep", 100, "./task_switch_labels.json")