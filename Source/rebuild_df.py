"""
rebuild_df.py
Reconstruct dummy and feather

Author
   Julia Wutschka

Usage
   This script should be called from main.py.
   **Running from command line**
   You may run it from command line: python rebuild.py <BASEFILENAME>
   Arguments:
       * BASEFILENAME
         prefix of the test file including the (the code's relative) path
   Example:
        python rebuild.py "Input\C3_19052022\1BEE\190604032" "_1BEE_nn.csv"

Files
    **BASEFILENAME.json**
        We get necessary values from and put calculated values in BASEFILENAME.json. It is necessary for calculations
        within this script it.
        Values put are:
        *   *ddf_finished-file-name*
            CSV-file containing the rebuilt data. Default it is base_name + "_ddf_finished.csv"
        *   *arena_radius*
            Radius of the arena. If it is not set, the value of global_constants.arena_dummy_relation * center_radius
            is used.
        *   *plot_rotation_approximation*
            Name of the .png showing the approximation. Default it is base_name + "_plot_rotation_approximation.png"
"""
import json
import math
import pandas
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pandas as pd
from termcolor import colored
import sys
from _socket import gethostname

import geolib
import global_constants

#
# Global constants
#
threshold_peak = 0.90
# if y-value >0.9 or <-0.9 then we assume we have a peak
threshold_max_acceptable = 1.3
# if y-value >1.3 then this is not a peak but a wrong value
threshold_sequence_length = 0.3
# next sequence must be 70%-130% (100% +/- 30% of last sequence)
average_weight = 0.7
# Last average is weighted with 0.7, new value (sequence) with 0.3


#
# Setup processing:
#
def main(base_name, wrapper: global_constants.Wrapper_class):
    """
    Main function of rebuild.py

    :param base_name: prefix of the test file including the (the code's relative) path
    :param wrapper: wrapper-object containing values should be passed by reference. see global_constants.py
    :return: -
    """
    info_filename = base_name + ".json"
    #
    # Load globals
    #
    try:
        with open(info_filename, "r") as in_file:
            info_block = json.load(in_file)
            in_file.close()
    except IOError:
        raise Exception("File ", info_filename, "does not exist. We will need it for processing...")
    finally:
        pass

    if not ("ddf_finished-file-name" in info_block):
        info_block["ddf_finished-file-name"] = base_name + "_ddf_finished.csv"
    if not ("arena_radius" in info_block):
        info_block["arena_radius"] = info_block["center_radius"] * global_constants.arena_dummy_relation
    if not ("plot_rotation_approximation" in info_block):
        info_block["plot_rotation_approximation"] = base_name + "_plot_rotation_approximation.png"
    center_point = geolib.Vector(info_block["center_x"], info_block["center_y"])
    ddf_sorted = wrapper.ddf_sorted
    if wrapper.fixed_start_frame is None and "fixed_start_frame" in info_block:
        wrapper.fixed_start_frame = info_block["fixed_start_frame"]

    #
    # Helper functions
    #

    def find_peaks(row_name):
        """
        Create the list of peaks of the given field-name
        :param row_name: name of the field
        :return: list-object with frame numbers representing a peak
        """
        nonlocal source_dataframe
        global threshold_peak, threshold_max_acceptable
        peaks = list()
        in_peak_area = False
        is_acceptable = True
        start_frame_number = -1
        last_inside = -1
        for fp_idx, current_row in source_dataframe.iterrows():
            if current_row[row_name] > threshold_peak and (not in_peak_area) and start_frame_number != -1:
                start_frame_number = current_row["frame_number"]
                in_peak_area = True
                is_acceptable = True
            elif current_row[row_name] > threshold_peak and in_peak_area:
                last_inside = current_row["frame_number"]
            if current_row[row_name] > threshold_max_acceptable:
                is_acceptable = False
            elif current_row[row_name] <= threshold_peak:
                if in_peak_area:
                    if is_acceptable:
                        peak = int((last_inside - start_frame_number) / 2) + start_frame_number
                        peaks.append(peak)
                    in_peak_area = False
                elif start_frame_number == -1:
                    start_frame_number = 0
        return peaks

    def next_frame_index(min_frame_number, frame_list, direction=1):
        """
        Get the next index of the frame list
        :param min_frame_number: the value of the reference frame
        :param frame_list: list-object of frame numbers to search. frame_list must be ordered ascending.
        :param direction: if direction ==1 then search in ascending order else descending order
        :return: if direction = 1: return the index of the first element of frame_list
            where frame_list[index] > min_frame_number
            if direction = -1, return the index of the last element of frame_list
            where frame_list[index] < min_frame_number
        """
        i = 0
        while i < len(frame_list) - 1 and frame_list[i] < min_frame_number:
            i += 1
        if direction == 1:
            return i, i >= len(frame_list) - 1
        else:
            return (i - 1) if i > 0 else 0, i == 0

    def get_sequences(col_name):
        """
        Build up a list which contains the sequences of given points
        :param col_name: name of the data field
        :return: list-object of tuples (type, start_frame, frame count)
            * type
              true, is the tuple represents a sequence of nan-values, false otherwise
            * start_frame
              number of the frame representing the start of the sequence
            * frame count
              length of the sequence
        """
        nonlocal result_dataframe
        sequence_list: list[tuple[bool, int, int]] = []
        start_frame_count = -1
        frame_counter = 0
        current_sequence_isnan = True
        for fp_idx, current_row in result_dataframe.iterrows():
            if start_frame_count == -1 or (current_sequence_isnan != pandas.isna(current_row[col_name])):
                if start_frame_count != -1:
                    sequence_list.append((current_sequence_isnan, start_frame_count, frame_counter))
                start_frame_count = current_row["frame_number"]
                frame_counter = 1
                current_sequence_isnan = pandas.isna(current_row[col_name])
            else:
                frame_counter += 1
        sequence_list.append((current_sequence_isnan, start_frame_count, frame_counter))
        return sequence_list

    def approximate_curve(src_col_name, src_neg_col_name, dest_col_name, dest_freq_col_name, dest_angel_col_name):
        """
        Calculate the sine-approximation for a given data column
        :param src_col_name: name of column which contains input values range -1 to 1
        :param src_neg_col_name: name of column which contains negated input values
        :param dest_col_name: name of column where to store the approximated values
        :param dest_freq_col_name: name of column where to store the current frequency (sequence length)
        :param dest_angel_col_name: for futere use...
        :return:
        """
        nonlocal result_dataframe

        seq_qual = []
        seq_pd = pandas.DataFrame(get_sequences(src_col_name), columns=["isnan", "frame_number", "length"])
        seq_pd = seq_pd[~seq_pd["isnan"]]
        seq_pd_count10 = int(seq_pd.shape[0] * 0.1)
        seq_pd = seq_pd.tail(seq_pd_count10 * 9)
        seq_pd = seq_pd.head(seq_pd_count10 * 8)
        for idx, seq in seq_pd.iterrows():
            rel_seq = seq_pd[(seq_pd["frame_number"] > seq["frame_number"]) &
                             (seq_pd["frame_number"] < (seq["frame_number"] + 1000))]
            seq_qual.append((seq["frame_number"], rel_seq["length"].sum() * 1.0))
        best_start_seq = max(seq_qual, key=lambda s: s[1])
        start_frame = best_start_seq[0]
        peak_lists = [find_peaks(src_col_name), find_peaks(src_neg_col_name)]
        p_count = [len(peak_lists[0]), len(peak_lists[1])]
        if p_count[0] < 2 or p_count[1] < 2:
            raise Exception("Too less peaks found")
        shifts = [-1, 1]
        if wrapper.fixed_start_frame is not None:
            start_frame = wrapper.fixed_start_frame
        next_peak = [next_frame_index(start_frame, peak_lists[0]), next_frame_index(start_frame, peak_lists[1])]
        start_frame = min(peak_lists[0][next_peak[0][0]], peak_lists[1][next_peak[1][0]])
        print("Start frame for", src_col_name, "is", start_frame)
        start_list = 1 if start_frame == peak_lists[0][next_peak[0][0]] else 0
        for direction in [1, -1]:
            reference_frame_number = start_frame
            current_list = start_list
            next_peak = [next_frame_index(reference_frame_number + direction, peak_lists[0], direction),
                         next_frame_index(reference_frame_number + direction, peak_lists[1], direction)]
            if not next_peak[current_list][1] and direction == 1 or next_peak[current_list][1] and direction == -1:
                current_sequence = (peak_lists[current_list][(next_peak[current_list][0])] - reference_frame_number) * 2
            else:
                current_sequence = (reference_frame_number - peak_lists[current_list][(next_peak[current_list][0])]) * 2
            current_sequence = current_sequence * direction
            avg_sequence = current_sequence
            if direction == 1:
                current_df = result_dataframe[result_dataframe["frame_number"] >= start_frame]
            else:
                current_df = result_dataframe[result_dataframe["frame_number"] < start_frame]
                current_df.set_index("frame_number")
                current_df = current_df.reindex().sort_index(ascending=False)
            for i, f in current_df.iterrows():
                fn = f["frame_number"]
                result_dataframe.at[i, dest_col_name] = shifts[current_list] * math.cos(
                    (fn - reference_frame_number) * direction / current_sequence * (2 * math.pi))
                result_dataframe.at[i, dest_angel_col_name] = ((fn - reference_frame_number) / current_sequence *
                                                               (2 * math.pi) +
                                                               (math.pi if shifts[current_list] == -1 else 0))
                result_dataframe.at[i, dest_freq_col_name] = current_sequence
                next_peak_reached = (not next_peak[current_list][1] and
                                     ((fn > peak_lists[current_list][next_peak[current_list][0]] and direction == 1) or
                                     (fn < peak_lists[current_list][next_peak[current_list][0]] and direction == -1)))
                if next_peak_reached:
                    reference_frame_number = fn
                    other_list = (current_list + 1) % 2
                    min_distance = int(current_sequence * (1 - threshold_sequence_length)) / 2
                    np_current = next_frame_index(fn + min_distance * 2, peak_lists[current_list], direction)
                    np_other = next_frame_index(fn + min_distance, peak_lists[other_list], direction)
                    other_sequence = current_sequence
                    if not np_other[1]:
                        next_peak[other_list] = np_other
                        other_sequence = (peak_lists[other_list][next_peak[other_list][0]] - fn) * 2
                    my_sequence = peak_lists[current_list][np_current[0]] - fn
                    next_peak[current_list] = np_current
                    if (math.fabs(other_sequence / avg_sequence - 1.0) < threshold_sequence_length and
                            peak_lists[other_list][next_peak[other_list][0]] <
                            peak_lists[current_list][next_peak[current_list][0]]):
                        np_current = next_frame_index(peak_lists[other_list][next_peak[other_list][0]],
                                                      peak_lists[current_list], direction)
                        if not np_current[1]:
                            next_peak[current_list] = np_current
                        current_list = (current_list + 1) % 2
                        current_sequence = other_sequence
                    else:
                        current_sequence = my_sequence
                        reference_frame_number -= int(current_sequence / 2) * direction
                    avg_sequence = avg_sequence * average_weight + current_sequence * (1 - average_weight)

    total_time = time()
    #
    # Step 1: Reading source date
    #
    start_time = time()
    source_dataframe = pd.DataFrame()
    if ddf_sorted.empty:
        if not ("tmp_dummy_ready" in info_block):
            raise "Neither source data or source data file name is given"
            
        print("Reading source data from", info_block["tmp_dummy_ready"])
        source_dataframe = pandas.read_csv(info_block["tmp_dummy_ready"])
    else:
        source_dataframe = ddf_sorted

    source_dataframe = source_dataframe.sort_values("frame_number")

    source_dataframe['Da_x'] = (source_dataframe['D_x'] - info_block["center_x"]) / info_block["center_radius"]
    source_dataframe['Da_y'] = (source_dataframe['D_y'] - info_block["center_y"]) / info_block["center_radius"]
    source_dataframe["Da_-x"] = - source_dataframe["Da_x"]
    source_dataframe["Da_-y"] = - source_dataframe["Da_y"]

    min_frame_number = min(source_dataframe["frame_number"].min(), info_block["bee-track-first-frame"])
    max_frame_number = max(source_dataframe["frame_number"].max(), info_block["bee-track-last-frame"])

    result_dataframe = pandas.merge(pandas.DataFrame(list(range(min_frame_number, max_frame_number)),
                                                     columns=["frame_number"]),
                                    source_dataframe,
                                    how="left", left_on="frame_number", right_on="frame_number")
    result_dataframe.set_index("frame_number")
    result_dataframe["Dc_x"] = 0.0
    result_dataframe["Dc_y"] = 0.0
    result_dataframe["Draw_x"] = result_dataframe["D_x"]
    result_dataframe["Draw_y"] = result_dataframe["D_y"]
    result_dataframe["DHraw_x"] = result_dataframe["DH_x"]
    result_dataframe["DHraw_y"] = result_dataframe["DH_y"]
    result_dataframe["d_dh_angel"] = 0.0
    result_dataframe["F1_rel_dist"] = 0.0
    result_dataframe["F1_rel_o"] = 0.0
    result_dataframe["F1_is_approx"] = 0
    result_dataframe["F1raw_x"] = result_dataframe["F1_x"]
    result_dataframe["F1raw_y"] = result_dataframe["F1_y"]
    result_dataframe["F2_rel_dist"] = 0.0
    result_dataframe["F2_rel_o"] = 0.0
    result_dataframe["F2_is_approx"] = 0
    result_dataframe["F2raw_x"] = result_dataframe["F2_x"]
    result_dataframe["F2raw_y"] = result_dataframe["F2_y"]
    result_dataframe["freq"] = 0.0

    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds read the source data.", "light_grey",
                  force_color=True))
    #
    # Step 2: Approximate rotation
    # Try to interpolate sinus-curves from peak-value to peak-value
    #
    start_time = time()
    print("Approximating rotation")
    approximate_curve("Da_x", "Da_-x", "Dc_x", "fx_0", "d_angel_x")
    approximate_curve("Da_y", "Da_-y", "Dc_y", "fy_0", "d_angel_y")

    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds to approximate the dummy rotation (points D and DH). \n", "light_grey",
                  force_color=True))
    #
    # Step 3: Feather-Holder - Smooth values and set x and y according to the center point and the radius
    # Feather: calculate feather points in relation to the feather holder
    # x^2 + y^2 should be 1
    #
    start_time = time()
    print("Reconstructing points D and DH")
    for idx, rdf_row in result_dataframe.iterrows():
        corr_f = 1 / math.sqrt(rdf_row["Dc_x"] * rdf_row["Dc_x"] + rdf_row["Dc_y"] * rdf_row["Dc_y"])
        d_x = info_block["center_x"] + info_block["center_radius"] * corr_f * rdf_row["Dc_x"]
        d_y = info_block["center_y"] + info_block["center_radius"] * corr_f * rdf_row["Dc_y"]
        dh_x = info_block["center_x"] - info_block["center_radius"] * corr_f * rdf_row["Dc_x"]
        dh_y = info_block["center_y"] - info_block["center_radius"] * corr_f * rdf_row["Dc_y"]
        d_dh_o = geolib.Straight(geolib.Vector(d_x, d_y), geolib.Vector(dh_x, dh_y)).get_orientation()
        result_dataframe.at[idx, "D_x"] = d_x
        result_dataframe.at[idx, "D_y"] = d_y
        result_dataframe.at[idx, "DH_x"] = dh_x
        result_dataframe.at[idx, "DH_y"] = dh_y
        result_dataframe.at[idx, "freq"] = (rdf_row["fx_0"] + rdf_row["fy_0"]) / 2.0
        result_dataframe.at[idx, "d_dh_angel"] = d_dh_o
        for feather_name in ["F1_", "F2_"]:
            if not pd.isna(result_dataframe.at[idx, feather_name + "x"]):
                f_x = result_dataframe.at[idx, feather_name + "x"]
                f_y = result_dataframe.at[idx, feather_name + "y"]
                f_rel = geolib.Vector(f_x, f_y).sub(center_point)
                if geolib.Vector(f_x, f_y).sub(geolib.Vector(d_x, d_y)).abs() > info_block["center_radius"] / 5 * 1.5:
                    # set distance only if F not next to the calculated D-point
                    result_dataframe.at[idx, feather_name + "rel_dist"] = f_rel.abs()
                    result_dataframe.at[idx, feather_name + "rel_o"] = geolib.angel_diff(f_rel.get_orientation(), d_dh_o)

    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds to reconstruct values for points D and DH.", "light_grey",
                  force_color=True))
    #
    # Step 4: Feather -  calculate feather points in relation to the feather holder
    #
    start_time = time()
    print("Interpolating missing points of F1 and F2")
    feather_list = ["F1", "F2"]
    in_series = [False, False]
    last_exists = [-1, -1]
    missing_ranges = [list(), list()]
    average_values = [[0.0, 0.0], [0.0, 0.0]]
    for idx, rdf_row in result_dataframe.iterrows():
        for f_ind, feather in enumerate(feather_list):
            if rdf_row[feather + "_rel_dist"] != 0.0:
                if in_series[f_ind]:
                    average_values[f_ind][0] = (average_values[f_ind][0] * global_constants.feather_smoothing +
                                                rdf_row[feather + "_rel_dist"] *
                                                (1.0 - global_constants.feather_smoothing))
                    average_values[f_ind][1] = (average_values[f_ind][1] * global_constants.feather_smoothing +
                                                rdf_row[feather + "_rel_o"] *
                                                (1.0 - global_constants.feather_smoothing))
                    result_dataframe.at[idx, feather + "_rel_dist"] = average_values[f_ind][0]
                    result_dataframe.at[idx, feather + "_rel_o"] = average_values[f_ind][1]
                    last_exists[f_ind] = idx
                else:
                    in_series[f_ind] = True
                    missing_ranges[f_ind].append((last_exists[f_ind], idx))
            else:
                in_series[f_ind] = False
    for f_ind, feather in enumerate(feather_list):
        for missing_range in missing_ranges[f_ind]:
            target_angel = result_dataframe.at[missing_range[1], feather + "_rel_o"]
            target_dist = result_dataframe.at[missing_range[1], feather + "_rel_dist"]
            if missing_range[0] == -1:
                start_angel = target_angel
                start_dist = target_dist
            else:
                start_angel = result_dataframe.at[missing_range[0], feather + "_rel_o"]
                start_dist = result_dataframe.at[missing_range[0], feather + "_rel_dist"]
            step_angel = (target_angel - start_angel) / (missing_range[1] - missing_range[0])
            step_dist = (target_dist - start_dist) / (missing_range[1] - missing_range[0])
            for fn in range(missing_range[0] + 1, missing_range[1]):
                f_rel_angel = start_angel + (fn - missing_range[0]) * step_angel
                f_rel_dist = start_dist + (fn - missing_range[0]) * step_dist
                result_dataframe.at[fn, feather + "_rel_o"] = f_rel_angel
                result_dataframe.at[fn, feather + "_rel_dist"] = f_rel_dist
                f_point = center_point.add(
                    geolib.Vector(phi=f_rel_angel + result_dataframe.at[fn, "d_dh_angel"], absolut=f_rel_dist))
                result_dataframe.at[fn, feather + "_x"] = f_point.x
                result_dataframe.at[fn, feather + "_y"] = f_point.y
                result_dataframe.at[fn, feather + "_is_approx"] = -1

    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds to interpolate values for points F1 and F2.", "light_grey",
                  force_color=True))

    #
    # Step 5: Write CSV
    #
    result_dataframe.to_csv(info_block["ddf_finished-file-name"],
                            columns=["frame_number", "D_x", "D_y", "DH_x", "DH_y",
                                     "Draw_x", "Draw_y", "DHraw_x", "DHraw_y",
                                     "F1_x", "F1_y", "F1_rel_dist", "F1_rel_o", "F1_is_approx", "F1raw_x", "F1raw_y",
                                     "F2_x", "F2_y", "F2_rel_dist", "F2_rel_o", "F2_is_approx", "F2raw_x", "F2raw_y",
                                     "freq"])
    print("Saved to disk as", info_block["ddf_finished-file-name"])

    graph_df = result_dataframe[result_dataframe["frame_number"] > 1900]
    graph_df = graph_df[graph_df["frame_number"] < 2900]
    f, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('raw data')
    ax1 = graph_df.plot(ax=ax1, x="frame_number", y="Da_x", color="r")
    ax1.set_xlabel("frame count")
    ax2 = graph_df.plot(ax=ax2, x="frame_number", y="Da_x", color="r")
    ax2 = graph_df.plot(ax=ax2, x="frame_number", y="Dc_x", color="g")
    ax2.set_xlabel("frame count")
    ax2.set_title('raw data and calculated values')
    f.canvas.manager.set_window_title('Approximation of the rotation')
    f.show()

    graph_df = result_dataframe[result_dataframe["frame_number"] > 1360]
    graph_df = graph_df[graph_df["frame_number"] < 1700]
    f, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('raw data')
    ax1 = graph_df.plot(ax=ax1, x="frame_number", y="Da_x", color="r")
    ax1.set_xlabel("frame count")
    ax2 = graph_df.plot(ax=ax2, x="frame_number", y="Da_x", color="r")
    ax2 = graph_df.plot(ax=ax2, x="frame_number", y="Dc_x", color="g")
    ax2.set_xlabel("frame count")
    ax2.set_title('raw data and calculated values')
    f.canvas.manager.set_window_title('Approximation of the rotation (first values)')
    f.show()

    zeros = []
    rawex = []
    result_dataframe["xrawexc"] = np.nan
    print(zeros)
    for i in result_dataframe.index:
        if i + 280 in source_dataframe.index:
            result_dataframe.at[i, 'xraw2'] = (source_dataframe['xraw'].iloc[i + 280] - info_block["center_x"]) / info_block["center_radius"]
        if i - 323 in source_dataframe.index:
            result_dataframe.at[i, 'xraw3'] = (source_dataframe['xraw'].iloc[i - 323] - info_block["center_x"]) / info_block["center_radius"]
    for i in result_dataframe.index:
        if result_dataframe["Dc_x"].iloc[i] == 0 or (result_dataframe["Dc_x"].iloc[i] >= -0.001 and result_dataframe["Dc_x"].iloc[i] <= 0.001 ) :
            zeros.append(result_dataframe["frame_number"].iloc[i])
        if result_dataframe["xrawexc"].iloc[i] == 0:
            result_dataframe.at[i, "xrawexc"] = np.nan
            rawex.append(result_dataframe["frame_number"].iloc[i])
        if i in source_dataframe.index:
            result_dataframe.at[i, 'xrawexc'] = (source_dataframe['xraw'].iloc[i] - info_block["center_x"]) / \
                                            info_block["center_radius"]
    df_merge = result_dataframe[["frame_number", "xrawexc"]].copy()
    df_merge["Da_x"] = df_merge["xrawexc"]
    df_ex = pd.merge(result_dataframe, df_merge, how="left", on="Da_x", indicator=True)
    df_ex["new"] = np.nan
    for i in df_ex.index:
        if df_ex["_merge"].iloc[i]=="right_only":
            df_ex.at[i, "new"] = df_ex["xrawexc_y"].iloc[i]
    print(df_ex)
    graph_df = result_dataframe
    ax = result_dataframe.plot( x="frame_number", y="xraw3", color="#a5d800")
    ax = graph_df.plot(ax=ax, x="frame_number", y="Da_x", color="grey")
    ax = graph_df.plot(ax=ax, x="frame_number", y="Dc_x", color="black")

    plt.legend([ "excluded raw data", "raw data", "approximated data"], loc="upper right")
    plt.axhline(y=0.9, color="#00aeff")
    plt.axhline(y=-0.9, color="#00aeff")
    ax.fill_between(result_dataframe["frame_number"], 0.9, 1.3, alpha= 0.2, color="#00aeff")
    plt.axhline(y=1.3, color="#00aeff")
    plt.axhline(y=-1.3, color="#00aeff")
    plt.axhline(y=0, color="black", lw=0.5)
    for z in zeros:
        plt.axvline(x=z, color="#ff7000")
    ax.fill_between(result_dataframe["frame_number"], -0.9, -1.3, alpha= 0.2, color="#00aeff")
    plt.title("Sine curve approximation")
    plt.xlabel("Frame")
    plt.ylabel("Normalized x coordinate")

    wrapper.ddf_finished = result_dataframe

    info_block["dummy-rotation-file-start_frame"] = result_dataframe["frame_number"].min().__str__()
    info_block["dummy-rotation-file-last_frame"] = result_dataframe["frame_number"].max().__str__()
    if wrapper.fixed_start_frame is not None:
        info_block["fixed_start_frame"] = wrapper.fixed_start_frame

    with open(info_filename, "w") as outfile:
        json.dump(info_block, outfile)
        outfile.close()

    print("Test-information file is updated", info_filename)
    print("Total time:", round(time() - total_time, 2))
    plt.savefig(info_block["plot_rotation_approximation"])
    if global_constants.unattended_mode:
        plt.show(block=False)
        plt.pause(0.1)
        plt.close('all')
    else:
        plt.show(block=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:  # dem script wird ein parameter übergeben (dateipfad)
        print(colored(f"MISSING BASE NAME! DERIVED NAMES ARE:"
                      f"\n*.json - TEST INFORMATION"
                      f"\n*_DUMMY_ai.csv - RAW CSV DATA (_ai .CSV)"
                      f"\n*_dummy_src_from_raw_ai.csv - PROCESSED DUMMY DATA", "red", force_color=True))
        exit(1)
    base_name = sys.argv[1]
    main(base_name, global_constants.Wrapper_class())
