"""
main.py
Control script to process all test data.

Author
   Julia Wutschka

Usage
   You can use it in two ways:
   * **for dedicated files**
     Uncomment code part named *ALTERNATIVE1*, comment out *ALTERNATIVE2*, and put the file information in the variable basename_list
   * **CSV-batch processing**
     Process all tests listed in a CSV-file. Uncomment code part named *ALTERNATIVE1*, comment out *ALTERNATIVE2*,
     Set the name in the variable *overview_csv_file*.
     You may use scan_directories.py to create an initial CSV-FILE-LIST
"""

# local files
import global_constants
import scan_bee_trace
import dummy
import rebuild_df
#import make_videooverlay_analysis

global_constants.unattended_mode = False
"""
ALTERNATIVE 1
Use following part of the code for one or more datasets (manual run). Put the following information in the list:
    basename    prefix of each test file including the (the code's relative) path
    bee_file    suffix of the bee-data-file
    video_file  suffix of the video-file
    start_frame frame number used as the beginning for the sine curve approximation (default: None)
See example below.
"""
basename_list = [
    (r"input\C3_19052022\1BEE\190604032", "_1BEE_nn.csv", "_1BEE_nn.mp4", None)
]

for bn in basename_list:
    wrapper = global_constants.Wrapper_class()
    wrapper.bee_filename = bn[1]
    wrapper.video_filename = bn[2]
    wrapper.fixed_start_frame = bn[3]
    scan_bee_trace.main(bn[0], wrapper)
    dummy.main(bn[0], wrapper)
    rebuild_df.main(bn[0], wrapper)
    # uncomment next line if you need a video
    # make_videooverlay_analysis.main(bn[0], wrapper)

"""
ALTERNATIVE 2
Use this part of code for batch processing. All dataset names need to be listed in a CSV-File. You should use
the script scan_directories.py to create it. The flags 
beetrace_done (dummy_done, rebuild_df_done) will be set to -1 after bee2.py (dummy.py, rebuild_df.py) finished.
### Uncomment Alternative 2 starting here

overview_csv_file = r"dummy_quality_and_bee\filelist.csv"
path_list = pandas.read_csv(overview_csv_file)
# path_list = path_list[path_list["inspect"] == "x"]
for i, path_entry in path_list.iterrows():
    wrapper = global_constants.Wrapper_class()
    wrapper.bee_filename = path_entry["bee_filename"]
    wrapper.video_filename = path_entry["video_filename"]
    try:
        if path_entry["beetrace_done"] == 0:
            scan_bee_trace.main(path_entry["path"], wrapper)
            path_list.at[i, "beetrace_done"] = 1
        if path_entry["dummy_done"] == 0:
            dummy.main(path_entry["path"], wrapper)
            path_list.at[i, "dummy_done"] = 1
        if path_entry["rebuild_df_done"] == 0:
            rebuild_df.main(path_entry["path"], wrapper)
            path_list.at[i, "rebuild_df_done"] = 1
        if path_entry["make_overlay"] == 1:
            make_videooverlay.main(path_entry["path"], wrapper)
            path_list.at[i, "overlay_done"] = 1
        if path_entry["make_quality_analysis"] == 1 and path_entry["quality_analysis_done"] == 0:
            source_data_quality.main(path_entry["path"], wrapper)
            path_list.at[i, "quality_analysis_done"] = 1
    except Exception as e:
        path_list.at[i, "exceptions"] = e.__str__()
    path_list.to_csv(overview_csv_file)
"""