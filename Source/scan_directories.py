"""
scan_directories.py
Creates a CSV file for batch processing of multiple loopy csv's using main.py.
If a json-file exists the dataset will be ignored. To create a new complete list of all tests you must
delete all json-files (and rerun main.py).

Author
   Julia Wutschka

Usage
   Edit the values for *directory_input* and *overview_csv_file* and run it.
"""

import sys
import os
import re
import pandas

if __name__ == '__main__' and len(sys.argv) == 2:
    base_name = sys.argv[1]


directory_input = r"dummy_and_bee_analysis\input"
overview_csv_file = r"dummy_and_bee_analysis\input\filelist.csv"

directory = os.path.basename(directory_input)
path_list = []
for directories in next(os.walk(directory_input))[1]:
    folder = directories
    for subdirectories in next(os.walk(os.path.join(directory_input, directories)))[1]:
        wd = os.listdir(os.path.join(directory_input, directories, subdirectories))
        json_list = list()
        for filename in wd:
            if filename.endswith(".json"):
                matches = re.findall(r"([\s\S]*)?\.json$", filename)
                if len(matches) == 1:
                    json_list.append(matches[0])
        for filename_input in wd:
            matches = re.findall(r"(\d{9})_(\d)BEE_nn.csv", filename_input)
            if len(matches) == 1:
                if not matches[0][0] in json_list:
                    path_list.append((os.path.join(directory_input, directories, subdirectories, matches[0][0]), matches[0][1], "_"  + matches[0][1] + "BEE_nn.csv", "_" + matches[0][1] + "BEE_nn.mp4"))
df_path = pandas.DataFrame(path_list, columns=["path", "Bees", "bee_filename", "video_filename"])
df_path["beetrace_done"] = 0
df_path["dummy_done"] = 0
df_path["rebuild_df_done"] = 0
df_path["make_overlay"] = 0
df_path["overlay_done"] = 0
df_path["make_quality_analysis"] = 0
df_path["quality_analysis_done"] = 0
df_path["exceptions"] = ""
df_path["inspect"] = ""
df_path.to_csv(overview_csv_file)
print("Saved to disk as", overview_csv_file)
