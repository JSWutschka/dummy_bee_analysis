"""
plots.py
Script to get an overview of the loopy data quality

Author
   Julia Wutschka

Usage
   Change the values for *directory_input* and *directory_output*, then run it.
"""

import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from time import time
from _socket import gethostname
from termcolor import colored


def filenum(filename):
    return re.search(r"\d+", filename).group(0)

# TODO: CHANGE directory_input
directory_input = r"...\dummy_bee_analysis\input"
directory_output = r"...\dummy_bee_analysis\output"
total_time = time()

directory = os.path.basename(directory_input)
for directories in next(os.walk(directory_input))[1]:
    folder = directories
    for subdirectories in next(os.walk(os.path.join(directory_input, directories)))[1]:
        filecount_DUMMY_ai = []
        wd = os.listdir(os.path.join(directory_input, directories, subdirectories))
        print(colored(f"\nCurrently working on: ", "blue", force_color=True))
        print(os.path.join(directory_input, directories, subdirectories))
        start_time = time()

        for filename_input in wd:
            if filename_input.endswith("_DUMMY_ai.csv"):
                filecount_DUMMY_ai.append(1)
        filecount = len(filecount_DUMMY_ai)
        rows = filecount // 4
        if filecount % 4 != 0:
            rows += 1
        fig, axs = plt.subplots(rows, 4, figsize=(8.27, 11.67), facecolor='w', edgecolor='k')
        subfolder = subdirectories
        fig.suptitle(folder + "_" + subfolder, fontsize=20, x=0.2, y=0.995)
        fig.tight_layout()
        for ax, filename_input in zip(axs.ravel(), wd):
            if filename_input.endswith("_DUMMY_ai.csv"):
                print(filename_input)
                source_dataframe = pd.read_csv(os.path.join(directory_input,directories, subdirectories, filename_input))
                ddf_sorted = source_dataframe[["frame_count", "oid", "x", "y"]].pivot(columns="oid", values=["x", "y"],
                                                                                      index=["frame_count"])
                ddf_sorted.columns = ["".join(str(col)) for col in ddf_sorted.columns.values]
                ddf_sorted = ddf_sorted.rename(
                    columns={"('x', 1)": "F1_x", "('y', 1)": "F1_y", "('x', 2)": "F2_x", "('y', 2)": "F2_y",
                             "('x', 3)": "DH_x", "('y', 3)": "DH_y", "('x', 4)": "D_x", "('y', 4)": "D_y"})
                ddf_sorted = ddf_sorted.reset_index()

                labels = ["F1","F2","DH","D"]
                ddf_sorted.plot.scatter(x="F1_x", y="F1_y", s=1, color="DarkBlue", label="F1", ax=ax)
                ddf_sorted.plot.scatter(x="F2_x", y="F2_y", s=1, color="Blue", label="F2", ax=ax)
                ddf_sorted.plot.scatter(x="DH_x", y="DH_y", s=1, color="Grey", label="DH", ax=ax)
                ddf_sorted.plot.scatter(x="D_x", y="D_y", s=1, color="LightGrey", label="D", ax=ax)
                ax.get_legend().remove()
                ax.set_xlabel("X", fontsize=5)
                ax.set_ylabel("Y", fontsize=5)
                ax.set_title(filenum(filename_input), fontsize=10)
                ax.set_aspect("equal")
                ax.tick_params(axis="x", labelsize=5)
                ax.tick_params(axis="y", labelsize=5)
        handles, labels = ax.get_legend_handles_labels()

        filename_output = os.path.join(directory_output,folder+"_"+subfolder+"_overview.png")
        fig.legend(handles, labels, ncol=4, markerscale=5, loc='upper right')
        fig.savefig(filename_output)
        print(colored(f"\nIt took {gethostname()} {round(time() - start_time, 3)}"
                      f" seconds to create the dummy overview for: ", "light_grey",
                      force_color=True), folder, subfolder)
print(colored(f"\nIt took {gethostname()} {round(time() - total_time, 3)}"
              f" seconds in total. \n", "light_grey",
              force_color=True))