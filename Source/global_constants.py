
"""
global_constants.py
Supporting script containing all global parameters

Author
   Julia Wutschka

Usage
   This script is needed for all other scripts to function.
"""

import pandas as pd


arena_dummy_relation = 14 / 5
# diameter of arena is 14cm, length of feather holder is 5cm

catching_ring_use_default = True
catching_ring_width = 30
feather_smoothing = 0.4
# 1: no smoothing

feather_type = 1

unattended_mode = True
# If True no user interaction is needed
unattended_range_d_level = 0.8
# use the +/- 0.3 (30%) values of point D to the median

class Wrapper_class:
    """
    The Wrapper_class is used to pass dataframes by reference (9 digit number) across the modules
    """
    def __init__(self):
        self.bdf = pd.DataFrame()
        self.ddf_raw = pd.DataFrame()
        self.ddf_sorted = pd.DataFrame()
        self.ddf_finished = pd.DataFrame()
        self.bee_filename = ""
        self.video_filename = ""
        self.analysis_df = pd.DataFrame()
        self.fixed_start_frame = None
        # if is set to None the start frame for point D- approximation will be auto selected to the section
        # where most data will be available
        # else set the start frame manually. if it is 0 the first frame will be the start frame
