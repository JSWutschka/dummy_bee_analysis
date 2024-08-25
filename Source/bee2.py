"""
bee2.py
Script to analyse the traces output by dave and the rebuilt dummy from main.py


Author
   Julia Wutschka

Usage
    PLEASE edit the Config below to your liking!


    This script is a standalone script. You may choose between auto and manual mode.
    *   Experiment length threshold: minimum length of data for analysis
    *   AUTO -> if True:
        *   Show_plots = False ;otherwise your attendance is needed during computing
        *   Create_new_mdf_nice = True
        *   Is_First_run -> if True you will be asked if you want to create a new master file containing the output values
    *   AUTO -> if False:
        *   Please enter your bee file path, dummy file path, json file path below as shown
    Please note that the Threshold parameters will be changed depending on the size of the dummy; changes made previously
    will not be considered!!

"""

import dummy_class
import json
import pandas as pd
import os
from termcolor import colored
from _socket import gethostname
from time import time
import matplotlib.pyplot as plt
import numpy as np
import geolib
import re
import fnmatch
os.system('color')
########################################################################################################################
########################################################################################################################
# Todo: CONFIG
MASTER_DF_EXISTENT = False
MASTER_DF_PATH = r"...\Output\master_df.csv"  # only if MASTER_DF_EXISTS = True
NEW_MASTER_DF_PATH = r"...\Output\master_df.csv"  # only if AUTO False!!
CHECK_CSV = pd.read_csv(r"...\Input\bee_overview_final.csv")  # csv with vid_ident etc

directory_input = r"...\Input\INPUT_DATEN"  # only needed if AUTO True
directory_output = r"...\Output"  # only needed if AUTO True


PATH_BEE =   r"...\Input\C9_23062023\1BEE\165242803_1BEE_nn.csv"
PATH_DUMMY = r"...\Input\C9_23062023\1BEE\165242803_ddf_finished.csv"
JSON =       r"...\Input\C9_23062023\1BEE\165242803.json"
#
EXPERIMENT_LENGTH_THRESH = 8000  # data with lesser frames is not analyzed!
AUTO = True  # automated if true -> need directories
SHOW_PLOTS = False  # not recommended if auto = True
CREATE_NEW_MDF_NICE = True  # recommended for auto = True
IS_FIRST_RUN = True  # needs to be true to create new master df

if AUTO is False:
    directory = os.path.dirname(PATH_BEE)
DIST_DUMMY_THRESHOLD = 60  # COUNTING FOR CONTACT
DIST_FEATHER_THRESHOLD = 70  # COUNTING FOR CONTACT
DIST_BEE_THRESHOLD = 65  # COUNTING FOR CONTACT
DIST_DUMMY_TIME_THRESHOLD = 30  # COUNTING FOR ACTIVE CONTACT
DIST_FEATHER_TIME_THRESHOLD = 60  # COUNTING FOR ACTIVE CONTACT
DIST_BEE_TIME_THRESHOLD = 10  # COUNTING FOR ACTIVE CONTACT

########################################################################################################################
########################################################################################################################




total_time = time()

if AUTO is False:
    bdf_PATH = PATH_BEE
    dbdf_PATH = PATH_DUMMY

if CREATE_NEW_MDF_NICE is True and IS_FIRST_RUN is True:
    print(colored("\nCREATING A NEW MASTER CSV!", "yellow", force_color=True))
    print(colored("WARNING! THIS MAY DELETE THE EXISTING MASTER CSV!", "red", force_color=True))
    INPUT = input(colored("DO YOU WANT TO CONTINUE? (y/n):", "yellow", force_color=True))
    if INPUT == "y" or INPUT == "Y" or INPUT == "yes" or INPUT == "YES":
        master_df = pd.DataFrame(index=[], data={"video_ident": np.nan,
                                                 "date": np.nan,
                                                 "colony": np.nan,
                                                 "Bees": np.nan,
                                                 "Trial": np.nan,
                                                 "IAA": np.nan,
                                                 "TRACKED": np.nan,
                                                 "Stings_recorded": np.nan,
                                                 "sec_near_dummy_b1": np.nan,
                                                 "sec_near_dummy_b2": np.nan,
                                                 "sec_near_feather_b1": np.nan,
                                                 "sec_near_feather_b2": np.nan,
                                                 "sec_near_bee": np.nan,
                                                 "sec_near_dummy_b1_active": np.nan,
                                                 "sec_near_dummy_b2_active": np.nan,
                                                 "sec_near_feather_b1_active": np.nan,
                                                 "sec_near_feather_b2_active": np.nan,
                                                 "sec_near_bee_active": np.nan,
                                                 "times_contacted_dummy_b1": np.nan,
                                                 "times_contacted_dummy_b2": np.nan,
                                                 "times_contacted_feather_b1": np.nan,
                                                 "times_contacted_feather_b2": np.nan,
                                                 "times_contacted_bees": np.nan,
                                                 "accuracy_stings": np.nan,
                                                 "mean_speed_b1_sec": np.nan,
                                                 "percentage_moved_b1": np.nan,
                                                 "mean_speed_b2_sec": np.nan,
                                                 "percentage_moved_b2": np.nan,
                                                 "b1_speed_bf_dc_sec": np.nan,
                                                 "b2_speed_bf_dc_sec": np.nan,
                                                 "b1_speed_bf_fc_sec": np.nan,
                                                 "b2_speed_bf_fc_sec": np.nan,
                                                 "b1_contact_b_first": np.nan,
                                                 "b1_contact_d_first": np.nan,
                                                 "b2_contact_b_first": np.nan,
                                                 "b2_contact_d_first": np.nan,
                                                 })
        master_df.to_csv(NEW_MASTER_DF_PATH)
        IS_FIRST_RUN = False
        MASTER_DF_PATH = NEW_MASTER_DF_PATH


def bees_main(bdf_path, dbdf_path):
    global DIST_DUMMY_THRESHOLD
    global DIST_FEATHER_THRESHOLD
    global DIST_BEE_THRESHOLD
    global DIST_DUMMY_TIME_THRESHOLD
    global DIST_FEATHER_TIME_THRESHOLD
    global DIST_BEE_TIME_THRESHOLD
    global SHOW_PLOTS
    global CREATE_NEW_MDF_NICE
    global IS_FIRST_RUN
    global EXPERIMENT_LENGTH_THRESH
    global MASTER_DF_EXISTENT
    global MASTER_DF_PATH
    global NEW_MASTER_DF_PATH
    global CHECK_CSV
    global JSON


    try:
        with open(JSON, "r") as in_file:
            D_Json = json.load(in_file)
            in_file.close()
    except IOError:
        raise Exception("File does not exist. We will need it for processing...")
    finally:
        pass
    Dummy_len = D_Json["center_radius"] * 2

    DIST_DUMMY_THRESHOLD = Dummy_len * 1/5# COUNTING FOR CONTACT
    DIST_FEATHER_THRESHOLD = Dummy_len * 2/15
    # print(CHECK_CSV)
    # READ DATAFRAME
    # BEE
    bdf = pd.read_csv(bdf_path, index_col=None, usecols=["frame_number", "name", "x", "y"])
    # DUMMY
    # dbdf_path = r"C:\Users\julia\PycharmProjects\dummy_quality_and_bee\input\191041744_ddf_finished.csv"
    dbdf = pd.read_csv(dbdf_path, index_col=None,
                       usecols=["frame_number", "D_x", "D_y", "DH_x", "DH_y", "F1_x", "F1_y", "F2_x", "F2_y"])

    VIDEO_IDENT = re.findall(r"(\d{9})", bdf_path)
    # MASTER_DF_EXISTS = False
    STINGS = 0
    if fnmatch.fnmatch((os.path.basename(bdf_path).split('/')[-1]), "*_1BEE_*"):
        BEES = 1
        if CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]), "frame_sting1"].iloc[0] == 0:
            STING1 = False
        else:
            STING1 = True
            STINGS = 1
            STING1_FRAME = bdf["frame_number"].iloc[0]+(18000-CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]),
            "frame_sting1"].iloc[0])

    elif fnmatch.fnmatch((os.path.basename(bdf_path).split('/')[-1]), "*_2BEE_*"):
        BEES = 2
        if CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]), "frame_sting1"].iloc[0] == 0:
            STING1 = False
        else:
            STING1 = True
            STINGS = 1
            STING1_FRAME = bdf["frame_number"].iloc[0] + CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]),
            "frame_sting1"].iloc[0]
        if CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]), "frame_sting2"].iloc[0] == 0:
            STING2 = False
        else:
            STING2 = True
            STINGS = 1
            STING2_FRAME = bdf["frame_number"].iloc[0] + CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]),
            "frame_sting2"].iloc[0]
        if STING1 is True and STING2 is True:
            STINGS = 2

        print(colored(f"\n There where", "light_blue", force_color=True), STINGS,
              colored(f"stings recorded", "light_blue", force_color=True))

    IAA = 0  # manual: True or False
    # VIDEO_IDENT = 141917542
    HAS_DATE = False
    DATE = ["yyyy/mm/dd"]
    # BEES = 1  # manual: 1 or 2
    TRIAL = CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]), "Test"].iloc[0]
    # TRIAL = 1  # manual: 1, 2, 3 or 4

    if IAA == 0:
        if CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]), "alarmpheromone"].iloc[0] == 0:
            IAA = False
        elif CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]), "alarmpheromone"].iloc[0] == 1:
            IAA = True
        else:
            c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                             "error": ["alarmpheromone"]
                                             })
            master_df = pd.read_csv(MASTER_DF_PATH)
            master_df = pd.concat([master_df, c_master_df], ignore_index=True)
            master_df.to_csv(MASTER_DF_PATH, index=False)
            if AUTO is True:
                return
            else:
                exit("No alarmpheromone information! Please enter manually. You may also want to add the trial manually!")
    COLONY = CHECK_CSV.loc[(CHECK_CSV["Vid_num"] == VIDEO_IDENT[0]), "col"].iloc[0]

    print(colored(f"\n WORKING ON VIDEO:", "yellow", force_color=True),
          colored(VIDEO_IDENT, "green", force_color=True),
          colored(f", TRIAL:", "yellow", force_color=True),
          colored(TRIAL, "green", force_color=True),
          colored(f", HAS ALARMPHEROMONE:", "yellow", force_color=True),
          colored(IAA, "green", force_color=True))



    ########################################################################################################################
    # EMPTY LISTS

    dist_bd1_true_frames = []
    dist_bf1_true_frames = []
    dist_bd2_true_frames = []
    dist_bf2_true_frames = []
    B1_SL = []
    B2_SL = []
    dist_bb_true_frames = []
    ########################################################################################################################

    ########################################################################################################################
    ########################################################################################################################
    # 2 BEES
    ########################################################################################################################
    ########################################################################################################################
    if BEES == 2:
        if len(bdf.name.unique()) > 2 or len(bdf.name.unique()) < 2:
            c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                             "error": ["traces"]
                                             })
            master_df = pd.read_csv(MASTER_DF_PATH)
            master_df = pd.concat([master_df, c_master_df], ignore_index=True)
            master_df.to_csv(MASTER_DF_PATH, index=False)
            if AUTO is True:
                return
            else:
                exit("Too many or too few traces left in .csv! Analysis is not possible! Make sure to select the "
                 "correct amount of bees!")
        else:
            start_time = time()

            ################################################################################################################
            # CALCULATE TIME BEE WAS NEAR DUMMY / FEATHER
            ################################################################################################################

            bdf = bdf[["frame_number", "name", "x", "y"]].pivot(columns="name", values=["x", "y"],
                                                                index=["frame_number"])
            bdf.columns = ['_'.join(col) for col in bdf.columns.values]
            # print(bdf.columns)
            bdf.columns = ["B1_x", "B2_x", "B1_y", "B2_y"]
            bdf = bdf.reset_index()

            # print(bdf)

            ax = bdf.plot.scatter(x="B1_x", y="B1_y", color="DarkBlue", label="Bee 1", title="VISUALIZED RAW-DATA")
            bdf.plot.scatter(x="B2_x", y="B2_y", color="DarkRed", label="Bee 2", ax=ax)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect('equal')
            if SHOW_PLOTS is True:
                plt.show()

            dummybee_df = pd.merge(dbdf, bdf, how="right", on=["frame_number"])
            dummybee_df = dummybee_df.reset_index()
            dummybee_df = dummybee_df[(dummybee_df.B1_x != -1)]
            dummybee_df = dummybee_df.dropna().reset_index(drop=True)
            """
            if len(dummybee_df.index) < EXPERIMENT_LENGTH_THRESH:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "error": ["length"]
                                                 })
                master_df = pd.read_csv(MASTER_DF_PATH)
                master_df = pd.concat([master_df, c_master_df], ignore_index=True)
                master_df.to_csv(MASTER_DF_PATH, index=False)
                if AUTO is True:
                    return
                else:
                    exit("Too little of the experiment was tracked")
            """
            # print(dummybee_df)
            for index, row in dummybee_df.iterrows():
                XP4 = row["D_x"]
                YP4 = row["D_y"]
                XP3 = row["DH_x"]
                YP3 = row["DH_y"]
                XP2 = row["F1_x"]
                YP2 = row["F1_y"]
                XP1 = row["F2_x"]
                YP1 = row["F2_y"]
                x1 = row["B1_x"]
                y1 = row["B1_y"]
                x2 = row["B2_x"]
                y2 = row["B2_y"]

                dummy = dummy_class.Dummy(geolib.Straight(geolib.Vector(XP3, YP3),
                                              geolib.Vector(XP4, YP4)), 0.2)
                feather = dummy_class.Feather(geolib.Vector(XP3, YP3), geolib.Vector(XP2, YP2), geolib.Vector(XP1, YP1))
                Bee_1 = geolib.Vector(x1, y1)
                Bee_2 = geolib.Vector(x2, y2)

                dist_bd1 = dummy.get_distance(Bee_1)
                dist_bf1 = feather.get_distance(Bee_1)
                dist_bd2 = dummy.get_distance(Bee_2)
                dist_bf2 = feather.get_distance(Bee_2)
                if dist_bd1 <= DIST_DUMMY_THRESHOLD:
                    dist_bd1_true_frames.append(1)
                    dummybee_df.at[index, "dist_BD1"] = 1
                else:
                    dist_bd1_true_frames.append(0)
                    dummybee_df.at[index, "dist_BD1"] = 0

                if dist_bf1 <= DIST_FEATHER_THRESHOLD:
                    dist_bf1_true_frames.append(1)
                    dummybee_df.at[index, "dist_BF1"] = 1
                else:
                    dist_bf1_true_frames.append(0)
                    dummybee_df.at[index, "dist_BF1"] = 0

                if dist_bd2 <= DIST_DUMMY_THRESHOLD:
                    dist_bd2_true_frames.append(1)
                    dummybee_df.at[index, "dist_BD2"] = 1
                else:
                    dist_bd2_true_frames.append(0)
                    dummybee_df.at[index, "dist_BD2"] = 0
                if dist_bf2 <= DIST_FEATHER_THRESHOLD:
                    dist_bf2_true_frames.append(1)
                    dummybee_df.at[index, "dist_BF2"] = 1
                else:
                    dist_bf2_true_frames.append(0)
                    dummybee_df.at[index, "dist_BF2"] = 0

            seconds_near_dummy_1 = sum(dist_bd1_true_frames) / 100
            seconds_near_dummy_2 = sum(dist_bd2_true_frames) / 100
            seconds_near_feather_1 = sum(dist_bf1_true_frames) / 100
            seconds_near_feather_2 = sum(dist_bf2_true_frames) / 100

            print(colored(f"\n The first bee was near the dummy for:", "light_blue", force_color=True))
            print(sum(dist_bd1_true_frames), "frames")
            print(seconds_near_dummy_1, "seconds")
            print(colored(f"\n The first bee was near the feather for:", "light_blue", force_color=True))
            print(sum(dist_bf1_true_frames), "frames")
            print(seconds_near_feather_1, "seconds")
            print(colored(f"\n The second bee was near the dummy for:", "light_blue", force_color=True))
            print(sum(dist_bd2_true_frames), "frames")
            print(seconds_near_dummy_2, "seconds")
            print(colored(f"\n The second bee was near the feather for:", "light_blue", force_color=True))
            print(sum(dist_bf2_true_frames), "frames")
            print(seconds_near_feather_2, "seconds")

            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to calculate the proximity of the bee(s) to the dummy. \n", "light_grey",
                          force_color=True))

            ################################################################################################################
            # CALCULATE TIME BEE WAS NEAR ANOTHER BEE
            ################################################################################################################
            start_time = time()

            for index, row in dummybee_df.iterrows():
                x1 = row["B1_x"]
                y1 = row["B1_y"]
                x2 = row["B2_x"]
                y2 = row["B2_y"]

                B1 = geolib.Vector(x1, y1)
                B2 = geolib.Vector(x2, y2)

                if B1.sub(B2).abs() <= DIST_BEE_THRESHOLD:
                    dist_bb_true_frames.append(1)
                    dummybee_df.at[index, "dist_BB"] = 1

                else:
                    dist_bb_true_frames.append(0)
                    dummybee_df.at[index, "dist_BB"] = 0

            seconds_near_bee = sum(dist_bb_true_frames) / 100

            print(colored(f"The bees were interacting with each other for:", "light_blue", force_color=True))
            print(sum(dist_bb_true_frames), "frames")
            print(seconds_near_bee, "seconds")

            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to calculate the proximity of the bees. \n", "light_grey",
                          force_color=True))

            start_time = time()
            dummybee_df["dist_BD1_active"] = np.nan
            dummybee_df["dist_BD2_active"] = np.nan
            dummybee_df["dist_BF1_active"] = np.nan
            dummybee_df["dist_BF2_active"] = np.nan
            dummybee_df["dist_BB_active"] = np.nan

            for index in dummybee_df.index:
                ############################################################################################################
                # CHECK IF ACTIVE OR PASSIVE
                ############################################################################################################
                if index + DIST_DUMMY_TIME_THRESHOLD in dummybee_df.index:
                    if (dummybee_df["dist_BD1"].iloc[index] == 1 and
                            dummybee_df["dist_BD1"].iloc[index + DIST_DUMMY_TIME_THRESHOLD] == 1):
                        dummybee_df.at[index, "dist_BD1_active"] = 1
                    else:
                        dummybee_df.at[index, "dist_BD1_active"] = 0
                if dummybee_df["dist_BD1"].iloc[index] == 1 and dummybee_df["dist_BD1_active"].iloc[index] == 1:
                    dummybee_df.at[index, "dist_BD1_active"] = 1
                else:
                    dummybee_df.at[index, "dist_BD1_active"] = 0
                ############################################################################################################

                if index + DIST_DUMMY_TIME_THRESHOLD in dummybee_df.index:
                    if (dummybee_df["dist_BD2"].iloc[index] == 1 and
                            dummybee_df["dist_BD2"].iloc[index + DIST_DUMMY_TIME_THRESHOLD] == 1):
                        dummybee_df.at[index, "dist_BD2_active"] = 1
                    else:
                        dummybee_df.at[index, "dist_BD2_active"] = 0

                if dummybee_df["dist_BD2"].iloc[index] == 1 and dummybee_df["dist_BD2_active"].iloc[index] == 1:
                    dummybee_df.at[index, "dist_BD2_active"] = 1
                else:
                    dummybee_df.at[index, "dist_BD2_active"] = 0
                ############################################################################################################

                if index + DIST_FEATHER_TIME_THRESHOLD in dummybee_df.index:
                    if dummybee_df["dist_BF1"].iloc[index] == 1 and \
                            dummybee_df["dist_BF1"].iloc[index + DIST_FEATHER_TIME_THRESHOLD] == 1:
                        dummybee_df.at[index, "dist_BF1_active"] = 1
                    else:
                        dummybee_df.at[index, "dist_BF1_active"] = 0
                if dummybee_df["dist_BF1"].iloc[index] == 1 and dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    dummybee_df.at[index, "dist_BF1_active"] = 1
                else:
                    dummybee_df.at[index, "dist_BF1_active"] = 0
                ############################################################################################################

                if index + DIST_FEATHER_TIME_THRESHOLD in dummybee_df.index:
                    if dummybee_df["dist_BF2"].iloc[index] == 1 and \
                            dummybee_df["dist_BF2"].iloc[index + DIST_FEATHER_TIME_THRESHOLD] == 1:
                        dummybee_df.at[index, "dist_BF2_active"] = 1
                    else:
                        dummybee_df.at[index, "dist_BF2_active"] = 0
                if dummybee_df["dist_BF2"].iloc[index] == 1 and dummybee_df["dist_BF2_active"].iloc[index] == 1:
                    dummybee_df.at[index, "dist_BF2_active"] = 1
                else:
                    dummybee_df.at[index, "dist_BF2_active"] = 0
                ############################################################################################################

                if index + DIST_BEE_TIME_THRESHOLD in dummybee_df.index:
                    if dummybee_df["dist_BB"].iloc[index] == 1 and \
                            dummybee_df["dist_BB"].iloc[index + DIST_BEE_TIME_THRESHOLD] == 1:
                        dummybee_df.at[index, "dist_BB_active"] = 1
                    else:
                        dummybee_df.at[index, "dist_BB_active"] = 0
                if dummybee_df["dist_BB"].iloc[index] == 1 and dummybee_df["dist_BB_active"].iloc[index] == 1:
                    dummybee_df.at[index, "dist_BB_active"] = 1
                else:
                    dummybee_df.at[index, "dist_BB_active"] = 0

                ############################################################################################################
                # CALCULATE SPEED OF THE BEE / ACTIVE TIME
                ############################################################################################################
                if index + 1 in dummybee_df.index:
                    B1_x1 = dummybee_df["B1_x"].iloc[index]
                    B1_x2 = dummybee_df["B1_x"].iloc[index + 1]
                    B1_y1 = dummybee_df["B1_y"].iloc[index]
                    B1_y2 = dummybee_df["B1_y"].iloc[index + 1]

                    B2_x1 = dummybee_df["B2_x"].iloc[index]
                    B2_x2 = dummybee_df["B2_x"].iloc[index + 1]
                    B2_y1 = dummybee_df["B2_y"].iloc[index]
                    B2_y2 = dummybee_df["B2_y"].iloc[index + 1]

                    B1_1 = geolib.Vector(B1_x1, B1_y1)
                    B1_2 = geolib.Vector(B1_x2, B1_y2)

                    B2_1 = geolib.Vector(B2_x1, B2_y1)
                    B2_2 = geolib.Vector(B2_x2, B2_y2)

                    B1_s = B1_1.sub(B1_2).abs()
                    B2_s = B2_1.sub(B2_2).abs()

                    B1_SL.append(B1_s)
                    B2_SL.append(B2_s)
                    dummybee_df.at[index, "speed_B1"] = B1_s
                    dummybee_df.at[index, "speed_B2"] = B2_s
                else:
                    pass

            seconds_near_dummy_1_active = dummybee_df["dist_BD1_active"].sum() / 100
            seconds_near_dummy_2_active = dummybee_df["dist_BD2_active"].sum() / 100
            seconds_near_feather_1_active = dummybee_df["dist_BF1_active"].sum() / 100
            seconds_near_feather_2_active = dummybee_df["dist_BF2_active"].sum() / 100
            seconds_near_bee_active = dummybee_df["dist_BB_active"].sum() / 100

            print(colored(f"\n The first bee was actively near the dummy for:", "light_blue", force_color=True))
            print(dummybee_df["dist_BD1_active"].sum(), "frames")
            print(seconds_near_dummy_1_active, "seconds")
            print(colored(f"\n The first bee was actively near the feather for:", "light_blue", force_color=True))
            print(dummybee_df["dist_BF1_active"].sum(), "frames")
            print(seconds_near_feather_1_active, "seconds")
            print(colored(f"\n The second bee was actively near the dummy for:", "light_blue", force_color=True))
            print(dummybee_df["dist_BD2_active"].sum(), "frames")
            print(seconds_near_dummy_2_active, "seconds")
            print(colored(f"\n The second bee was actively near the feather for:", "light_blue", force_color=True))
            print(dummybee_df["dist_BF2_active"].sum(), "frames")
            print(seconds_near_feather_2_active, "seconds")

            B1_SL_only_movement = list(filter(lambda num: num != 0, B1_SL))
            mean_speed_1 = sum(B1_SL_only_movement) / len(B1_SL_only_movement)
            mean_speed_1_sec = mean_speed_1 / 100
            percentage_speed_1 = len(B1_SL_only_movement) / len(B1_SL)

            print(colored(f"\n The first bee was moving at a speed of:", "light_blue", force_color=True))
            print(mean_speed_1, "px/frame")
            print(mean_speed_1 / 100, "px/s")
            print(colored(f"\n and was moving in", "light_blue", force_color=True))
            print(percentage_speed_1 * 100, "%")
            print(colored(f" of the video.", "light_blue", force_color=True))

            B2_SL_only_movement = list(filter(lambda num: num != 0, B2_SL))
            mean_speed_2 = sum(B2_SL_only_movement) / len(B2_SL_only_movement)
            mean_speed_2_sec = mean_speed_2 / 100
            percentage_speed_2 = len(B2_SL_only_movement) / len(B2_SL)

            print(colored(f"\n The second bee was moving at a speed of:", "light_blue", force_color=True))
            print(mean_speed_2, "px/frame")
            print(mean_speed_2 / 100, "px/s")
            print(colored(f"\n and was moving in", "light_blue", force_color=True))
            print(percentage_speed_2 * 100, "%")
            print(colored(f" of the video.", "light_blue", force_color=True))

            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to calculate the speed of the bee(s) throughout "
                          f"the video and the active contact time. \n",
                          "light_grey",
                          force_color=True))

            for index in dummybee_df.index:
                if dummybee_df["dist_BD1_active"].iloc[index] == 1 and dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    if index + 10 in dummybee_df.index:
                        if (dummybee_df["dist_BF1_active"].iloc[index + 5] == 1 or
                                dummybee_df["dist_BF1_active"].iloc[index + 10] == 1):
                            dummybee_df.at[index, "dist_BD1_active"] = 0
                            dummybee_df.at[index, "dist_BF1_active"] = 1
                        else:
                            dummybee_df.at[index, "dist_BF1_active"] = 0
                    else:
                        dummybee_df.at[index, "dist_BF1_active"] = 0 #todo

                if dummybee_df["dist_BD2_active"].iloc[index] == 1 and dummybee_df["dist_BF2_active"].iloc[index] == 1:
                    if index + 10 in dummybee_df.index:
                        if (dummybee_df["dist_BF2_active"].iloc[index + 5] == 1 or
                                dummybee_df["dist_BF2_active"].iloc[index + 10] == 1):
                            dummybee_df.at[index, "dist_BD2_active"] = 0
                            dummybee_df.at[index, "dist_BF2_active"] = 1
                        else:
                            dummybee_df.at[index, "dist_BF2_active"] = 0
                    else:
                        dummybee_df.at[index, "dist_BF2_active"] = 0

                if dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BD1_active"].iloc[index + 1] == 1 or
                                dummybee_df["dist_BD1_active"].iloc[index + 5] == 1):
                            dummybee_df.at[index, "dist_BD1_active"] = 1
                            dummybee_df.at[index, "dist_BF1_active"] = 0

                if dummybee_df["dist_BF2_active"].iloc[index] == 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BD2_active"].iloc[index + 1] == 1 or
                                dummybee_df["dist_BD2_active"].iloc[index + 5] == 1):
                            dummybee_df.at[index, "dist_BD2_active"] = 1
                            dummybee_df.at[index, "dist_BF2_active"] = 0

            for index in dummybee_df.index:
                if dummybee_df["dist_BD1_active"].iloc[index] == 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BF1_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 3] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 5] == 1):
                            dummybee_df.at[index, "dist_BF1_active"] = 1
                            dummybee_df.at[index, "dist_BD1_active"] = 0
                if dummybee_df["dist_BD2_active"].iloc[index] == 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BF2_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BF2_active"].iloc[index + 3] == 1 and
                                dummybee_df["dist_BF2_active"].iloc[index + 5] == 1):
                            dummybee_df.at[index, "dist_BF2_active"] = 1
                            dummybee_df.at[index, "dist_BD2_active"] = 0
                if dummybee_df["dist_BB_active"].iloc[index] == 1:
                    if index + 3 in dummybee_df.index:
                        if dummybee_df["dist_BB_active"].iloc[index + 3] == 1:
                            dummybee_df.at[index + 1, "dist_BB_active"] = 1
                            dummybee_df.at[index + 2, "dist_BB_active"] = 1

            ################################################################################################################
            # SPEED BEFORE CONTACT AND WHAT DID BEE CONTACT FIRST VARIABLES
            ################################################################################################################
            start_time = time()

            n = 0  # todo
            nf = 0
            n2 = 0
            n2f = 0
            nx = 0  # todo
            nf = 0
            n2x = 0
            n2xf = 0
            na = 0
            naf = 0
            n3 = 0
            n3f = 0
            BD1a_group = []  # todo
            BF1a_group = []
            BD2a_group = []  # todo
            BF2a_group = []
            BBa_group = []

            dummybee_df["BD1a_group"] = np.nan
            dummybee_df["BF1a_group"] = np.nan
            dummybee_df["BD2a_group"] = np.nan
            dummybee_df["BF2a_group"] = np.nan
            dummybee_df["BBa_group"] = np.nan
            dummybee_df["speed_b4_BD1"] = np.nan
            dummybee_df["speed_b4_BD2"] = np.nan
            dummybee_df["speed_b4_BF1"] = np.nan
            dummybee_df["speed_b4_BF2"] = np.nan
            dummybee_df["speed_b4_BB1"] = np.nan
            dummybee_df["speed_b4_BB2"] = np.nan

            contact_bee_first1 = 0
            contact_dummy_first1 = 0
            contact_bee_first2 = 0
            contact_dummy_first2 = 0
            for index in dummybee_df.index:
                if dummybee_df["dist_BD1_active"].iloc[index] == 1:
                    if dummybee_df["dist_BD1_active"].iloc[index - 1] == 0:
                        for i in range(index - 20, index):
                            dummybee_df.at[i, "speed_b4_BD1"] = dummybee_df["speed_B1"].iloc[i]
                if dummybee_df["dist_BD2_active"].iloc[index] == 1:
                    if dummybee_df["dist_BD2_active"].iloc[index - 1] == 0:
                        for i in range(index - 20, index):
                            dummybee_df.at[i, "speed_b4_BD2"] = dummybee_df["speed_B2"].iloc[i]

                if dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    if dummybee_df["dist_BF1_active"].iloc[index - 1] == 0:
                        for i in range(index - 20, index):
                            dummybee_df.at[i, "speed_b4_BF1"] = dummybee_df["speed_B1"].iloc[i]
                if dummybee_df["dist_BF2_active"].iloc[index] == 1:
                    if dummybee_df["dist_BF2_active"].iloc[index - 1] == 0:
                        for i in range(index - 20, index):
                            dummybee_df.at[i, "speed_b4_BF2"] = dummybee_df["speed_B2"].iloc[i]

                if dummybee_df["dist_BB_active"].iloc[index] == 1:
                    if dummybee_df["dist_BB_active"].iloc[index - 1] == 0:
                        for i in range(index - 20, index):
                            dummybee_df.at[i, "speed_b4_BB1"] = dummybee_df["speed_B1"].iloc[i]
                            dummybee_df.at[i, "speed_b4_BB2"] = dummybee_df["speed_B2"].iloc[i]
                ############################################################################################################
                # WHAT DID BEE CONTACT FIRST
                ############################################################################################################

                if contact_dummy_first1 == 0 and contact_bee_first1 == 0:
                    if dummybee_df["dist_BD1_active"].iloc[index] == 1:
                        contact_dummy_first1 = 1
                    if dummybee_df["dist_BF1_active"].iloc[index] == 1:
                        contact_dummy_first1 = 1
                if contact_bee_first1 == 0 and contact_dummy_first1 == 0:
                    if dummybee_df["dist_BB_active"].iloc[index] == 1:
                        contact_bee_first1 = 1
                if contact_dummy_first2 == 0 and contact_bee_first2 == 0:
                    if dummybee_df["dist_BD2_active"].iloc[index] == 1:
                        contact_dummy_first2 = 1
                    if dummybee_df["dist_BF2_active"].iloc[index] == 1:
                        contact_dummy_first2 = 1
                if contact_bee_first2 == 0 and contact_dummy_first2 == 0:
                    if dummybee_df["dist_BB_active"].iloc[index] == 1:
                        contact_bee_first2 = 1

                ############################################################################################################
                # CONTACT GROUPS #todo
                ############################################################################################################
                na += 1  # todo
                if na > 9:
                    n = 2
                    n2 = 2
                    nx = 2
                    n2x = 2
                    n3 = 2
                if n > 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BD1_active"].iloc[index] == 1 and
                                dummybee_df["dist_BD1_active"].iloc[index - 1] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 2] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 3] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 4] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 5] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 6] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 7] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 8] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 9] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 10] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BD1_active"].iloc[index + 2] == 1 and
                                dummybee_df["dist_BD1_active"].iloc[index + 3] == 1):
                            dummybee_df.at[index, "BD1a_group"] = dummybee_df["frame_number"].iloc[index]
                            BD1a_group.append(dummybee_df["frame_number"].iloc[index])

                if n2 > 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BF1_active"].iloc[index] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index - 1] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 2] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 3] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 4] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 5] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 6] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 7] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 8] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 9] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 10] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 2] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 3] == 1):
                            dummybee_df.at[index, "BF1a_group"] = dummybee_df["frame_number"].iloc[index]
                            BF1a_group.append(dummybee_df["frame_number"].iloc[index])

                if dummybee_df["dist_BD1_active"].iloc[index] == 1:
                    dummybee_df.at[index, "BD1a_group"] = dummybee_df["frame_number"].iloc[index]
                if dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    dummybee_df.at[index, "BF1a_group"] = dummybee_df["frame_number"].iloc[index]

                if nx > 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BD2_active"].iloc[index] == 1 and
                                dummybee_df["dist_BD2_active"].iloc[index - 1] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 2] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 3] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 4] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 5] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 6] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 7] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 8] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 9] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index - 10] == 0 and
                                dummybee_df["dist_BD2_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BD2_active"].iloc[index + 2] == 1 and
                                dummybee_df["dist_BD2_active"].iloc[index + 3] == 1):
                            dummybee_df.at[index, "BD2a_group"] = dummybee_df["frame_number"].iloc[index]
                            BD2a_group.append(dummybee_df["frame_number"].iloc[index])

                if n2 > 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BF2_active"].iloc[index] == 1 and
                                dummybee_df["dist_BF2_active"].iloc[index - 1] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 2] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 3] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 4] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 5] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 6] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 7] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 8] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 9] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index - 10] == 0 and
                                dummybee_df["dist_BF2_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BF2_active"].iloc[index + 2] == 1 and
                                dummybee_df["dist_BF2_active"].iloc[index + 3] == 1):
                            dummybee_df.at[index, "BF2a_group"] = dummybee_df["frame_number"].iloc[index]
                            BF2a_group.append(dummybee_df["frame_number"].iloc[index])

                if dummybee_df["dist_BD2_active"].iloc[index] == 1:
                    dummybee_df.at[index, "BD2a_group"] = dummybee_df["frame_number"].iloc[index]
                if dummybee_df["dist_BF2_active"].iloc[index] == 1:
                    dummybee_df.at[index, "BF2a_group"] = dummybee_df["frame_number"].iloc[index]
                if n3 > 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BB_active"].iloc[index] == 1 and
                                dummybee_df["dist_BB_active"].iloc[index - 1] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 2] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 3] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 4] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 5] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 6] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 7] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 8] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 9] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index - 10] == 0 and
                                dummybee_df["dist_BB_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BB_active"].iloc[index + 2] == 1 and
                                dummybee_df["dist_BB_active"].iloc[index + 3] == 1):
                            dummybee_df.at[index, "BBa_group"] = dummybee_df["frame_number"].iloc[index]
                            BBa_group.append(dummybee_df["frame_number"].iloc[index])

                if dummybee_df["dist_BB_active"].iloc[index] == 1:
                    dummybee_df.at[index, "BBa_group"] = dummybee_df["frame_number"].iloc[index]

            if (contact_bee_first1 + contact_bee_first2 + contact_dummy_first1 + contact_dummy_first2) > 2 or \
                    (contact_dummy_first1 + contact_dummy_first1) > 2 or \
                    (contact_dummy_first2 + contact_bee_first2) > 2:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "error": ["contact"]
                                                 })
                master_df = pd.read_csv(MASTER_DF_PATH)
                master_df = pd.concat([master_df, c_master_df], ignore_index=True)
                master_df.to_csv(MASTER_DF_PATH, index=False)
                if AUTO is True:
                    return
                else:
                    exit("One bee can't contact both bee and dummy first at the same time!")
            if contact_dummy_first1 == 1:
                print(colored(f"\nThe first bee contacted the dummy first!", "light_blue", force_color=True))
            if contact_dummy_first2 == 1:
                print(colored(f"\nThe second bee contacted the dummy first!", "light_blue", force_color=True))
            if contact_bee_first1 == 1:
                print(colored(f"The first bee contacted the bee first!", "light_blue", force_color=True))
            if contact_bee_first2 == 1:
                print(colored(f"The second bee contacted the bee first!", "light_blue", force_color=True))

            TIMES_CONTACTED_FEATHER = len(BF1a_group)
            print(colored(f"\nThe first bee contacted the feather", "light_blue", force_color=True),
                  TIMES_CONTACTED_FEATHER, colored(f"times.", "light_blue", force_color=True))
            TIMES_CONTACTED_FEATHER2 = len(BF2a_group)
            print(colored(f"\nThe second bee contacted the feather", "light_blue", force_color=True),
                  TIMES_CONTACTED_FEATHER2, colored(f"times.", "light_blue", force_color=True))
            TIMES_CONTACTED_BEES = len(BBa_group)
            print(colored(f"\nThe bees contacted each other", "light_blue", force_color=True),
                  TIMES_CONTACTED_BEES, colored(f"times.", "light_blue", force_color=True))
            ACCURACY = -1
            if STING1 is True:
                if ((STING1_FRAME or STING1_FRAME + 1 or STING1_FRAME - 1 or STING1_FRAME + 2 or STING1_FRAME - 2 ) in
                        dummybee_df["BF1a_group"].values or STING1_FRAME in dummybee_df["BF2a_group"].values):
                    ACCURACY = 0.5
                else:
                    ACCURACY = 0
            if STING2 is True:
                if ((STING2_FRAME or STING2_FRAME + 1 or STING2_FRAME - 1 or STING2_FRAME + 2 or STING2_FRAME - 2 ) in
                        dummybee_df["BF1a_group"].values or STING2_FRAME in dummybee_df["BF2a_group"].values):
                    ACCURACY += 0.5
                else:
                    ACCURACY += 0

            TIMES_CONTACTED_DUMMY = len(BD1a_group)
            print(colored(f"\nThe first bee contacted the dummy", "light_blue", force_color=True),
                  TIMES_CONTACTED_DUMMY, colored(f"times.", "light_blue", force_color=True))
            TIMES_CONTACTED_DUMMY2 = len(BD1a_group)
            print(colored(f"\nThe second bee contacted the dummy", "light_blue", force_color=True),
                  TIMES_CONTACTED_DUMMY2, colored(f"times.", "light_blue", force_color=True))
            if STING1 is True:
                if ((STING1_FRAME or STING1_FRAME + 1 or STING1_FRAME - 1 or STING1_FRAME + 2 or STING1_FRAME - 2 ) in
                        dummybee_df["BD1a_group"].values or STING1_FRAME in dummybee_df["BD2a_group"].values):
                    ACCURACY += 1
                else:
                    ACCURACY += 0
            if STING2 is True:
                if ((STING2_FRAME or STING2_FRAME + 1 or STING2_FRAME - 1 or STING2_FRAME + 2 or STING2_FRAME - 2 ) in
                        dummybee_df["BD1a_group"].values or STING2_FRAME in dummybee_df["BD2a_group"].values):
                    ACCURACY += 1
                else:
                    ACCURACY += 0

            if ACCURACY == 0:
                print(colored(f"\nThe recorded stinging time is not within the calculated contact time!", "light_blue",
                              force_color=True),
                      colored(f"\nAccuracy-level:", "light_blue", force_color=True), ACCURACY)
            elif ACCURACY == -1:
                print(colored(f"\nNo sting recorded!", "light_blue",
                              force_color=True),
                      colored(f"\nNo accuracy-level!", "light_blue", force_color=True))
            else:

                print(colored(f"\nThe recorded stinging times are within the calculated contact time!", "light_blue",
                              force_color=True),
                      colored(f"\nAccuracy-level:", "light_blue", force_color=True), ACCURACY)

            speed_b4_BD1 = dummybee_df[dummybee_df["speed_b4_BD1"] != 0]["speed_b4_BD1"].mean()
            speed_b4_BD2 = dummybee_df[dummybee_df["speed_b4_BD2"] != 0]["speed_b4_BD2"].mean()
            speed_b4_BF1 = dummybee_df[dummybee_df["speed_b4_BF1"] != 0]["speed_b4_BF1"].mean()
            speed_b4_BF2 = dummybee_df[dummybee_df["speed_b4_BF2"] != 0]["speed_b4_BF2"].mean()
            speed_b4_BD1_sec = speed_b4_BD1 / 100
            speed_b4_BD2_sec = speed_b4_BD2 / 100
            speed_b4_BF1_sec = speed_b4_BF1 / 100
            speed_b4_BF2_sec = speed_b4_BF2 / 100

            print(colored(f"\n20 Frames before the first bee was in contact with the dummy (each time) "
                          f"it was moving at ca.:",
                          "light_blue", force_color=True))
            print(speed_b4_BD1, "px/frame")
            print(colored(f"20 Frames before the second bee was in contact with the dummy (each time) "
                          f"it was moving at ca.:",
                          "light_blue", force_color=True))
            print(speed_b4_BD2, "px/frame")

            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to calculate the speed of the bee(s) near the dummy and to calculate "
                          f"what the bees interacted with first. \n",
                          "light_grey",
                          force_color=True))

            # dummybee_df.to_csv(r"C:\Users\julia\PycharmProjects\dummy_quality_and_bee\output\testtestest6.csv")

            start_time = time()
            percentage_speed_1 = percentage_speed_1 * 100
            percentage_speed_2 = percentage_speed_2 * 100
            if HAS_DATE is True:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "date": DATE,
                                                 "colony": COLONY,
                                                 "Bees": [2],
                                                 "Trial": TRIAL,
                                                 "IAA": IAA,
                                                 "TRACKED": len(dummybee_df.index),
                                                 "Stings_recorded": STINGS,
                                                 "sec_near_dummy_b1": [seconds_near_dummy_1],
                                                 "sec_near_dummy_b2": [seconds_near_dummy_2],
                                                 "sec_near_feather_b1": [seconds_near_feather_1],
                                                 "sec_near_feather_b2": [seconds_near_feather_2],
                                                 "sec_near_bee": [seconds_near_bee],
                                                 "sec_near_dummy_b1_active": [seconds_near_dummy_1_active],
                                                 "sec_near_dummy_b2_active": [seconds_near_dummy_2_active],
                                                 "sec_near_feather_b1_active": [seconds_near_feather_1_active],
                                                 "sec_near_feather_b2_active": [seconds_near_feather_2_active],
                                                 "sec_near_bee_active": [seconds_near_bee_active],
                                                 "times_contacted_dummy_b1": [TIMES_CONTACTED_DUMMY],
                                                 "times_contacted_dummy_b2": [TIMES_CONTACTED_DUMMY2],
                                                 "times_contacted_feather_b1": [TIMES_CONTACTED_FEATHER],
                                                 "times_contacted_feather_b2": [TIMES_CONTACTED_FEATHER2],
                                                 "times_contacted_bees": TIMES_CONTACTED_BEES,
                                                 "accuracy_stings": [ACCURACY],
                                                 "mean_speed_b1_sec": [mean_speed_1_sec],
                                                 "percentage_moved_b1": [percentage_speed_1],
                                                 "mean_speed_b2_sec": [mean_speed_2_sec],
                                                 "percentage_moved_b2": [percentage_speed_2],
                                                 "b1_speed_bf_dc_sec": [speed_b4_BD1_sec],
                                                 "b2_speed_bf_dc_sec": [speed_b4_BD2_sec],
                                                 "b1_speed_bf_fc_sec": [speed_b4_BF1_sec],
                                                 "b2_speed_bf_fc_sec": [speed_b4_BF2_sec],
                                                 "b1_contact_b_first": [contact_bee_first1],
                                                 "b1_contact_d_first": [contact_dummy_first1],
                                                 "b2_contact_b_first": [contact_bee_first2],
                                                 "b2_contact_d_first": [contact_dummy_first2],
                                                 })
            else:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "colony": COLONY,
                                                 "Bees": [2],
                                                 "Trial": TRIAL,
                                                 "IAA": IAA,
                                                 "TRACKED": len(dummybee_df.index),
                                                 "Stings_recorded": STINGS,
                                                 "sec_near_dummy_b1": [seconds_near_dummy_1],
                                                 "sec_near_dummy_b2": [seconds_near_dummy_2],
                                                 "sec_near_feather_b1": [seconds_near_feather_1],
                                                 "sec_near_feather_b2": [seconds_near_feather_2],
                                                 "sec_near_bee": [seconds_near_bee],
                                                 "sec_near_dummy_b1_active": [seconds_near_dummy_1_active],
                                                 "sec_near_dummy_b2_active": [seconds_near_dummy_2_active],
                                                 "sec_near_feather_b1_active": [seconds_near_feather_1_active],
                                                 "sec_near_feather_b2_active": [seconds_near_feather_2_active],
                                                 "sec_near_bee_active": [seconds_near_bee_active],
                                                 "times_contacted_dummy_b1": [TIMES_CONTACTED_DUMMY],
                                                 "times_contacted_dummy_b2": [TIMES_CONTACTED_DUMMY2],
                                                 "times_contacted_feather_b1": [TIMES_CONTACTED_FEATHER],
                                                 "times_contacted_feather_b2": [TIMES_CONTACTED_FEATHER2],
                                                 "times_contacted_bees": TIMES_CONTACTED_BEES,
                                                 "accuracy_stings": [ACCURACY],
                                                 "mean_speed_b1_sec": [mean_speed_1_sec],
                                                 "percentage_moved_b1": [percentage_speed_1],
                                                 "mean_speed_b2_sec": [mean_speed_2_sec],
                                                 "percentage_moved_b2": [percentage_speed_2],
                                                 "b1_speed_bf_dc_sec": [speed_b4_BD1_sec],
                                                 "b2_speed_bf_dc_sec": [speed_b4_BD2_sec],
                                                 "b1_speed_bf_fc_sec": [speed_b4_BF1_sec],
                                                 "b2_speed_bf_fc_sec": [speed_b4_BF2_sec],
                                                 "b1_contact_b_first": [contact_bee_first1],
                                                 "b1_contact_d_first": [contact_dummy_first1],
                                                 "b2_contact_b_first": [contact_bee_first2],
                                                 "b2_contact_d_first": [contact_dummy_first2],
                                                 })
            print(c_master_df)
            if MASTER_DF_EXISTENT is True:
                master_df = pd.read_csv(MASTER_DF_PATH)
                master_df = pd.concat([master_df, c_master_df], ignore_index=True)
                master_df.to_csv(MASTER_DF_PATH, index=False)
            if MASTER_DF_EXISTENT is False:
                c_master_df.to_csv(NEW_MASTER_DF_PATH, index=False)
            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to create / append the master dataframe. \n",
                          "light_grey",
                          force_color=True))
            if AUTO is True:
                new_file_name = VIDEO_IDENT[0] + "_" + str(BEES) + "BEE_analysis.csv"
                new_file_path = os.path.join(directory_input, directories, subdirectories, new_file_name)
                dummybee_df.to_csv(new_file_path)
            if AUTO is False:
                new_file_name = VIDEO_IDENT[0] + "_" + str(BEES) + "BEE_analysis.csv"
                new_file_path = os.path.join(directory, new_file_name)
                dummybee_df.to_csv(new_file_path)
            return

    ########################################################################################################################
    ########################################################################################################################
    # 1 BEE Todo: do same as for two bees
    ########################################################################################################################
    ########################################################################################################################

    elif BEES == 1:
        if len(bdf.name.unique()) > 1:
            c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                             "error": ["traces"]})
            master_df = pd.read_csv(MASTER_DF_PATH)
            master_df = pd.concat([master_df, c_master_df], ignore_index=True)
            master_df.to_csv(MASTER_DF_PATH, index=False)
            if AUTO is True:
                return
            else:
                exit(
                "Too many traces left in .csv! Analysis is not possible! Make sure to select the correct amount of bees!")
        else:
            start_time = time()

            ################################################################################################################
            # CALCULATE TIME BEE WAS NEAR DUMMY / FEATHER
            ################################################################################################################

            bdf.columns = ["frame_number", "name", "B1_x", "B1_y"]
            bdf = bdf.reset_index()

            # print(bdf)

            ax = bdf.plot.scatter(x="B1_x", y="B1_y", color="DarkBlue", label="Bee 1", title="VISUALIZED RAW-DATA")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect('equal')
            if SHOW_PLOTS is True:
                plt.show()

            dummybee_df = pd.merge(dbdf, bdf, how="right", on=["frame_number"])
            dummybee_df = dummybee_df.reset_index()
            dummybee_df = dummybee_df[(dummybee_df.B1_x != -1)]
            dummybee_df = dummybee_df.dropna().reset_index(drop=True)

            """
            if len(dummybee_df.index) < EXPERIMENT_LENGTH_THRESH:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "error": ["length"]
                                                 })
                master_df = pd.read_csv(MASTER_DF_PATH)
                master_df = pd.concat([master_df, c_master_df], ignore_index=True)
                master_df.to_csv(MASTER_DF_PATH, index=False)
                if AUTO is True:
                    return
                else:
                    exit("Too little of the experiment was tracked")
            """
            # print(dummybee_df)

            for index, row in dummybee_df.iterrows():
                XP4 = row["D_x"]
                YP4 = row["D_y"]
                XP3 = row["DH_x"]
                YP3 = row["DH_y"]
                XP2 = row["F1_x"]
                YP2 = row["F1_y"]
                XP1 = row["F2_x"]
                YP1 = row["F2_y"]
                x1 = row["B1_x"]
                y1 = row["B1_y"]

                dummy = dummy_class.Dummy(geolib.Straight(geolib.Vector(XP3, YP3),
                                              geolib.Vector(XP4, YP4)), 0.2)
                feather = dummy_class.Feather(geolib.Vector(XP3, YP3), geolib.Vector(XP2, YP2), geolib.Vector(XP1, YP1))
                Bee_1 = geolib.Vector(x1, y1)
                dist_bd1 = dummy.get_distance(Bee_1)
                dist_bf1 = feather.get_distance(Bee_1)
                if dist_bd1 <= DIST_DUMMY_THRESHOLD:
                    dist_bd1_true_frames.append(1)
                    dummybee_df.at[index, "dist_BD1"] = 1
                else:
                    dist_bd1_true_frames.append(0)
                    dummybee_df.at[index, "dist_BD1"] = 0

                if dist_bf1 <= DIST_FEATHER_THRESHOLD:
                    dist_bf1_true_frames.append(1)
                    dummybee_df.at[index, "dist_BF1"] = 1
                else:
                    dist_bf1_true_frames.append(0)
                    dummybee_df.at[index, "dist_BF1"] = 0

            seconds_near_dummy_1 = sum(dist_bd1_true_frames) / 100
            seconds_near_feather_1 = sum(dist_bf1_true_frames) / 100

            print(colored(f"\n The bee was near the dummy for:", "light_blue", force_color=True))
            print(sum(dist_bd1_true_frames), "frames")
            print(seconds_near_dummy_1, "seconds")
            print(colored(f"\n The bee was near the feather for:", "light_blue", force_color=True))
            print(sum(dist_bf1_true_frames), "frames")
            print(seconds_near_feather_1, "seconds")

            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to calculate the proximity of the bee(s) to the dummy. \n", "light_grey",
                          force_color=True))

            start_time = time()
            dummybee_df["dist_BD1_active"] = np.nan
            dummybee_df["dist_BF1_active"] = np.nan
            for index in dummybee_df.index:
                ############################################################################################################
                # CHECK IF ACTIVE OR PASSIVE
                ############################################################################################################
                if index + DIST_DUMMY_TIME_THRESHOLD in dummybee_df.index:
                    if (dummybee_df["dist_BD1"].iloc[index] == 1 and
                            dummybee_df["dist_BD1"].iloc[index + DIST_DUMMY_TIME_THRESHOLD] == 1):
                        dummybee_df.at[index, "dist_BD1_active"] = 1
                    else:
                        dummybee_df.at[index, "dist_BD1_active"] = 0
                if dummybee_df["dist_BD1"].iloc[index] == 1 and dummybee_df["dist_BD1_active"].iloc[index] == 1:
                    dummybee_df.at[index, "dist_BD1_active"] = 1
                else:
                    dummybee_df.at[index, "dist_BD1_active"] = 0

                ############################################################################################################

                if index + DIST_FEATHER_TIME_THRESHOLD in dummybee_df.index:
                    if dummybee_df["dist_BF1"].iloc[index] == 1 and \
                            dummybee_df["dist_BF1"].iloc[index + DIST_FEATHER_TIME_THRESHOLD] == 1:
                        dummybee_df.at[index, "dist_BF1_active"] = 1
                    else:
                        dummybee_df.at[index, "dist_BF1_active"] = 0
                if dummybee_df["dist_BF1"].iloc[index] == 1 and dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    dummybee_df.at[index, "dist_BF1_active"] = 1
                else:
                    dummybee_df.at[index, "dist_BF1_active"] = 0

                ############################################################################################################
                # CALCULATE SPEED OF THE BEE / ACTIVE TIME
                ############################################################################################################

                if index + 1 in dummybee_df.index:
                    B1_x1 = dummybee_df["B1_x"].iloc[index]
                    B1_x2 = dummybee_df["B1_x"].iloc[index + 1]
                    B1_y1 = dummybee_df["B1_y"].iloc[index]
                    B1_y2 = dummybee_df["B1_y"].iloc[index + 1]

                    B1_1 = geolib.Vector(B1_x1, B1_y1)
                    B1_2 = geolib.Vector(B1_x2, B1_y2)

                    B1_s = B1_1.sub(B1_2).abs()

                    B1_SL.append(B1_s)
                    dummybee_df.at[index, "speed_B1"] = B1_s
                else:
                    pass

            seconds_near_dummy_1_active = dummybee_df["dist_BD1_active"].sum() / 100
            seconds_near_feather_1_active = dummybee_df["dist_BF1_active"].sum() / 100

            print(colored(f"\n The bee was actively near the dummy for:", "light_blue", force_color=True))
            print(dummybee_df["dist_BD1_active"].sum(), "frames")
            print(seconds_near_dummy_1_active, "seconds")
            print(colored(f"\n The bee was actively near the feather for:", "light_blue", force_color=True))
            print(dummybee_df["dist_BF1_active"].sum(), "frames")
            print(seconds_near_feather_1_active, "seconds")

            B1_SL_only_movement = list(filter(lambda num: num != 0, B1_SL))
            mean_speed_1 = sum(B1_SL_only_movement) / len(B1_SL_only_movement)
            mean_speed_1_sec = mean_speed_1 / 100
            percentage_speed_1 = len(B1_SL_only_movement) / len(B1_SL)

            print(colored(f"\n The bee was moving at a speed of:", "light_blue", force_color=True))
            print(mean_speed_1, "px/frame")
            print(mean_speed_1 / 100, "px/s")
            print(colored(f"\n and was moving in", "light_blue", force_color=True))
            print(percentage_speed_1 * 100, "%")
            print(colored(f" of the video.", "light_blue", force_color=True))

            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to calculate the speed of the bee(s) throughout "
                          f"the video and the active contact time. \n",
                          "light_grey",
                          force_color=True))

            ################################################################################################################
            # SPEED BEFORE CONTACT AND WHAT DID BEE CONTACT FIRST VARIABLES
            ################################################################################################################
            start_time = time()
            n = 0
            nf = 0
            n2 = 0
            n2f = 0
            na = 0
            naf = 0
            BD1a_group = []
            BF1a_group = []
            BD1a_group_bf = []
            BF1a_group_bf = []
            dummybee_df["BD1a_group"] = np.nan
            dummybee_df["BF1a_group"] = np.nan
            dummybee_df["speed_b4_BD1"] = np.nan
            dummybee_df["speed_b4_BF1"] = np.nan

            contact_bee_first1 = 0
            contact_dummy_first1 = 0

            for index in dummybee_df.index:
                if dummybee_df["dist_BD1_active"].iloc[index] == 1 and dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    if index + 10 in dummybee_df.index:
                        if (dummybee_df["dist_BF1_active"].iloc[index + 5] == 1 or
                                dummybee_df["dist_BF1_active"].iloc[index + 10] == 1):
                            dummybee_df.at[index, "dist_BD1_active"] = 0
                            dummybee_df.at[index, "dist_BF1_active"] = 1
                        else:
                            dummybee_df.at[index, "dist_BF1_active"] = 0
                    else:
                        dummybee_df.at[index, "dist_BF1_active"] = 0 #todo
                if dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BD1_active"].iloc[index + 1] == 1 or
                                dummybee_df["dist_BD1_active"].iloc[index + 5] == 1):
                            dummybee_df.at[index, "dist_BD1_active"] = 1
                            dummybee_df.at[index, "dist_BF1_active"] = 0
            for index in dummybee_df.index:
                if dummybee_df["dist_BD1_active"].iloc[index] == 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BF1_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 3] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 5] == 1):
                            dummybee_df.at[index, "dist_BF1v"] = 1
                            dummybee_df.at[index, "dist_BD1_active"] = 0
            B1_SL_new = []
            for index in dummybee_df.index:
                if dummybee_df["dist_BD1_active"].iloc[index] == 1:
                    if dummybee_df["dist_BD1_active"].iloc[index - 1] == 0:
                        for i in range(index - 20, index):
                            dummybee_df.at[i, "speed_b4_BD1"] = dummybee_df["speed_B1"].iloc[i]

                if dummybee_df["dist_BF1_active"].iloc[index] == 1:
                    if dummybee_df["dist_BF1_active"].iloc[index - 1] == 0:
                        for i in range(index - 20, index):
                            dummybee_df.at[i, "speed_b4_BF1"] = dummybee_df["speed_B1"].iloc[i]
                ############################################################################################################
                # CONTACT GROUPS
                ############################################################################################################
                na += 1 #todo
                if na > 9:
                    n = 2
                    n2 = 2

                if n > 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BD1_active"].iloc[index] == 1 and
                                dummybee_df["dist_BD1_active"].iloc[index - 1] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 2] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 3] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 4] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 5] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 6] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 7] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 8] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 9] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index - 10] == 0 and
                                dummybee_df["dist_BD1_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BD1_active"].iloc[index + 2] == 1 and
                                dummybee_df["dist_BD1_active"].iloc[index + 3] == 1):
                            dummybee_df.at[index, "BD1a_group"] = dummybee_df["frame_number"].iloc[index]
                            BD1a_group.append(dummybee_df["frame_number"].iloc[index])
                            if STING1 is True:
                                if dummybee_df["frame_number"].iloc[index] < STING1_FRAME:
                                    BD1a_group_bf.append(dummybee_df["frame_number"].iloc[index])


                if n2 > 1:
                    if index + 5 in dummybee_df.index:
                        if (dummybee_df["dist_BF1_active"].iloc[index] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index - 1] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 2] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 3] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 4] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 5] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 6] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 7] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 8] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 9] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index - 10] == 0 and
                                dummybee_df["dist_BF1_active"].iloc[index + 1] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 2] == 1 and
                                dummybee_df["dist_BF1_active"].iloc[index + 3] == 1):
                            dummybee_df.at[index, "BF1a_group"] = dummybee_df["frame_number"].iloc[index]
                            BF1a_group.append(dummybee_df["frame_number"].iloc[index])
                            if STING1 is True:
                                if dummybee_df["frame_number"].iloc[index] < STING1_FRAME:
                                    BF1a_group_bf.append(dummybee_df["frame_number"].iloc[index])


                ############################################################################################################
                # WHAT DID BEE CONTACT FIRST
                ############################################################################################################

                if contact_dummy_first1 == 0 and contact_bee_first1 == 0:
                    if dummybee_df["dist_BD1_active"].iloc[index] == 1:
                        contact_dummy_first1 = 1
                    if dummybee_df["dist_BF1_active"].iloc[index] == 1:
                        contact_dummy_first1 = 1
                if index + 1 in dummybee_df.index:
                    if dummybee_df["dist_BD1_active"].iloc[index] == 0 and dummybee_df["dist_BF1_active"].iloc[index] == 0:
                        B1_x1 = dummybee_df["B1_x"].iloc[index]
                        B1_x2 = dummybee_df["B1_x"].iloc[index + 1]
                        B1_y1 = dummybee_df["B1_y"].iloc[index]
                        B1_y2 = dummybee_df["B1_y"].iloc[index + 1]

                        B1_1 = geolib.Vector(B1_x1, B1_y1)
                        B1_2 = geolib.Vector(B1_x2, B1_y2)

                        B1_s = B1_1.sub(B1_2).abs()

                        B1_SL_new.append(B1_s)
                        dummybee_df.at[index, "speed_B1_wo"] = B1_s
                    else:
                        pass
            if (contact_dummy_first1 + contact_dummy_first1) > 2:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "error": ["contact"]
                                                 })
                master_df = pd.read_csv(MASTER_DF_PATH)
                master_df = pd.concat([master_df, c_master_df], ignore_index=True)
                master_df.to_csv(MASTER_DF_PATH, index=False)
                if AUTO is True:
                    return
                else:
                    exit("One bee can't contact both bee and dummy first at the same time!")
            if contact_dummy_first1 == 1:
                print(colored(f"\nThe bee contacted the dummy!", "light_blue", force_color=True))

            B1_SL_only_movement_new = list(filter(lambda num: num != 0, B1_SL_new))
            mean_speed_1_wo = sum(B1_SL_only_movement_new) / len(B1_SL_only_movement_new)
            mean_speed_1_wo_sec = mean_speed_1_wo / 100


            TIMES_CONTACTED_FEATHER = len(BF1a_group)
            times_cc_wo = len(BD1a_group_bf) + len(BF1a_group_bf)
            print(colored(f"\nThe bee contacted the feather", "light_blue", force_color=True),
                  TIMES_CONTACTED_FEATHER, colored(f"times.", "light_blue", force_color=True))
            ACCURACY = -1
            if STING1 is True:
                if ((STING1_FRAME or STING1_FRAME + 1 or STING1_FRAME - 1 or STING1_FRAME + 2 or STING1_FRAME - 2 ) in
                        dummybee_df["BF1a_group"].values):
                    ACCURACY = 0.5
                else:
                    ACCURACY = 0

            TIMES_CONTACTED_DUMMY = len(BD1a_group)
            print(colored(f"\nThe bee contacted the dummy", "light_blue", force_color=True),
                  TIMES_CONTACTED_DUMMY, colored(f"times.", "light_blue", force_color=True))
            if STING1 is True:
                if ((STING1_FRAME or STING1_FRAME + 1 or STING1_FRAME - 1 or STING1_FRAME + 2 or STING1_FRAME - 2 ) in
                        dummybee_df["BD1a_group"].values):  # todo
                    ACCURACY += 1
                else:
                    ACCURACY += 0

            if ACCURACY == 0:
                print(colored(f"\nThe recorded stinging time is not within the calculated contact time!", "light_blue",
                              force_color=True),
                      colored(f"\nAccuracy-level:", "light_blue", force_color=True), ACCURACY)
            elif ACCURACY == -1:
                print(colored(f"\nNo sting recorded!", "light_blue",
                              force_color=True),
                      colored(f"\nNo accuracy-level!", "light_blue", force_color=True))
            else:

                print(colored(f"\nThe recorded stinging time is within the calculated contact time!", "light_blue",
                              force_color=True),
                      colored(f"\nAccuracy-level:", "light_blue", force_color=True), ACCURACY)

            speed_b4_BD1 = dummybee_df[dummybee_df["speed_b4_BD1"] != 0]["speed_b4_BD1"].mean()
            speed_b4_BF1 = dummybee_df[dummybee_df["speed_b4_BF1"] != 0]["speed_b4_BF1"].mean()
            speed_b4_BD1_sec = speed_b4_BD1 / 100
            speed_b4_BF1_sec = speed_b4_BF1 / 100

            print(
                colored(f"\n20 Frames before the bee was in contact with the dummy (each time) it was moving at ca.:",
                        "light_blue", force_color=True))
            print(speed_b4_BD1, "px/frame")

            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to calculate the speed of the bee(s) near the dummy and to calculate "
                          f"what the bees interacted with first. \n",
                          "light_grey",
                          force_color=True))#

            start_time = time()
            percentage_speed_1 = percentage_speed_1 * 100
            if HAS_DATE is True:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "date": DATE,
                                                 "colony": COLONY,
                                                 "Bees": [1],
                                                 "Trial": TRIAL,
                                                 "IAA": IAA,
                                                 "TRACKED": len(dummybee_df.index),
                                                 "Stings_recorded": STINGS,
                                                 "sec_near_dummy_b1": [seconds_near_dummy_1],
                                                 "sec_near_feather_b1": [seconds_near_feather_1],
                                                 "sec_near_dummy_b1_active": [seconds_near_dummy_1_active],
                                                 "sec_near_feather_b1_active": [seconds_near_feather_1_active],
                                                 "times_contacted_dummy_b1": [TIMES_CONTACTED_DUMMY],
                                                 "times_contacted_feather_b1": [TIMES_CONTACTED_FEATHER],
                                                 "accuracy_stings": [ACCURACY],
                                                 "mean_speed_b1_sec": [mean_speed_1_sec],
                                                 "percentage_moved_b1": [percentage_speed_1],
                                                 "b1_speed_bf_dc_sec": [speed_b4_BD1_sec],
                                                 "b1_speed_bf_fc_sec": [speed_b4_BF1_sec],
                                                 "b1_contact_b_first": [contact_bee_first1],
                                                 "b1_contact_d_first": [contact_dummy_first1],
                                                 "mean_speed_wo": [mean_speed_1_wo_sec],
                                                 "times_cc_wo":[times_cc_wo]
                                                 })
            else:
                c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                 "colony": COLONY,
                                                 "Bees": [1],
                                                 "Trial": TRIAL,
                                                 "IAA": IAA,
                                                 "TRACKED": len(dummybee_df.index),
                                                 "Stings_recorded": STINGS,
                                                 "sec_near_dummy_b1": [seconds_near_dummy_1],
                                                 "sec_near_feather_b1": [seconds_near_feather_1],
                                                 "sec_near_dummy_b1_active": [seconds_near_dummy_1_active],
                                                 "sec_near_feather_b1_active": [seconds_near_feather_1_active],
                                                 "times_contacted_dummy_b1": [TIMES_CONTACTED_DUMMY],
                                                 "times_contacted_feather_b1": [TIMES_CONTACTED_FEATHER],
                                                 "accuracy_stings": [ACCURACY],
                                                 "mean_speed_b1_sec": [mean_speed_1_sec],
                                                 "percentage_moved_b1": [percentage_speed_1],
                                                 "b1_speed_bf_dc_sec": [speed_b4_BD1_sec],
                                                 "b1_speed_bf_fc_sec": [speed_b4_BF1_sec],
                                                 "b1_contact_b_first": [contact_bee_first1],
                                                 "b1_contact_d_first": [contact_dummy_first1],
                                                 "mean_speed_wo": [mean_speed_1_wo_sec],
                                                 "times_cc_wo":[times_cc_wo]
                                                 })
            print(c_master_df)
            if MASTER_DF_EXISTENT is True:
                master_df = pd.read_csv(MASTER_DF_PATH)
                master_df = pd.concat([master_df, c_master_df], ignore_index=True)
                master_df.to_csv(MASTER_DF_PATH, index=False)
            if MASTER_DF_EXISTENT is False:
                c_master_df.to_csv(NEW_MASTER_DF_PATH, index=False)
            print(colored(f"\n It took {gethostname()} {round(time() - start_time, 3)}"
                          f" seconds to create / append the master dataframe. \n",
                          "light_grey", force_color=True))
            if AUTO is True:
                new_file_name = VIDEO_IDENT[0] + "_" + str(BEES) + "BEE_analysis.csv"
                new_file_path = os.path.join(directory_input, directories, subdirectories, new_file_name)
                dummybee_df.to_csv(new_file_path)
            if AUTO is False:
                new_file_name = VIDEO_IDENT[0] + "_" + str(BEES) + "BEE_analysis.csv"
                new_file_path = os.path.join(directory, new_file_name)
                dummybee_df.to_csv(new_file_path)

            return

    else:
        c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                         "error": ["bees"]})
        master_df = pd.read_csv(MASTER_DF_PATH)
        master_df = pd.concat([master_df, c_master_df], ignore_index=True)
        master_df.to_csv(MASTER_DF_PATH, index=False)
        if AUTO is True:
            return
        else:
            exit("It is not possible to analyze more than 2 bees at the moment!")


if AUTO is True:
    directory = os.path.basename(directory_input)
    for directories in next(os.walk(directory_input))[1]:
        folder = directories
        for subdirectories in next(os.walk(os.path.join(directory_input, directories)))[1]:
            wd = os.listdir(os.path.join(directory_input, directories, subdirectories))
            for filename_input in wd:
                dbdf_PATH = 0
                print(filename_input)
                if filename_input.endswith("1BEE_nn.csv"):
                    bdf_PATH = os.path.join(directory_input,directories, subdirectories, filename_input)
                    VIDEO_IDENT = re.findall(r"(\d{9})", bdf_PATH)
                else:
                    continue

                check_df = pd.read_csv(MASTER_DF_PATH)
                if VIDEO_IDENT in check_df["video_ident"].values:
                    print(colored(f"\n VIDEO ALREADY IN MASTER CSV! SKIPPING THIS ONE...", "yellow", force_color=True))
                    continue
                for ind, filename_input in enumerate(wd):
                    dbdf_PATH = 0
                    if filename_input.startswith(VIDEO_IDENT[0]):
                        if filename_input.endswith("_ddf_finished.csv"):
                            dbdf_PATH = os.path.join(directory_input, directories, subdirectories, filename_input)
                            round_time = time()
                            bees_main(bdf_PATH, dbdf_PATH)
                            print(colored(f"\nIt took {gethostname()} {round(time() - round_time, 3)}"
                                          f" seconds in total. \n", "light_grey", force_color=True))
                        elif ind == len(wd) -1 and dbdf_PATH == 0:
                            c_master_df = pd.DataFrame(data={"video_ident": VIDEO_IDENT,
                                                             "error": ["no dummy"]})
                            master_df = pd.read_csv(MASTER_DF_PATH)
                            master_df = pd.concat([master_df, c_master_df], ignore_index=True)
                            master_df.to_csv(MASTER_DF_PATH, index=False)
                            continue
else:
    bees_main(bdf_PATH, dbdf_PATH)
print(colored(f"\nIt took {gethostname()} {round(time() - total_time, 3)}"
                              f" seconds in total. \n", "light_grey", force_color=True))

