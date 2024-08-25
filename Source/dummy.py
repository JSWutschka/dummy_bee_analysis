"""
dummy.py
Supporting script for processing the _ai.csv raw data file

Author
   Julia Wutschka

Usage
   This script should be called from main.py.
   **Running from command line**
   You may run it from command line: python dummy.py <BASEFILENAME>
   Arguments:
       * BASEFILENAME
         prefix of the test file including the (the code's relative) path
   Example:
        python dummy.py "INPUT_DATEN\C3_19052022\1BEE\190604032" "_1BEE_nn.csv"

Files
    **BASEFILENAME.json**
        The found values will be stored in BASEFILENAME.json. If it doesn't exist, it will be created.
        These values are:
        *   *ai-csv-file*
            name of the file containng the dummy data from loopy
        *   *tmp_dummy_ready*
            name of the output-csv file. Default it is base_name + "_tmp_dummy_ready.csv".
        *   *plot_select_D*
            name of the image file showing the selection of point D ("Broom 4").
            Default value is base_name + "_plot_selection_dummy_points.png".
        *   *center_x*
            x-value of the calculated center_point.
        *   *center_y*
            x-value of the calculated center_point.
        *   *center_radius*
            The radius of the "Broom4-circle". It shall be the half dummy-length.
        *   *catch_distance*
            Maximum distance between a point marked as "Broom4" and the calculated circle as the point
            will be accepted. Default value is catching_ring_use_default/2 (see global_constants.py)

    **BASEFILENAME_tmp_dummy_ready.csv**
        We will create this CSV-table with each frame in one raw. This is only used by rebuild.py.
"""
import json
import math
import sys
import matplotlib.widgets
import numpy as np
from termcolor import colored
import pandas
import geolib
import matplotlib.pyplot as plt
from time import time
from _socket import gethostname

import global_constants

"""
Process the ai raw data file
Command-Line: Base-Name of the test e.g.09072024/D183141315
Generates file:
*_dummy_finished.csv: 
    csv-table
    format [unnamed index], frame_count,....
_json:
    Will be modified, if it already exists
    Contains information of the test

"""

def main(base_name, wrapper=global_constants.Wrapper_class()):
    """
    Main function of dummy.py

    :param base_name: prefix of the test file including the (the code's relative) path
    :param wrapper: wrapper-object containing values should be passed by reference. see global_constants.py
    :return: -
    """
    ####################################################################################################################
    # Setup processing:
    ####################################################################################################################
    ddf_raw = wrapper.ddf_raw
    info_filename = base_name + ".json"
    info_block = {
        "ai-csv-file": base_name + "_DUMMY_ai.csv",
        "dummy-ready-file": base_name + "_dummy_ready.csv",
        "tmp_dummy_ready": base_name + "_tmp_dummy_ready.csv",
        "plot_select_D": base_name + "_plot_selection_dummy_points.png"
    }
    print(colored("Reading File:", "red", force_color=True), info_filename)

    try:
        with open(info_filename, "r") as in_file:
            info_block = json.load(in_file)
            in_file.close()
        if not ("ai-csv-file" in info_block):
            info_block["ai-csv-file"] = base_name + "_DUMMY_ai.csv"
        if not ("tmp_dummy_ready" in info_block):
            info_block["tmp_dummy_ready"] = base_name + "_tmp_dummy_ready.csv"
        if not ("dummy_ready" in info_block):
            info_block["dummy_ready"] = base_name + "_dummy_ready.csv"
        if not ("plot_select_D" in info_block):
            info_block["plot_select_D"] = base_name + "_plot_selection_dummy_points.png"

    except IOError:
        print(colored(f"File:", "light_blue", force_color=True), info_filename,
              colored(f"does not yet exist. It will be created after completion", "light_blue", force_color=True))

    finally:
        pass

    ####################################################################################################################
    # Step 1 Show source data in a diagram and identify the dummy circle points
    # Notice: because there are a lot of points we will show only every 10th point for performance reasons
    ####################################################################################################################
    # We need
    #   circle_base: a circle around the dummy points; it is defined by selecting 3 points on the plot
    #   catch_distance: all points closer than the catch_distance to the circle will be used as dummy-points
    ####################################################################################################################

    print(colored(f"Reading source data... ", "blue", force_color=True))
    start_time = time()
    if ddf_raw.empty:
        source_dataframe = pandas.read_csv(info_block["ai-csv-file"])
        ddf_raw = source_dataframe
    else:
        source_dataframe = ddf_raw
    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds read the source data.", "light_grey",
                  force_color=True))

    # noinspection PyTypeChecker
    circle_base: geolib.Circle = None
    catch_distance = 0

    ####################################################################################################################
    # Interactive selecting starts here
    ####################################################################################################################
    ddf_sorted = source_dataframe[["frame_number", "oid", "x", "y"]].pivot(columns="oid", values=["x", "y"],
                                                                          index=["frame_number"])
    ddf_sorted.columns = ["".join(str(col)) for col in ddf_sorted.columns.values]
    ddf_sorted = ddf_sorted.rename(
        columns={"('x', 1)": "F1_x", "('y', 1)": "F1_y", "('x', 2)": "F2_x", "('y', 2)": "F2_y",
                 "('x', 3)": "DH_x", "('y', 3)": "DH_y", "('x', 4)": "D_x", "('y', 4)": "D_y"})
    ddf_sorted = ddf_sorted.reset_index()
    ddf_sorted["xraw"] = ddf_sorted["D_x"]
    fig, ax = plt.subplots()
    df_plot = ddf_sorted.iloc[::10, ]

    df_plot.plot.scatter(x="D_x", y="D_y", color="LightGrey", label="D", ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')

    circle_inner_drawing = None
    circle_base_drawing = None
    circle_outer_drawing = None
    list_of_points = []
    is_finished = False
    # is_finished: flag will be True when a click outside of plot for finishing is recognized
    # and the selecting is complete
    is_border_mode = False
    # is_border_mode: is True if the next click should define the border/catching-distance
    is_complete = False
    # is_complete: will set to True when circle and catching-distance is set

    ####################################################################################################################

    def onclick(event):
        nonlocal circle_inner_drawing, circle_base_drawing, circle_outer_drawing, list_of_points, cid, \
            circle_base, is_finished, is_border_mode, is_complete, catch_distance, cid, ax, fig

        # Eventhandler called by matplotlib.pyplot
        draw_border = False
        if event.xdata is None:
            if is_complete:
                fig.canvas.mpl_disconnect(cid)
                is_finished = True
            else:
                print(colored(
                    f"\n PLEASE CLICK ON THREE POINTS TO CREATE THE CIRCLE. CLICK ONCE MORE TO CREATE THE BORDER",
                    "yellow", force_color=True))
        else:
            if is_complete:
                is_border_mode = False
                list_of_points = []
                circle_inner_drawing.remove()
                circle_base_drawing.remove()
                circle_outer_drawing.remove()
                fig.canvas.draw()
                is_complete = False
            if is_border_mode:
                catch_distance = math.fabs(circle_base.radius -
                                           geolib.Vector(event.xdata, event.ydata).sub(circle_base.center_point).abs())
                draw_border = True
            else:
                point = geolib.Vector(event.xdata, event.ydata)
                list_of_points.append(point)
                print(colored(f" Adding to the coordinates that define the circle: ",
                              "light_blue", force_color=True), "Point", len(list_of_points),
                      colored(f":", "light_blue", force_color=True), point)

                if len(list_of_points) == 3:
                    circle_base = geolib.Circle(list_of_circle_points=list_of_points)
                    circle_base_drawing = plt.Circle((circle_base.center_point.x, circle_base.center_point.y),
                                                     circle_base.radius, color="magenta", fill=False)
                    ax.add_patch(circle_base_drawing)
                    is_border_mode = True
                    if global_constants.catching_ring_use_default:
                        catch_distance = round(global_constants.catching_ring_width / 2, 0)
                        draw_border = True
                    else:
                        print(
                            colored(f"\nSELECT THE MOST OUTER COORDINATE (AI: LIGHT GREY) OF THE PREVIOUSLY DEFINED CIRCLE "
                                    f"TO SELECT THE CATCHING RING",
                                    "yellow", force_color=True))
            if draw_border:
                circle_inner_drawing = plt.Circle((circle_base.center_point.x, circle_base.center_point.y),
                                                  circle_base.radius - catch_distance, color="r", fill=False)
                circle_outer_drawing = plt.Circle((circle_base.center_point.x, circle_base.center_point.y),
                                                  circle_base.radius + catch_distance, color="r", fill=False)
                ax.add_patch(circle_inner_drawing)
                ax.add_patch(circle_outer_drawing)
                is_complete = True
                is_border_mode = False
                fig.canvas.draw()
                print(colored(f"TO FINISH PLEASE CLICK OUTSIDE OF THE MAIN PLOT!\n", "yellow", force_color=True))

    ####################################################################################################################
    # noinspection PyTypeChecker
    if global_constants.unattended_mode:
        d_points = ddf_sorted
        d_points = d_points.dropna(subset=['D_x']).dropna(subset=['D_y']).sort_values("D_x")
        d_points_count = d_points.shape[0]
        head_percent = (1 - global_constants.unattended_range_d_level) / 2.0
        d_points = d_points.tail(int(d_points_count * (1 - head_percent)))
        d_points = d_points.head(int(d_points_count * global_constants.unattended_range_d_level)).sort_values("D_y")
        d_points_count = d_points.shape[0]
        d_points = d_points.tail(int(d_points_count * (1 - head_percent)))
        d_points = d_points.head(int(d_points_count * global_constants.unattended_range_d_level))
        min_x = d_points["D_x"].min()
        min_y = d_points["D_y"].min()
        max_x = d_points["D_x"].max()
        max_y = d_points["D_y"].max()
        center_point = geolib.Vector((max_x + min_x) / 2.0, (max_y + min_y) / 2.0)
        outer_radius = ((max_x + min_x) / 2.0 - min_x + (max_y + min_y) / 2.0 - min_y) / 2.0
        catch_distance = round(global_constants.catching_ring_width / 2, 0)
        circle_base = geolib.Circle(center_point, outer_radius)
        circle_base_drawing = plt.Circle((circle_base.center_point.x, circle_base.center_point.y),
                                         circle_base.radius, color="magenta", fill=False)
        ax.add_patch(circle_base_drawing)
        circle_inner_drawing = plt.Circle((circle_base.center_point.x, circle_base.center_point.y),
                                          circle_base.radius - catch_distance, color="r", fill=False)
        circle_outer_drawing = plt.Circle((circle_base.center_point.x, circle_base.center_point.y),
                                          circle_base.radius + catch_distance, color="r", fill=False)
        ax.add_patch(circle_inner_drawing)
        ax.add_patch(circle_outer_drawing)
        fig.canvas.draw()
        plt.show(block=False)
    else:
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        print(colored(f"CLICK 3 TIMES ON TO THE LIGHT GREY CIRCLE (CREATING A TRIANGLE) "
                      f"TO SELECT MARK THE MAIN DUMMY-ROTATION-CIRCLE!", "yellow", force_color=True))
        print(colored(f"THEN SELECT THE MOST OUTER COORDINATE (AI: LIGHT GREY) OF THE PREVIOUSLY DEFINED CIRCLE",
                      "yellow", force_color=True))
        print(colored(f"CLICK OUTSIDE OF THE MAIN PLOT TO FINISH\n", "yellow", force_color=True))
        cursor = matplotlib.widgets.Cursor(ax, useblit=True, color='red', linewidth=2)
        plt.show(block=False)
        while not is_finished:
            plt.pause(0.1)

    print(colored(f" The selected center of the arena is at: ", "light_blue", force_color=True),
          circle_base.center_point)
    print(colored(f"\nContinuing to process coordinates...", "blue", force_color=True))

    ####################################################################################################################
    # Step 2: Extract point inner the catching ring
    # circle_points: all points inside the ring
    ####################################################################################################################

    start_time = time()
    total_time = time()
    range_min2 = (circle_base.radius - catch_distance) * (circle_base.radius - catch_distance)
    range_max2 = (circle_base.radius + catch_distance) * (circle_base.radius + catch_distance)
    ddf_sorted["dist_d"] = ((ddf_sorted["D_x"] - circle_base.center_point.x) *
                            (ddf_sorted["D_x"] - circle_base.center_point.x) +
                            (ddf_sorted["D_y"] - circle_base.center_point.y) *
                            (ddf_sorted["D_y"] - circle_base.center_point.y))
    ddf_sorted["in_circle"] = (range_max2 >= ddf_sorted["dist_d"]) & (ddf_sorted["dist_d"] > range_min2)
    # now get all points D which are within the catching circles
    circle_points = ddf_sorted[ddf_sorted["in_circle"]]
    circle_points = circle_points.reset_index(drop=True)
    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds to extract all points inside the given catching ring.", "light_grey",
                  force_color=True))

    ####################################################################################################################
    # Step 3: Estimate the real midpoint
    ####################################################################################################################
    # Dummy rotation is about 5 rounds per 11 sec => 1100 Frames / 5 rounds -> about 220 frames/round
    # we will select points every 50 frames so that the angel between the points measures 50/220*360 = 82°
    # Estimated center point will be the median of all potential centers
    ####################################################################################################################

    start_time = time()
    step = 50
    max_index = len(circle_points.index)
    tmp_circle_points:list[geolib.Vector] = []
    list_center_points = []
    for start_index in range(0, step, int(step / 3)):
        for next_index in range(start_index, max_index, step):
            tmp_circle_points.append(geolib.Vector(circle_points.loc[next_index].D_x, circle_points.loc[next_index].D_y))
            if len(tmp_circle_points) == 3:
                c = geolib.Circle(list_of_circle_points=tmp_circle_points)
                list_center_points.append([c.center_point.x, c.center_point.y])
                tmp_circle_points = []
    df_list_center_points = pandas.DataFrame(list_center_points, columns=["x", "y"])
    pd_center_point = df_list_center_points.median()
    center_point = geolib.Vector(pd_center_point["x"], pd_center_point["y"])
    # "dist" square distance between each circle point (selected manually above) and the calculated circle center point
    circle_points["dist"] = ((circle_points["D_x"] - center_point.x) * (circle_points["D_x"] - center_point.x) +
                             (circle_points["D_y"] - center_point.y) * (circle_points["D_y"] - center_point.y))
    mean_distance2 = circle_points["dist"].mean()
    mean_distance = math.sqrt(mean_distance2)  # mean dist from center
    circle_base = geolib.Circle(center_point, mean_distance)
    circle_base_drawing_new = plt.Circle((circle_base.center_point.x, circle_base.center_point.y),
                                         circle_base.radius, color="blue", fill=False)
    ax.add_patch(circle_base_drawing_new)
    fig.canvas.draw()
    plt.pause(0.1)
    plt.savefig(info_block["plot_select_D"])
    if global_constants.unattended_mode:
        plt.close('all')
    print(colored(f" The detected center of the arena is at: ","light_blue", force_color=True), center_point)
    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds to calculate the center of the arena.", "light_grey",
                  force_color=True))

    ####################################################################################################################
    # Step 4: Extract valid datasets for D, i.e. select only those datasets where D is inside the catching ring
    #         set D=nan for all other datasets and add them
    ####################################################################################################################

    start_time = time()
    valid_frames = ddf_sorted
    valid_frames["distance2"] = ((valid_frames["D_x"] - center_point.x) *
                                 (valid_frames["D_x"] - center_point.x) +
                                 (valid_frames["D_y"] - center_point.y) *
                                 (valid_frames["D_y"] - center_point.y))
    valid_frames["diff_distance2"] = valid_frames["distance2"] - mean_distance2
    valid_frames["diff_distance2"] = valid_frames["diff_distance2"].abs()
    valid_frames['in_catching_circle'] = ((range_max2 >= valid_frames["distance2"]) &
                                          (valid_frames["distance2"] > range_min2))
    other_frames = valid_frames[~valid_frames['in_catching_circle']]
    other_frames.loc[:, "D_x"] = np.nan
    other_frames.loc[:, "D_y"] = np.nan
    valid_frames = valid_frames[valid_frames['in_catching_circle']]
    ddf_sorted = pandas.concat([valid_frames, other_frames])
    ddf_sorted.sort_values("frame_number")
    ddf_sorted.to_csv(info_block["tmp_dummy_ready"])
    print(colored(f" Data saved as:", "light_blue", force_color=True), info_block["tmp_dummy_ready"])
    print(colored(f"It took on {gethostname()} {round(time() - start_time, 3)}"
                  f" seconds save the data.", "light_grey",
                  force_color=True))

    info_block["center_x"] = circle_base.center_point.x
    info_block["center_y"] = circle_base.center_point.y
    info_block["center_radius"] = circle_base.radius
    info_block["catch_distance"] = catch_distance
    wrapper.ddf_sorted = ddf_sorted
    wrapper.ddf_raw = ddf_raw

    ####################################################################################################################

    with open(info_filename, "w") as outfile:
        json.dump(info_block, outfile)
        outfile.close()

    print(colored(f"\n Test-info.json file is saved/updated as:", "light_blue", force_color=True), info_filename)
    print(colored(f"It took on {gethostname()} {round(time() - total_time, 3)}"
                  f" seconds in total. ", "light_grey",
                  force_color=True))

#
# If runs alone, get base_name from command line
#
if __name__ == "__main__":
    if len(sys.argv) != 2:  # dem script wird ein parameter übergeben (dateipfad)
        print(colored(f"MISSING BASE NAME! DERIVED NAMES ARE:"
                      f"\n*.json - TEST INFORMATION"
                      f"\n*_DUMMY_ai.csv - RAW CSV DATA (_ai .CSV)"
                      f"\n*_dummy_src_from_raw_ai.csv - PROCESSED DUMMY DATA", "red", force_color=True))
        exit(1)
    base_name = sys.argv[1]
    main(base_name)
