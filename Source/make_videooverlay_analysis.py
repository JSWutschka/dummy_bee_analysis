"""
make_videooverlay_analysis.py
Creates a video overlay displaying the calculated objects and the bee

Author
   Julia Wutschka

Usage
   This script should be called from main.py.
   **Running from command line**
   You may run it from command line: python rebuild.py <BASEFILENAME> <VIDEOFILENAME>
   Arguments:
       * BASEFILENAME
         prefix of the test file including the (the code's relative) path
       * VIDEOFILENAME
         suffix of the video-file

   Example:
        python rebuild.py "input\C3_19052022\1BEE\190604032" "_1BEE.mp4"

Files
    **BASEFILENAME.json**
        We get Necessary values from and put calculated values in BASEFILENAME.json. It is NECESSARY for this script.
        Values put are:
        *   *dummy-overlay-video*
            Suffix for a recorded overlay video. Default value is base_name + "_visualized_data.avi".
    **Overlay-Video**

"""

import json
import pandas
import geolib
from time import time
import cv2
# pycharm-shell:  pip install opencv-python
import sys
import dummy_class
from termcolor import colored
import global_constants
#
# Global constants
#
max_tracking_length = 50
# length (frames) of the bee's moving track displayed
max_action_length = 25
# duration (frames) a contact is shown
color_action = (0, 0, 255)

#
# Setup processing:
#
def main(base_name, wrapper=global_constants.Wrapper_class()):
    info_filename = base_name + ".json"

    try:
        with open(info_filename, "r") as in_file:
            info_block = json.load(in_file)
            in_file.close()
    except IOError:
        raise Exception("File ", info_filename, "does not exist. We will need it for processing...")
    finally:
        pass
    # Creation of new Parameter here:
    info_block["dummy-src-video"] = base_name + wrapper.video_filename
    if not ("dummy-overlay-video" in info_block):
        info_block["dummy-overlay-video"] = base_name + "_visualized_data.avi"
    #
    # -----------------------------------------------------
    #

    total_time = time()
    analysis_df = pandas.read_csv(wrapper.analysis_df)
    if analysis_df.empty:
        print("Reading bee source data from", info_block["analysis-data-file"])
        analysis_df = pandas.read_csv(info_block["analysis-data-file"])
    print("Opening video file", info_block["dummy-src-video"])
    cap = cv2.VideoCapture(info_block["dummy-src-video"])
    try:
        frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    except AttributeError:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(info_block["dummy-overlay-video"],
                             cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))

    cx = info_block["center_x"]
    cy = info_block["center_y"]
    of_x = 0
    of_y = 0
    if "video-offset-x" in info_block:
        of_x = info_block["video-offset-x"]
    if "video-offset-y" in info_block:
        of_y = info_block["video-offset-y"]
    dummy_len = info_block["center_radius"] * 2
    dist_dummy_threshold = dummy_len * 1/5 # COUNTING FOR CONTACT
    dist_feather_threshold = dummy_len * 2/15

    def plot_box(img, x,y, caption, color):
        cv2.rectangle(img, (int(x-3),int(y+3)), (int(x+3), int(y-3)), color, -1)
        cv2.putText(img, caption, (int(x+5),int(y+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def plot_box_bee(img, x,y, color, track):
        size = 30
        cv2.rectangle(img, (int(x - size),int(y + size)), (int(x + size), int(y - size)), color, thickness=1)
        cv2.line(img, (int(x), int(y + size + 10)), (int(x), int(y - size - 10)), color, thickness=1)
        cv2.line(img, (int(x + size + 10), int(y)), (int(x - size - 10), int(y)), color, thickness=1)
        reverse_track = track[::-1]
        for i, point in enumerate(reverse_track):
            if i > 0:
                cv2.line(img, (point[0], point[1]), (reverse_track[i-1][0], reverse_track[i-1][1]),
                        (int(color[0] - (color[0] - 128) * i / max_tracking_length),
                         int(color[1] - (color[1] - 128) * i / max_tracking_length),
                         int(color[2] - (color[2] - 128) * i / max_tracking_length )), thickness=2)

    def move_color(base_color: tuple[int, int, int], target_color: tuple[int, int, int], level):
        col = list(range(3))
        for i in range(3):
            col[i] = (target_color[i] - base_color[i]) * level + base_color[i]
        return tuple(col)
    def onMouse(event, x, y, flags, param):
        nonlocal of_x, of_y, cx, cy
        if event == cv2.EVENT_LBUTTONDOWN:
            # draw circle here (etc...)
            print('x = %d, y = %d' % (x, y))
            of_x = int(x - cx)
            of_y = int(y - cy)

    video_dataframe = pandas.merge(pandas.DataFrame(list(range(0, frames)), columns=["frame_number"]),
                                    analysis_df,
                                    how="left", left_on="frame_number", right_on="frame_number")
    video_dataframe.set_index("frame_number")
    max_frame_number = video_dataframe["frame_number"].max()
    cv2.namedWindow('Bee Analysis', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('Bee Analysis', onMouse)
    cv2.resizeWindow('Bee Analysis', 600, 600)
    first_frame = int(info_block["dummy-rotation-file-start_frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    recording = False
    feather_type = 0

    if "dist_BD2" in analysis_df.columns:
        bee_list = ["1", "2"]
    else:
        bee_list = ["1"]
    action_traces = {}
    for bee in bee_list:
        action_traces[bee] = {
            "dummy_active": 0,
            "feather_active": 0,
            "track": []
        }
    while cap.isOpened():
        ret, frame = cap.read()
        frame_number = int(cap.get(1))
        cv2.putText(frame, "Frame " + frame_number.__str__(), (5,100), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 112, 0), 3)
        if not pandas.isna(video_dataframe.at[frame_number, "D_x"]):
            dummy = dummy_class.Dummy(geolib.Straight(geolib.Vector(video_dataframe.at[frame_number, "D_x"],
                                                                    video_dataframe.at[frame_number, "D_y"]),
                                                      geolib.Vector(video_dataframe.at[frame_number, "DH_x"],
                                                                    video_dataframe.at[frame_number, "DH_y"])),

                                      0.2)
            feather = dummy_class.FeatherPattern(dummy,
                                                 geolib.Vector(video_dataframe.at[frame_number, "F1_x"],
                                                               video_dataframe.at[frame_number, "F1_y"]),
                                                 geolib.Vector(video_dataframe.at[frame_number, "F2_x"],
                                                               video_dataframe.at[frame_number, "F2_y"]),
                                                 feather_type)
            cv2.line(frame, (int(dummy.point_r.x + of_x), int(dummy.point_r.y + of_y)),
                            (int(dummy.point_r2.x + of_x), int(dummy.point_r2.y + of_y)),
                                 (255,174,0),2 )
            dummy.set_zones([("inner", dist_dummy_threshold)])
            feather.set_zones([("inner", dist_feather_threshold)])

            for bee in bee_list:
                bx = int(video_dataframe.at[frame_number, "B" + bee + "_x"] + of_x)
                by = int(video_dataframe.at[frame_number, "B" + bee + "_y"] + of_y)
                action_traces[bee]["track"].append((bx, by))
                action_traces[bee]["track"] = action_traces[bee]["track"][-max_tracking_length:]
                plot_box_bee(frame, bx, by, (0,216,165), action_traces[bee]["track"])
                if video_dataframe.at[frame_number, "dist_BD" + bee + "_active"] == 1:
                    action_traces[bee]["dummy_active"] = max_action_length
                elif action_traces[bee]["dummy_active"] > 0:
                    action_traces[bee]["dummy_active"] -= 1
                if video_dataframe.at[frame_number, "dist_BF" + bee + "_active"] == 1:
                    action_traces[bee]["feather_active"] = max_action_length
                elif action_traces[bee]["feather_active"] > 0:
                    action_traces[bee]["feather_active"] -= 1
                if action_traces[bee]["dummy_active"] > 0:
                    color = move_color(dummy.color_figure, color_action,
                                       (action_traces[bee]["dummy_active"]) / max_action_length)
                    dummy.plot_cv2(frame, geolib.Vector(of_x, of_y)) #, color, 4
                else:
                    dummy.plot_cv2(frame, geolib.Vector(of_x, of_y))
                if action_traces[bee]["feather_active"] > 0:
                    pass
                    color = move_color(feather.color_figure, color_action,
                                       (action_traces[bee]["feather_active"]) / max_action_length)
                    feather.plot_cv2(frame, geolib.Vector(of_x, of_y)) #, color, 4
                else:
                    feather.plot_cv2(frame, geolib.Vector(of_x, of_y))
                for p in [("D", "Broom 4"),  ("F1", "Broom 1"), ("F2", "Broom 2")]:
                    if not pandas.isna(video_dataframe.at[frame_number, p[0] + "_x"]):
                        plot_box(frame, video_dataframe.at[frame_number, p[0] + "_x"] + of_x,
                                        video_dataframe.at[frame_number, p[0] + "_y"] + of_y, p[1],  (255,174,0))
            if recording:
                writer.write(frame)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            print("Paused - press SPACE to continue")
            while not  cv2.waitKey(1) == ord(' '):
                pass
        else:
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                print("Recording:", recording)
            elif key == ord(' '):
                print("Frame:", frame_number)
            elif key == ord('1'):
                feather_type = 1
            elif key == ord('0'):
                feather_type = 0
            elif key in [ord('a'), ord('s'), ord('d'), ord('f')]:
                if key == ord('a'):
                    frame_number -= 1000
                elif key == ord('s'):
                    frame_number -= 100
                elif key == ord('d'):
                    frame_number += 100
                elif key == ord('f'):
                    frame_number += 1000
                if frame_number < 0:
                    frame_number = 0
                if frame_number >= max_frame_number:
                    frame_number = max_frame_number - 1
                for bee in bee_list:
                    action_traces[bee] = {
                        "dummy_active": 0,
                        "feather_active": 0,
                        "track": []
                    }
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


    info_block["video-offset-x"] = of_x
    info_block["video-offset-y"] = of_y

    with open(info_filename, "w") as outfile:
        json.dump(info_block, outfile)
        outfile.close()

    print("Test-information file is updated", info_filename)
    print("Total time:", round(time() - total_time, 2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(colored(f"MISSING ARGUMENTS: <BASEFILENAME> <VIDEOFILENAME>", "red", force_color=True))
        exit(1)
    base_name = sys.argv[1]
    wrapper = global_constants.Wrapper_class()
    wrapper.video_filename = sys.argv[2]
    main(base_name, wrapper)
