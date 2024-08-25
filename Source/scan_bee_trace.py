"""
scan_bee_trace.py
Supporting script for scanning the bee-tracking file to get the first and the last frame number. We need these values
because we want to rebuild the dummy and feather only for the part of the video where loopy tracked the bees

Author
   Julia Wutschka

Usage
   This script should be called from main.py.
   **Running from command line**
   You may run it from command line: python scan_bee_trace.py <BASEFILENAME> <BEEFILENAME>
   Arguments:
       * BASEFILENAME
         prefix of the test file including the (the code's relative) path
       * BEEFILENAME
         suffix of the bee-data-file
   Example:
        python scan_bee_trace.py "input\C3_19052022\1BEE\190604032" "_1BEE_nn.csv"

Files
    **BASEFILENAME.json**
    The calculated values will be stored in BASEFILENAME.json. If it doesn't exist yet, it will be created.
    These values are:
        * *bee-track-first-frame*
        * +bee-track-last-frame*
"""
import json
import sys
from termcolor import colored
import pandas
import global_constants


def main(base_name, wrapper=global_constants.Wrapper_class()):
    """
    Main function of scan_bee_trace.

    :param base_name: prefix of the test file including the (the code's relative) path
    :param wrapper: wrapper-object containing values should be passed by reference. see global_constants.py
    :return: -
    """
    ####################################################################################################################
    # Setup processing:
    ####################################################################################################################
    bdf = wrapper.bdf
    info_filename = base_name + ".json"
    info_block = {
        "bee-track-file": base_name+ wrapper.bee_filename
    }
    print(colored("Reading File:", "red", force_color=True), info_filename)

    try:
        with open(info_filename, "r") as in_file:
            info_block = json.load(in_file)
            in_file.close()
        if not ("bee-track-file" in info_block):
            info_block["bee-track-file"] = base_name + wrapper.bee_filename

    except IOError:
        print(colored(f"File:", "light_blue", force_color=True), info_filename,
              colored(f"does not yet exist. It will be created after completion", "light_blue", force_color=True))

    finally:
        pass

    if bdf.empty:
        bdf = pandas.read_csv(info_block["bee-track-file"])

    info_block["bee-track-first-frame"] = bdf["frame_number"].min() * 1.0
    info_block["bee-track-last-frame"] = bdf["frame_number"].max() * 1.0

    with open(info_filename, "w") as outfile:
        json.dump(info_block, outfile)
        outfile.close()

    print(colored(f"\n Test-info.json file is saved/updated as:", "light_blue", force_color=True), info_filename)


#
# If runs alone, get base_name from command line
#
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(colored(f"MISSING ARGUMENTS: <BASEFILENAME> <BEEFILENAME>", "red", force_color=True))
        exit(1)
    base_name = sys.argv[1]
    wrapper = global_constants.Wrapper_class()
    wrapper.bee_filename = sys.argv[2]
    main(base_name)