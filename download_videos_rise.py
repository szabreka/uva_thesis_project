"""
This script downloads all videos in the dataset. The code was created based on: https://github.com/CMU-CREATE-Lab/deep-smoke-machine/blob/master/back-end/www/download_videos.py
"""

import sys
import urllib.request
import os
import json


def is_file_here(file_path):
    """
    Check if a file exists.

    Parameters
    ----------
    file_path : str
        The file path that we want to check.

    Returns
    -------
    bool
        If the file exists (True) or not (False).
    """
    return os.path.isfile(file_path)


def check_and_create_dir(dir_path):
    """
    Check and create a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        The dictionary path that we want to create.
    """
    if dir_path is None: return
    dir_name = os.path.dirname(dir_path)
    if dir_name != "" and not os.path.exists(dir_name):
        try: # This is used to prevent race conditions during parallel computing
            os.makedirs(dir_name)
        except Exception as ex:
            print(ex)


def get_video_url(v):
    """
    Get the video URL.

    Parameters
    ----------
    v : dict
        The dictionary with keys and values in the video dataset JSON file.

    Returns
    -------
    str
        The full URL of the video.
    """
    camera_names = ["clairton1", "braddock1", "westmifflin1"] 
    return v["url_root"] + camera_names[v["camera_id"]] + "/" +  v["url_part"] + "/" + v["file_name"] + ".mp4"


'''def main(argv):
    # Specify the path to the JSON file
    json_file_path = "/Users/szaboreka/Documents/UvA/Thesis/uva_thesis_project/data/metadata_ijmond_jan_22_2024.json"

    # Specify the path that we want to store the videos and create it
    download_path = "/Users/szaboreka/Documents/UvA/Thesis/uva_thesis_project/data/ijmond_videos/"
    check_and_create_dir(download_path)

    # Open the file and load its contents into a dictionary
    with open(json_file_path, "r") as json_file:
        data_dict = json.load(json_file)'''

# Download all videos in the metadata json file
def main(argv):
    #json_file_path = "data/metadata_02242020.json"
    json_file_path = "data/datasets/metadata_02242020.json"
    download_path = "/projects/0/prjs0930/data/merged_videos/"
    check_and_create_dir(download_path)
    problem_video_ids = []
    # Open the file and load its contents into a dictionary
    with open(json_file_path, "r") as json_file:
        data_dict = json.load(json_file)
    
    for v in data_dict:
        # Do not download videos with bad data
        if v["label_state"] == -2 or v["label_state_admin"] == -2:
            continue
        file_path = download_path + v["file_name"] + ".mp4"
        if is_file_here(file_path): continue # skip if file exists
        print("Download video", v["id"])
        try:
            urllib.request.urlretrieve(v["url_root"] + v["url_part"], file_path)
        except:
            print("\tError downloading video", v["id"])
            problem_video_ids.append(v["id"])
    print("Done download_videos.py")
    if len(problem_video_ids) > 0:
        print("The following videos were not downloaded due to errors:")
        for i in problem_video_ids:
            print("\ti\n")
    print("DONE")


if __name__ == "__main__":
    main(sys.argv)
