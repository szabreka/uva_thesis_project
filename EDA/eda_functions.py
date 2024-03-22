def get_label(row):
    if row['mode'] == 'individual mode R':
        if row['label_state_admin'] == 23:
            return 'smoke'
        if row['label_state_admin'] == 16:
            return 'no smoke'
        if row['label_state_admin'] == 47:
            return 'smoke'
        if row['label_state_admin'] == 32:
            return 'no smoke'

    if row['mode'] == 'individual mode C':
        if row['label_state'] == 23:
            return 'smoke'
        if row['label_state'] == 16:
            return 'no smoke'
        if row['label_state'] == 19:
            return 'smoke'
        if row['label_state'] == 20:
            return 'no smoke'
    if row['mode'] == 'collaborative mode':
        if row['label_state_admin'] == 23:
            return 'smoke'
        if row['label_state_admin'] == 16:
            return 'no smoke'
        if row['label_state_admin'] == 47:
            return 'smoke'
        if row['label_state_admin'] == 32:
            return 'no smoke'
        else:
            if row['label_state'] == 23:
                return 'smoke'
            if row['label_state'] == 16:
                return 'no smoke'
    else:
        if row['label_state_admin'] == -2:
            return 'bad videos'
        else:
            return 'video not labeled'

def get_mode(row):
    if row['label_state'] == -1:
        return 'individual mode R'
    elif row['label_state_admin'] == -1:
        return 'individual mode C'
    elif row['label_state_admin'] == -2:
        return 'bad videos'
    else:
        return 'collaborative mode'

def get_season(month):
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'autumn'
    else:
        return 'winter'

#The code below is copied from the https://github.com/MultiX-Amsterdam/ijmond-camera-monitor/tree/92e1b446c4c0fb5f36cfe487e645e5065c9800aa/dataset/2024-01-22#ijmond-video-dataset-2024-01-22 repository README file to achive the URLs of the IJMond dataset
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
    camera_names = ["hoogovens", "kooksfabriek_1", "kooksfabriek_2"]
    return v["url_root"] + camera_names[v["camera_id"]] + "/" +  v["url_part"] + "/" + v["file_name"] + ".mp4"

def get_video_panorama_url(v):
    """
    Get the video panorama URL.

    Parameters
    ----------
    v : dict
        The dictionary with keys and values in the video dataset JSON file.

    Returns
    -------
    str
        The full URL of the panorama video.
    """
    return "https://www.youtube.com/watch?v=" + v["url_part"]
#%%
