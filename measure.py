import re, cv2, const
import numpy as np
from pathlib import Path


def load_observations(file_dir):
    """
    According to the directory path for detections of each frame ('txt' files),
    output a list of detected bounding box positions per video frame.

    Args:
        file_dir [Str]: Directory path for 'txt' files.
                        The detections of each frame correspond to a txt file.
                        Multiple target detections in each file are written line by line.

    Returns:
        Output [List]: A list of elements. Every element in this list is also a list.
                       This list consists of all the detected bounding box positions per video frame.
                       Each BB position is described as an 1D numpy array ("xyxy").
    """

    # Path("path" [Str]): Convert an absolute path into an object (object-oriented file path system).
    # Path("path" [Str]).rglob("testvideo_*.txt"):
    #   According to the matching mode,
    #   search for all the absolute file pathes (object) with name "testvideo_*.txt" ("*" represents any value)
    #   under the current absolute path (object) 'Path("path" [Str])' and its subpathes (object),
    #   pack all the found absolute file pathes (object) and return them together into an object named 'files'.
    files = Path(file_dir).rglob("testvideo_*.txt")

    # sorted(Object): Sort an iterable object. (Here 'files' is an iterable object.)
    # The 'sorted(Object)' function provides a new sorted output
    # without changing the order of the original values.
    files = sorted(files, key=lambda x : int(re.findall("testvideo_(.*?).txt", str(x))[0]))

    # np.loadtxt('file path' [Str],
    #            'dtype' (Output data type; Ok: from int to float; Not ok: from float to int) [Int],
    #            'usecols' (Which columns to read, 0 is the first column) [List]) 
    return [list(np.loadtxt(str(file), int, usecols=[1, 2, 3, 4])) for file in files]


if __name__ == "__main__":
    cap = cv2.VideoCapture(const.VIDEO_PATH)
    mea_list = load_observations(const.FILE_DIR)

    for mea_frame_list in mea_list:
        ret, frame = cap.read()
        for mea in mea_frame_list:
            cv2.rectangle(frame, tuple(mea[:2]), tuple(mea[2:]), const.GREEN, thickness=1)

        """Refresh the window "Demo" with the current video frame and all the drawings on it,
           neglecting all the drawings on the previous video frames."""
        cv2.imshow("Demo", frame) 
        cv2.waitKey(100)  # Display the current video frame for 100 ms.

    cap.release()
    cv2.destroyAllWindows()
