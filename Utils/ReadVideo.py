# +
import numpy as np
import cv2


def convert_video_to_images(video_file_path, output_directory):
    vid_cap = cv2.VideoCapture(video_file_path)
    success, image = vid_cap.read()
    count = 0
    while success:
        cv2.imwrite(output_directory+("/frame%d.jpg" % count), image)  # save frame as JPEG file
        success, image = vid_cap.read()
        print('Read a new frame: ', success)
        count += 1
        if not success or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('frame', image)

    cv2.destroyAllWindows()
# -






