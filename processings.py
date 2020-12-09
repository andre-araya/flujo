import logging
import logging.handlers
import numpy as np
import os
import gc
import time
import sys
import cv2
from utils import init_logging, get_centroid, save_frame, path_writer
from utils import filter_mask, detect_vehicles, process_frame, plotter
from utils import total_frame_number_writer
from tracker import VehicleCounter, PathWay

# ============================================================================

# O/I Videos
INPUT_VIDEO_NAME = "../video/output.ogv" #Sample_Videos/pretil.mp4" # Video file
OUTPUT_VIDEO_NAME = "../results/Output.avi"

#Video Capturing
cap = cv2.VideoCapture(INPUT_VIDEO_NAME)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = cap.get(cv2.CAP_PROP_FPS)

#print cap.get(cv2.CAP_PROP_FRAME_COUNT)
#Car-free image for background subtractor Pre-training 
CAR_FREE_IMAGE = "../video/limpio01.png" #modified from left_lane_#

IMAGE_DIR = "../results/images_0.01_nf"
IMAGE_FILENAME_FORMAT = IMAGE_DIR + "/frame_%04d.png"

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(OUTPUT_VIDEO_NAME,fourcc, 16.0, (int(frame_width),int(frame_height)))

# Time to wait between frames, 0=forever
WAIT_TIME = 1 # 250 # ms

# COLORS
DIVIDER_COLOUR = (0, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)

#Y position of counting line
counter_line_Y = 628

# ============================================================================

def main():
    log = logging.getLogger("main")

    # Initializing vehicle counter Class
    frame_shape = [frame_height, frame_width]
    car_counter = VehicleCounter(frame_shape, counter_line_Y)


    #Creating background subtractor...
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    

    #WARNING Pre-training the background subtractor...
    default_bg = cv2.imread(CAR_FREE_IMAGE)
    bg_subtractor.apply(default_bg, None, 2.0)


    #Starting capture loop...
    frame_number = -1
    frame_counter = 0
    while True:
        frame_number += 1

        #Time
        time = (frame_number * 1.0/fps)/60.0 #minutes
        log.debug("Video Time: %0.001f min", time)

        #Capturing Frame
        ret, frame = cap.read()
        
        if not ret:
            log.error("Frame capture failed, stopping...")
            break
        
        # Archive raw frames from video 
        #if frame_number > 100 * frame_counter:
        #    save_frame(IMAGE_FILENAME_FORMAT
        #        , frame_number, frame, "source frame #%d")
        #    if frame_number > 100 * ( 1+ frame_counter):
        #        frame_counter += 10

        #Processing frame 
        processed = process_frame(frame_number, frame_counter, frame,
                    bg_subtractor, car_counter, counter_line_Y, 
                    DIVIDER_COLOUR, BOUNDING_BOX_COLOUR, 
                    CENTROID_COLOUR, IMAGE_DIR)
        #if frame_number == 200: break
        #Saving processed frame
        #if frame_number > 50 + 100 * frame_counter: #and frame_number < 100 * (1 + frame_counter):
        #    save_frame(IMAGE_DIR + "/processed_%04d.png"
        #    , frame_number, processed, "processed frame #%d")
        #    if frame_number > 100 * (1 + frame_counter):
        #        frame_counter += 10
        #else:
        #    save_frame(IMAGE_DIR + "/processed_%04d.png"
        #    , frame_number, processed, "processed frame #%d")
           # pass
        #Writing Output Video
        #out.write(processed)

        #Stopping if ESC is detected
        c = cv2.waitKey(WAIT_TIME)
        if c == 27:
            log.debug("ESC detected, stopping...")
            break


    #Closing video capture device...
    cap.release()
    cv2.destroyAllWindows()
    log.debug("Done.")
    #out.release()

    #POST-Processing
    total_frame_number_writer(frame_number)
    path_ways = car_counter.well_tracked_cars_pathways()
    path_writer(path_ways)
# ============================================================================

if __name__ == "__main__":
    log = init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
    print
