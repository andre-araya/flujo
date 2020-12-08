import logging
import logging.handlers
import sys
import cv2
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tracker import VehicleCounter


# ============================================================================
# WARNING!!! Arbitrary definitions below! 
# Work Area delimiters

#point1 = (0,472)
#point2 = (961, 472)tils

#point3 = (1123,1080)


point1 = (900,115) 
point2 = (1000,110)
point3 = (1050,460)  
point4 = (1370,430)
pointA = (229,218)
pointB = (433,225)


m_line = (point4[0]-point3[0])/(point4[1]-point3[1])
b_line = point4[0]-m_line*point4[1]


# ============================================================================

def init_logging():
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if True:
        handler_file = logging.handlers.RotatingFileHandler("../results/debug.log"
            , maxBytes = 2**24
            , backupCount = 10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger

# ============================================================================

def gamma_correction(image, gamma = 1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 25 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# ============================================================================
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

# ============================================================================

def save_frame(file_name_format, frame_number, frame, label_format):
	log = logging.getLogger("save_frame")
	file_name = file_name_format % frame_number
	label = label_format % frame_number
	cv2.imwrite(file_name, frame)

# ============================================================================
#Filters frame in high contrast

def filter_mask(fg_mask):
    opening, closing, dilatation = None, None, None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(3):
    # Remove noise
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations = 4)
    # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations = 15)
    # Fill any small holes
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 12)

    return closing

# ============================================================================

def detect_vehicles(fg_mask):
    log = logging.getLogger("detect_vehicles")

    MIN_CONTOUR_WIDTH = 50
    MIN_CONTOUR_HEIGHT = 50

    # Finding the contours of any vehicles in the image
    contours, hierarchy = cv2.findContours(fg_mask #modified _,
        , cv2.RETR_EXTERNAL
        , cv2.CHAIN_APPROX_SIMPLE)

    # Filtering of contours
    matches = []
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        #x_max = m_line*y + b_line # Trapezoid edge
        condition1 = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)
        condition2 = y + h/2 > point1[1] and y + h/2 < point4[1] and x > point1[0] and x < point4[0]
        contour_valid = condition1 and condition2

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid))

    return matches

# ============================================================================


def process_frame(frame_number, frame_counter, frame, bg_subtractor, car_counter, 
                 counter_line_Y, DIVIDER_COLOUR, BOUNDING_BOX_COLOUR, 
                 CENTROID_COLOUR, IMAGE_DIR):

    log = logging.getLogger("process_frame")


    # Creating a copy of source frame to draw into
    processed = frame.copy()

    
    # Drawing dividing lines -- Put a new one to
    # Use it in flow estimation!!.
    cv2.line(processed, point1, point2, DIVIDER_COLOUR, 4)
    cv2.line(processed, point3, point4, DIVIDER_COLOUR, 4)
    cv2.line(processed, point1, point3, DIVIDER_COLOUR, 4)
    cv2.line(processed, point2, point4, DIVIDER_COLOUR, 4)

####### Not implemented counting 
    #cv2.line(processed, pointA, pointB, CENTROID_COLOUR, 4)
######

    # Removing the background
    fg_mask = bg_subtractor.apply(frame, None, 0.01)
    fg_mask = filter_mask(fg_mask)


    # Saving filtered frames
    #if frame_number > 100*frame_counter and frame_number < 100*(frame_counter+1):
     #   save_frame(IMAGE_DIR + "/mask_%04d.png"
     #   , frame_number, fg_mask, "foreground mask for frame #%d")


   # Detecting Vehicle contours and  centroids
    matches = detect_vehicles(fg_mask)

   
    # Marking the bounding box and the centroid on the processed frame 
    for (i, match) in enumerate(matches):
        contour, centroid = match
        x, y, w, h = contour 

        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1),
        BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)


    #Updating vehicle count
    car_counter.update_count(matches, frame_number, processed)

    return processed

# ============================================================================

def plotter(path_ways):

    plt.ylabel('depth (m) position')
    plt.xlabel('Time (s)')
    for path in path_ways:
        y = path.z_depths
        x = path.times

        plt.plot(x, y, label='path id: {id}'.format(id = path.id))
        plt.scatter(x, y)
        

    plt.show()

# ============================================================================

def path_writer(path_ways):
    
    address = '../results/path_ways.txt'
    file = open(address, 'a')

    for path in path_ways:
        string_0 = ""
        i = 0
        for depth in path.z_depths:
            
            if i < len(path.z_depths)-1: suffix = ','
            else: suffix = '\n'
            
            string_0 += str(depth) + suffix 
            i += 1

        file.write(string_0)
       
        string_1 = ""
        j = 0
        for time in path.times:
            
            if j < len(path.times)-1: suffix = ','
            else: suffix = '\n'
            
            string_1 += str(time) + suffix 
            j += 1

        file.write(string_1)

        
    file.close()

# ============================================================================

def total_frame_number_writer(frame_number):
    
    address = '../results/num_frames.txt'
    file = open(address, 'w')
    file.write(str(frame_number))    
    file.close()
