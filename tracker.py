import logging
import math
import cv2
import numpy as np


# z_depth_fitting_params
height = 480
a = -2102.00383851 
b = -5.23633263
c = 51.61593937


 # ============================================================================
def calc_z_depth(positions):
    depths = []
    for position in positions:
        #Fitting made with (y_pixels)/100 as indenpendent parameter 
        y_pix = (height - position[1])/100
        depths.append(a/(y_pix - b)**2 + c)

    return depths
# ============================================================================

CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]

# ============================================================================

class PathWay():
    def __init__(self, id, positions, frame_numbers):
        self.id = id
        self.positions = positions
        self.frame_numbers = frame_numbers
        self.z_depths = np.array(calc_z_depth(positions))
        self.times = np.array(frame_numbers)*0.0625
# ============================================================================

class Vehicle(object):
    def __init__(self, id, position, frame_number):
        self.id = id
        self.positions = [position] 
        self.frame_numbers = [frame_number]
        self.frames_since_seen = 0
        self.counted = False

    @property

    def last_position(self):
        return self.positions[-1]

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0

    def add_frame_number(self, new_frame_number):
        self.frame_numbers.append(new_frame_number)

    def draw(self, output_image):
        car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)]
        for point in self.positions:
            cv2.circle(output_image, point, 2, car_colour, -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, car_colour, 1)


# ============================================================================

class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.log = logging.getLogger("vehicle_counter")
        self.height, self.width = shape  # Height and Width Of the Frame
        self.divider = divider  # Y cood of line used to count cars (not implemented yet) 
        self.vehicles = []
        self.path_ways = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.max_unseen_frames = 5
        self.well_tracked_vehicles = []

    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        

        return distance, angle 


    @staticmethod
    def is_valid_vector(a):
        distance, angle = a
        threshold_distance = max(50.0, -0.008 * angle**2 + 0.4 * angle + 25.0)
        return (distance <= threshold_distance)


    def update_vehicle(self, vehicle, matches, frame_number):
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            contour, centroid = match

            vector = self.get_vector(vehicle.last_position, centroid)
            if self.is_valid_vector(vector):
                vehicle.add_position(centroid)
                vehicle.add_frame_number(frame_number)
                self.log.debug("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)"
                    , centroid[0], centroid[1], vehicle.id, vector[0], vector[1])
                return i

        # No matches fit...        
        vehicle.frames_since_seen += 1
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d"
            , vehicle.id, vehicle.frames_since_seen)

        return None


    def update_count(self, matches, frame_number, output_image = None):
        self.log.debug("Updating count using %d matches...", len(matches))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches, frame_number)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid,
                         frame_number)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            self.log.debug("Created new vehicle #%d from match (%d, %d)."
                , new_vehicle.id, centroid[0], centroid[1])

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            if not vehicle.counted and (vehicle.last_position[1] > self.divider):
                self.vehicle_count += 1
                vehicle.counted = True
                self.log.debug("Counted vehicle #%d (total count=%d)."
                    , vehicle.id, self.vehicle_count)

        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)

            cv2.putText(output_image, ("%02d" % self.vehicle_count), (142, 10)
                , cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)


        
        # Remove vehicles that have not been seen long enough
        tracked_vehicles = []
        removed_vehicles = []
        for v in self.vehicles:
            if v.frames_since_seen >= self.max_unseen_frames:
                removed_vehicles.append(v.id)
                tracked_vehicles.append(v)

       
        self.vehicles[:] = [ v for v in self.vehicles
            if not v.frames_since_seen >= self.max_unseen_frames ]
        for id in removed_vehicles:
            self.log.debug("Removed vehicle #%d.", id)
        self.log.debug("Count updated, tracking %d vehicles.", len(self.vehicles))


    
        for vehicle in tracked_vehicles:
            if len(vehicle.positions) > 6:
                    self.well_tracked_vehicles.append(vehicle)


    def well_tracked_cars_pathways(self): 
        for vehicle in self.well_tracked_vehicles:
            new_path_way = PathWay(vehicle.id, vehicle.positions, 
                                   vehicle.frame_numbers)
            self.path_ways.append(new_path_way)

        return self.path_ways
        
# ===========================================================================
