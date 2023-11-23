from .bounding_box import BoundingBox
import numpy as np

class Robot:
    def __init__(self, x, y, sensing_range=None, bounding_box=None):
        self.sensing_range = sensing_range
        if self.sensing_range is None:
            raise ValueError("Robot must have a sensing range.")
        
        self.bounding_box = bounding_box
        if self.bounding_box is not None:
            if not isinstance(self.bounding_box, BoundingBox):
                raise ValueError("Bounding box must be a BoundingBox instance.")
            if not self.bounding_box.is_inside(x, y):
                raise ValueError("Robot's position is outside the bounding box.")
            
        if self.bounding_box:
            self.__set_position(x, y)
        self.neighbors = [] # List of neighbors
        self.collection = np.empty((0, 3)) # [x, y, value]

    def __set_position(self, x, y):
        if self.bounding_box.is_inside(x, y):
            self.position = np.array([x, y])
        else:
            raise ValueError("Robot's position is outside the bounding box.")
        
    def move(self, new_x, new_y):
        if self.bounding_box.is_inside(new_x, new_y):
            self.position = np.array([new_x, new_y])
        else:
            raise ValueError("Robot's movement would place it outside the bounding box.")