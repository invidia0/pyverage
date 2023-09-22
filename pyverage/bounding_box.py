class BoundingBox:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def is_inside(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max