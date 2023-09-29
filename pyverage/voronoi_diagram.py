from .robot import Robot
import numpy as np
import scipy as sp
from matplotlib.path import Path

class Voronoi:
    def __init__(self, bounding_box):
        self.bounding_box = bounding_box
        self.robots = np.array([]) # This will be a list of Robot instances
        self.voronoi_data = []

    def __check_robot(self, robot):
        if not isinstance(robot, Robot):
            raise ValueError("{} must be a Robot instance.".format(robot))
        if not robot in self.robots:
            raise ValueError("{} is not in the Voronoi diagram.".format(robot))
        
    def add_robot(self, robot):
        if not isinstance(robot, Robot):
            raise ValueError("{} must be a Robot instance.".format(robot))
        self.robots = np.append(self.robots, robot)

    def remove_robot(self, robot):
        self.__check_robot(robot)
        self.robots = np.delete(self.robots, np.where(self.robots == robot))
    
    def __distance_to(self, robot, other_robot):
        self.__check_robot(robot)
        return ((robot.position[0] - other_robot.position[0]) ** 2 + (robot.position[1] - other_robot.position[1]) ** 2) ** 0.5

    def __sense_neighbors(self):
        if len(self.robots) == 0:
            raise ValueError("There are no robots in the Voronoi diagram.")
        elif len(self.robots) == 1:
            self.robots[0].neighbors = []
            return
        for robot in self.robots:
            # Get the robots that are within the sensing range excluding the robot itself
            robot.neighbors = [other_robot for other_robot in self.robots if self.__distance_to(robot, other_robot) <= robot.sensing_range and robot != other_robot]
    
    def __compute_perimeter(self, robot):
        self.__check_robot(robot)
        # Half of the sensing range to guarantee the continuity of the computation centroid
        sens_range = robot.sensing_range * 0.5
        # Discritize the bounding box always to 20 points
        delta_x = self.bounding_box.x_max / 100
        delta_y = self.bounding_box.y_max / 100
        xpts = np.arange(self.bounding_box.x_min, self.bounding_box.x_max + delta_x, delta_x)
        ypts = np.arange(self.bounding_box.y_min, self.bounding_box.y_max + delta_y, delta_y)
        bottom_points = np.array([[x, self.bounding_box.y_min] for x in xpts])
        top_points = np.array([[x, self.bounding_box.y_max] for x in xpts])
        left_points = np.array([[self.bounding_box.x_min, y] for y in ypts])
        right_points = np.array([[self.bounding_box.x_max, y] for y in ypts])

        # Discretize the sensing range
        theta = np.linspace(0, 2 * np.pi, 9)
        x = robot.position[0] + sens_range * np.cos(theta)
        y = robot.position[1] + sens_range * np.sin(theta)
        # Check if the points are inside the bounding box
        mask = (self.bounding_box.x_min <= x) & (x <= self.bounding_box.x_max) & (self.bounding_box.y_min <= y) & (y <= self.bounding_box.y_max)
        circle_points = np.array([x[mask], y[mask]]).T

        # Check if the points of the box are inside the circle
        dist_bottom = np.linalg.norm(bottom_points - robot.position, axis=1)
        dist_top = np.linalg.norm(top_points - robot.position, axis=1)
        dist_left = np.linalg.norm(left_points - robot.position, axis=1)
        dist_right = np.linalg.norm(right_points - robot.position, axis=1)
        # Mask them to get the points inside the circle
        mask_bottom = bottom_points[dist_bottom <= sens_range]
        mask_top = top_points[dist_top <= sens_range]
        mask_left = left_points[dist_left <= sens_range]
        mask_right = right_points[dist_right <= sens_range]
        
        # Adapt the points of the perimeter to the bounding box
        points = np.concatenate((circle_points, mask_top, mask_bottom, mask_left, mask_right), axis=0)
        # Calculate the centroid of the points
        centroid = np.mean(points, axis=0)
        # Calculate the angles of the points with respect to the centroid
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        # Sort the points based on the angles
        sorted_indices = np.argsort(angles)
        # Create a new array of points in clockwise proximity order
        sorted_points = points[sorted_indices]
        sorted_points = np.append(sorted_points, [sorted_points[0]], axis=0)

        return sorted_points
    
    def __local_voronoi(self, robot, mean, cov):
        """
        Bounded Voronoi Computation
        """
        # Setups
        self.__check_robot(robot) # To check if the robot is valid
        self.__sense_neighbors() # To sense the neighbors of the robot
        perimeter = self.__compute_perimeter(robot) # To compute the perimeter of the robot and adapt to the bounding box
        x_min, x_max = self.bounding_box.x_min, self.bounding_box.x_max
        y_min, y_max = self.bounding_box.y_min, self.bounding_box.y_max
        # Mirror points
        points_center = np.array([robot.position] + [neighbor.position for neighbor in robot.neighbors])
        points_left = np.copy(points_center)
        points_left[:, 0] = x_min - (points_left[:, 0] - x_min)
        points_right = np.copy(points_center)
        points_right[:, 0] = x_max + (x_max - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = y_min - (points_down[:, 1] - y_min)
        points_up = np.copy(points_center)
        points_up[:, 1] = y_max + (y_max - points_up[:, 1])
        points = np.vstack((points_center, points_left, points_right, points_down, points_up))
        # Compute Voronoi
        vor = sp.spatial.Voronoi(points)
        # Filter regions
        vor.filtered_points = points_center
        vor.filtered_regions = np.array(vor.regions)[vor.point_region[:vor.npoints//5]]
        # Cut the Voronoi cells to the perimeter of the robot
        points = np.array([[]])
        intersection_points = np.empty((0, 2))
        points_in_region = np.empty((0, 2))
        # Get the vertices of the cell with the robot
        vertices = vor.vertices[vor.filtered_regions[0] + [vor.filtered_regions[0][0]], :]
        # Find the intersection points between the perimeter and the voronoi edges
        for i in range(len(perimeter) - 1):
            p1 = perimeter[i]
            p2 = perimeter[i + 1]
            for j in range(len(vertices) - 1):
                p3 = vertices[j]
                p4 = vertices[j + 1]
                denominator = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
                numerator1 = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])
                numerator2 = (p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])
                if denominator != 0:
                    ua = numerator1 / denominator
                    ub = numerator2 / denominator
                    if 0 <= ua <= 1 and 0 <= ub <= 1:
                        intersection_point_x = p1[0] + ua * (p2[0] - p1[0])
                        intersection_point_y = p1[1] + ua * (p2[1] - p1[1])
                        intersection_points = np.append(intersection_points, np.array([[intersection_point_x, intersection_point_y]]), axis=0)
        # Find the points in the region
        points_in_region = np.array([point for point in perimeter if self.__point_in_poly(point, vertices)])
        # Find the vertices in the perimeter
        vertices_in_perimeter = np.atleast_2d(np.array([vertex for vertex in vertices if self.__point_in_poly(vertex, perimeter)])).reshape(-1, 2)
        points = np.concatenate((intersection_points, points_in_region, vertices_in_perimeter), axis=0)
        # Calculate the centroid of the points
        centroid = np.mean(points, axis=0)
        # Calculate the angles of the points with respect to the centroid
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        # Sort the points based on the angles
        sorted_indices = np.argsort(angles)
        # Create a new array of points in clockwise proximity order
        sorted_points = points[sorted_indices]
        # Append a point to close the polygon
        sorted_points = np.append(sorted_points, [sorted_points[0]], axis=0)
        # Remove the duplicate points
        unique_data = [sorted_points[0]]
        for i in range(1, len(sorted_points)):
            if not np.all(sorted_points[i] == sorted_points[i - 1]):
                unique_data.append(sorted_points[i])
        voronoi = np.array(unique_data)
        # Compute the centorid of the voronoi cell
        if not mean is None and not cov is None:
            centroid = self.compute_centroid_custom(voronoi, mean, cov)
            self.voronoi_data.append({"robot": robot, "voronoi": voronoi, "centroid": centroid})
        else:
            self.voronoi_data.append({"robot": robot, "voronoi": voronoi})

    def diagram(self, mean=None, cov=None):
        """
        Explanation of the algorithm:
        Compute the local voronoi diagram for each robot given the mean and covariance of the gaussian distribution of
        the interest point.
        """
        self.voronoi_data = []
        for robot in self.robots:
            self.__local_voronoi(robot, mean, cov)
        return self.voronoi_data

    def __point_in_poly(self, point, vertices):
        # Check if a point is inside a polygon
        # Code from https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
        # Create a point for line segment from p to infinite
        extreme = [100000, point[1]]
        count = 0
        i = 0
        n = len(vertices)
        while True:
            next = (i + 1) % n
            # Check if the line segment from 'p' to 'extreme' intersects with the line segment from 'vertices[i]' to 'vertices[next]'
            if self.__do_intersect(vertices[i], vertices[next], point, extreme):
                # If the point 'p' is colinear with line segment 'i-next', then check if it lies on segment. If it lies, return true, otherwise false
                if self.__orientation(vertices[i], point, vertices[next]) == 0:
                    return self.__on_segment(vertices[i], point, vertices[next])
                count += 1
            i = next
            if i == 0:
                break
        return count % 2 == 1
    
    def __orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2
        
    def __do_intersect(self, p1, q1, p2, q2):
        # Check if the line segment 'p1-q1' and 'p2-q2' intersect
        o1 = self.__orientation(p1, q1, p2)
        o2 = self.__orientation(p1, q1, q2)
        o3 = self.__orientation(p2, q2, p1)
        o4 = self.__orientation(p2, q2, q1)
        # General case
        if o1 != o2 and o3 != o4:
            return True
        # Special cases
        # p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if o1 == 0 and self.__on_segment(p1, p2, q1):
            return True
        # p1, q1 and q2 are colinear and q2 lies on segment p1q1
        if o2 == 0 and self.__on_segment(p1, q2, q1):
            return True
        # p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if o3 == 0 and self.__on_segment(p2, p1, q2):
            return True
        # p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if o4 == 0 and self.__on_segment(p2, q1, q2):
            return True
        return False
    
    def __on_segment(self, p, q, r):
        # Check if point q lies on line segment 'pr'
        # Code from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def gauss_pdf_array(self, x, y, sigma, mean):
        xt = mean[0]
        yt = mean[1]
        val = np.zeros(len(x))

        tmp = ((x - xt)**2 + (y - yt)**2) / (2 * sigma**2)
        val = np.exp(-tmp)  
        return val

    def compute_centroid_custom(self, vertices, mean, covariance, discretize_int=100):
        x_inf, y_inf = np.min(vertices, axis=0)
        x_sup, y_sup = np.max(vertices, axis=0)

        t_discretize = 1.0 / discretize_int

        dx = (x_sup - x_inf) / 2.0 * t_discretize
        dy = (y_sup - y_inf) / 2.0 * t_discretize
        dA = dx * dy

        A = 0
        Cx = 0
        Cy = 0

        p = Path(vertices)
        x, y = np.mgrid[x_inf:x_sup:dx, y_inf:y_sup:dy]
        points = np.c_[x.ravel(), y.ravel()]

        bool_val = p.contains_points(points)

        weight = self.gauss_pdf_array(points[:, 0], points[:, 1], covariance, mean)
        weight = weight[bool_val]

        A = np.sum(weight) * dA   
        Cx = np.sum(points[:, 0][bool_val] * weight) * dA / A
        Cy = np.sum(points[:, 1][bool_val] * weight) * dA / A

        return np.array([[Cx, Cy]])