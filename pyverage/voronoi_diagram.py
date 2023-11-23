from .robot import Robot
import numpy as np
import scipy as sp
from matplotlib.path import Path
import matplotlib.pyplot as plt
class Voronoi:
    def __init__(self, bounding_box):
        self.bounding_box = bounding_box
        self.robots = np.array([]) # This will be a list of Robot instances
        self.voronoi = []

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
        x_min, x_max = self.bounding_box.x_min, self.bounding_box.x_max
        y_min, y_max = self.bounding_box.y_min, self.bounding_box.y_max

        # Half of the sensing range to guarantee the continuity of the computation centroid
        sens_range = robot.sensing_range * 0.5
        
        # Discretize the sensing range
        theta = np.linspace(0, 2 * np.pi, 9)
        x = robot.position[0] + sens_range * np.cos(theta)
        y = robot.position[1] + sens_range * np.sin(theta)
        perimeter = np.array([x, y]).T

        points = np.empty((0, 2))
        intersection_points = np.empty((0, 2))
        # Keep only the perimiter points inside the bounding box
        for i in range(len(perimeter)):
            if x_min <= perimeter[i][0] <= x_max and y_min <= perimeter[i][1] <= y_max:
                points = np.append(points, [perimeter[i]], axis=0)
        # Find the intersection points between the perimeter and the bounding box
        for i in range(4):
            if i == 0:
                x1, y1, x2, y2 = x_min, y_min, x_max, y_min
            elif i == 1:
                x1, y1, x2, y2 = x_max, y_min, x_max, y_max
            elif i == 2:
                x1, y1, x2, y2 = x_max, y_max, x_min, y_max
            elif i == 3:
                x1, y1, x2, y2 = x_min, y_max, x_min, y_min

            for j in range(len(perimeter) - 1):
                x3, y3, x4, y4 = (
                    perimeter[j][0],
                    perimeter[j][1],
                    perimeter[(j + 1) % len(perimeter)][0],
                    perimeter[(j + 1) % len(perimeter)][1],
                )

                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator == 0:
                    continue

                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

                if 0 <= t <= 1 and 0 <= u <= 1:
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    intersection_points = np.append(intersection_points, [[x, y]], axis=0)

        intersection_points = np.array(intersection_points).reshape(-1, 2)
        points = np.append(points, intersection_points, axis=0)

        # Check if the perimeter contains the bounding box points
        for i in range(4):
            if i == 0:
                x, y = x_min, y_min
            elif i == 1:
                x, y = x_max, y_min
            elif i == 2:
                x, y = x_max, y_max
            elif i == 3:
                x, y = x_min, y_max

            if self.__is_inside_poly([x, y], perimeter):
                points = np.append(points, [[x, y]], axis=0)

        # Sort the points in proximity order
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(
            points[:, 1] - centroid[1], points[:, 0] - centroid[0]
        )
        points = points[np.argsort(angles)]
        # Close the polygon
        points = np.append(points, [points[0]], axis=0)
        # Remove the duplicates
        unique_points = [points[0]]
        for i in range(1, len(points)):
            if not np.all(points[i] == points[i - 1]):
                unique_points.append(points[i])
        perimeter = np.array(unique_points)

        return perimeter
    
    def __voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def __is_inside_poly(self, point, polygon):
        """
        Check if a point is inside a polygon.
        Parameters
        ----------
        point : list or tuple
            The point to check.
        polygon : list of lists or tuples
            A list of points defining the polygon.
        Returns
        -------
        inside : bool
            True if the point is inside the polygon, False otherwise.
        """
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def __local_voronoi(self, robot):
        """
        Bounded Voronoi Computation
        """
        # Setups
        self.__check_robot(robot) # To check if the robot is valid
        self.__sense_neighbors() # To sense the neighbors of the robot
        perimeter = self.__compute_perimeter(robot) # To compute the perimeter of the robot and adapt to the bounding box
        x_min, x_max = self.bounding_box.x_min, self.bounding_box.x_max
        y_min, y_max = self.bounding_box.y_min, self.bounding_box.y_max

        robot_positions = np.array([robot.position] + [neighbor.position for neighbor in robot.neighbors])
        
        points_left = np.copy(robot_positions)
        points_left[:, 0] = x_min - (points_left[:, 0] - x_min)
        points_right = np.copy(robot_positions)
        points_right[:, 0] = x_max + (x_max - points_right[:, 0])
        points_down = np.copy(robot_positions)
        points_down[:, 1] = y_min - (points_down[:, 1] - y_min)
        points_up = np.copy(robot_positions)
        points_up[:, 1] = y_max + (y_max - points_up[:, 1])
        points = np.vstack((robot_positions, points_left, points_right, points_down, points_up))
        # Compute Voronoi
        vor = sp.spatial.Voronoi(points)
        # Filter regions
        vor.filtered_points = robot_positions
        vor.filtered_regions = np.array(vor.regions, dtype=object)[vor.point_region[:vor.npoints//5]]
        # Cut the Voronoi cells to the perimeter of the robot
        points = np.array([[]])
        intersection_points = np.empty((0, 2))
        points_in_region = np.empty((0, 2))
        # Get the vertices of the cell with the robot
        vertices = vor.vertices[vor.filtered_regions[0] + [vor.filtered_regions[0][0]], :]

        # Cut the Voronoi cells to the perimeter of the robot
        points = np.empty((0, 2))
        intersection_points = np.empty((0, 2))
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
        
        intersection_points = np.array(intersection_points).reshape(-1, 2)
        points = np.append(points, intersection_points, axis=0)

        # Keep only the points that are inside the voronoi cell
        for i in range(len(perimeter)):
            if self.__is_inside_poly(perimeter[i], vertices):
                points = np.append(points, [perimeter[i]], axis=0)
        
        # Find the vertices in the perimeter
        vertices_in_perimeter = np.atleast_2d(np.array([vertex for vertex in vertices if self.__is_inside_poly(vertex, perimeter)])).reshape(-1, 2)
        points = np.append(points, vertices_in_perimeter, axis=0)

        # Sort the points in proximity order
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(
            points[:, 1] - centroid[1], points[:, 0] - centroid[0]
        )
        points = points[np.argsort(angles)]
        # Close the polygon
        points = np.append(points, [points[0]], axis=0)

        # Remove the duplicates
        unique_points = [points[0]]
        for i in range(1, len(points)):
            if not np.all(points[i] == points[i - 1]):
                unique_points.append(points[i])
        points = np.array(unique_points)

        self.voronoi.append(points)

        
    def diagram(self):
        """
        Explanation of the algorithm:
        Compute the local voronoi diagram for each robot given the mean and covariance of the gaussian distribution of
        the interest point.
        """
        self.voronoi = []
        for robot in self.robots:
            self.__local_voronoi(robot)
        return self.voronoi