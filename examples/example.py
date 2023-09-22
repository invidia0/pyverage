import sys
sys.path.append('../pyverage/')
from pyverage.bounding_box import BoundingBox
from pyverage.robot import Robot
from pyverage.voronoi_diagram import VoronoiDiagram
import matplotlib.pyplot as plt
import numpy as np

mean = (15, 15)
covariance = 2

bbox = BoundingBox(0, 0, 20, 20)
voronoi = VoronoiDiagram(bbox)
# Generate random agents with sensing range of 
agents = np.array([[1, 1, 2], [5, 5, 2], [2, 1, 2], [19, 19, 2]])

# 3 Different figures for each robot
plt.figure(figsize=(10, 10))

for agent in agents:
    voronoi.add_robot(Robot(agent[0], agent[1], agent[2], bbox))

for k in range(100):
    plt.clf()

    vor = voronoi.voronoi(mean, covariance)

    for i in range(len(voronoi.robots)):
        robot = voronoi.robots[i]
        plt.plot(robot.position[0], robot.position[1], 'ro')

        # Plot the sensing range of the robots
        circle = plt.Circle((robot.position[0], robot.position[1]), robot.sensing_range, color='r', alpha=0.1, fill=False)
        plt.gcf().gca().add_artist(circle)

        # For each robot plot his voronoi
        plt.plot(vor[i]["voronoi"][:, 0], vor[i]["voronoi"][:, 1], 'k-', alpha=0.1)
        # Plot the centroid of the diagram
        centroid = vor[i]["centroid"]
        plt.plot(centroid[0][0], centroid[0][1], 'bo')
        robot.move(centroid[0][0], centroid[0][1])

    plt.xlim(bbox.x_min - 1, bbox.x_max + 1)
    plt.ylim(bbox.y_min - 1, bbox.y_max + 1)
    plt.plot([bbox.x_min, bbox.x_min, bbox.x_max, bbox.x_max, bbox.x_min], [bbox.y_min, bbox.y_max, bbox.y_max, bbox.y_min, bbox.y_min], 'k-', alpha=0.2)
    plt.grid(True, alpha=0.2)
    plt.pause(0.001)
plt.show()