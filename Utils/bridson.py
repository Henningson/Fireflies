# Code from: https://github.com/emulbreh/bridson
# Accessed 21.11.2023 11:05


import random
from math import cos, sin, floor, sqrt, pi, ceil
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List


def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random):
    tau = 2 * pi
    cellsize = r / sqrt(2)

    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= r:
                    return False
        return True

    p = width * random.random(), height * random.random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random.random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random.random()
            d = r * sqrt(3 * random.random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    return [p for p in grid if p is not None]


def scaled_poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random):
    tau = 2 * pi
    cellsize = r / sqrt(2)

    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    return [p for p in grid if p is not None]



class PoissonDiskSampling:
    @staticmethod
    def GeneratePoints(radius: float, sampleRegionSize: Tuple[float], numSamplesBeforeRejection: int = 30):
        cellSize = radius / sqrt(2.0)

        grid = np.zeros((int(ceil(sampleRegionSize[0] / cellSize)), int(ceil(sampleRegionSize[1] / cellSize))), dtype=int)
        points = []
        spawnPoints = []

        spawnPoints.append(np.array([sampleRegionSize[0]/2, sampleRegionSize[1]/2]))

        while len(spawnPoints) > 0:
            spawnIndex = random.randint(0, len(spawnPoints) - 1)
            spawnCenter = spawnPoints[spawnIndex]

            candidateAccepted = False
            for _ in range(numSamplesBeforeRejection):
                angle = random.uniform(0, 1) * math.pi * 2
                direction = np.array([sin(angle), cos(angle)])
                candidate = spawnCenter + random.uniform(radius, 2*radius) * direction
                
                if PoissonDiskSampling.isValid(candidate, sampleRegionSize, cellSize, radius, points, grid):
                    points.append(candidate)
                    spawnPoints.append(candidate)
                    grid[int(candidate[0] / cellSize), int(candidate[1] / cellSize)] = len(points)
                    candidateAccepted = True
                    break

            if not candidateAccepted:
                del spawnPoints[spawnIndex]
                
        return points, grid


    @staticmethod
    def GeneratePoints(min_radius: float, max_radius: float, sampling_texture: np.array, sampleRegionSize: np.array, numSamplesBeforeRejection: int = 30):
        cellSize = min_radius / sqrt(2.0)

        grid = np.zeros((int(ceil(sampleRegionSize[0] / cellSize)), int(ceil(sampleRegionSize[1] / cellSize))), dtype=int)
        points = []
        spawnPoints = []

        spawnPoints.append(np.array([sampleRegionSize[0]/2, sampleRegionSize[1]/2]))

        while len(spawnPoints) > 0:
            spawnIndex = random.randint(0, len(spawnPoints) - 1)
            spawnCenter = spawnPoints[spawnIndex]

            candidateAccepted = False
            for _ in range(numSamplesBeforeRejection):
                angle = random.uniform(0, 1) * math.pi * 2
                direction = np.array([sin(angle), cos(angle)])
                candidate = spawnCenter + random.uniform(radius, 2*radius) * direction
                
                if PoissonDiskSampling.isValid(candidate, sampleRegionSize, cellSize, radius, points, grid):
                    points.append(candidate)
                    spawnPoints.append(candidate)
                    grid[int(candidate[0] / cellSize), int(candidate[1] / cellSize)] = len(points)
                    candidateAccepted = True
                    break

            if not candidateAccepted:
                del spawnPoints[spawnIndex]
                
        return points, grid


    @staticmethod
    def isValid(candidate: Tuple, sampleRegionSize: Tuple, cellSize: float, radius: float, points: List, grid:np.array) -> bool:
        if floor(candidate[0]) < 0 or ceil(candidate[0]) >= sampleRegionSize[0]:
            return False
        
        if floor(candidate[1]) < 0 or ceil(candidate[1]) >= sampleRegionSize[1]:
            return False
        
        cellX = int(candidate[0] / cellSize)
        cellY = int(candidate[1] / cellSize)

        searchStartX = max(0, cellX - 2)
        searchEndX = min(cellX + 3, grid.shape[0])

        searchStartY = max(0, cellY - 2)
        searchEndY = min(cellY + 3, grid.shape[1])

        for x in range(searchStartX, searchEndX):
            for y in range(searchStartY, searchEndY):
                pointIndex = grid[x, y] - 1

                if pointIndex != -1:
                    dst = np.linalg.norm(candidate - points[pointIndex])

                    if dst < radius:
                        return False

        return True
    
if __name__ == "__main__":

    width = 512
    height = 512
    radius = 25

    points, grid = PoissonDiskSampling.GeneratePoints(radius, [width, height], numSamplesBeforeRejection = 5)
    points = np.stack(points)
    #points = np.array(poisson_disc_samples(width=width, height=height, r=radius))

    plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()
    