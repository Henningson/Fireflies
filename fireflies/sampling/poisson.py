import numpy as np

"""
Implementation of the fast Poisson Disk Sampling algorithm of 
Bridson (2007) adapted to support spatially varying sampling radii. 

Adrian Bittner, 2021
Published under MIT license. 
"""


def getGridCoordinates(coords):
    return np.floor(coords).astype("int")


def bridson(radius, k=30, radiusType="default"):
    """
    Implementation of the Poisson Disk Sampling algorithm.

    :param radius: 2d array specifying the minimum sampling radius for each spatial position in the sampling box. The
                   size of the sampling box is given by the size of the radius array.
    :param k: Number of iterations to find a new particle in an annulus between radius r and 2r from a sample particle.
    :param radiusType: Method to determine the distance to newly spawned particles. 'default' follows the algorithm of
                       Bridson (2007) and generates particles uniformly in the annulus between radius r and 2r.
                       'normDist' instead creates new particles at distances drawn from a normal distribution centered
                       around 1.5r with a dispersion of 0.2r.
    :return: nParticle: Number of particles in the sampling.
             particleCoordinates: 2d array containing the coordinates of the created particles.
    """
    # Set-up background grid
    gridHeight, gridWidth = radius.shape
    grid = np.zeros((gridHeight, gridWidth))

    # Pick initial (active) point
    coords = (np.random.random() * gridHeight, np.random.random() * gridWidth)
    idx = getGridCoordinates(coords)
    nParticle = 1
    grid[idx[0], idx[1]] = nParticle

    # Initialise active queue
    queue = [
        coords
    ]  # Appending to list is much quicker than to numpy array, if you do it very often
    particleCoordinates = [
        coords
    ]  # List containing the exact positions of the final particles

    # Continue iteration while there is still points in active list
    while queue:

        # Pick random element in active queue
        idx = np.random.randint(len(queue))
        activeCoords = queue[idx]
        activeGridCoords = getGridCoordinates(activeCoords)

        success = False
        for _ in range(k):

            if radiusType == "default":
                # Pick radius for new sample particle ranging between 1 and 2 times the local radius
                newRadius = radius[activeGridCoords[0], activeGridCoords[1]] * (
                    np.random.random() + 1
                )
            elif radiusType == "normDist":
                # Pick radius for new sample particle from a normal distribution around 1.5 times the local radius
                newRadius = radius[
                    activeGridCoords[0], activeGridCoords[1]
                ] * np.random.normal(1.5, 0.2)

            # Pick the angle to the sample particle and determine its coordinates
            angle = 2 * np.pi * np.random.random()
            newCoords = np.zeros(2)
            newCoords[0] = activeCoords[0] + newRadius * np.sin(angle)
            newCoords[1] = activeCoords[1] + newRadius * np.cos(angle)

            # Prevent that the new particle is outside of the grid
            if not (0 <= newCoords[1] <= gridWidth and 0 <= newCoords[0] <= gridHeight):
                continue

            # Check that particle is not too close to other particle
            newGridCoords = getGridCoordinates((newCoords[0], newCoords[1]))

            radiusThere = np.ceil(radius[newGridCoords[0], newGridCoords[1]])

            gridRangeX = (
                np.max([newGridCoords[1] - radiusThere, 0]).astype("int"),
                np.min([newGridCoords[1] + radiusThere + 1, gridWidth]).astype("int"),
            )
            gridRangeY = (
                np.max([newGridCoords[0] - radiusThere, 0]).astype("int"),
                np.min([newGridCoords[0] + radiusThere + 1, gridHeight]).astype("int"),
            )

            searchGrid = grid[
                slice(gridRangeY[0], gridRangeY[1]), slice(gridRangeX[0], gridRangeX[1])
            ]
            conflicts = np.where(searchGrid > 0)

            if len(conflicts[0]) == 0 and len(conflicts[1]) == 0:
                # No conflicts detected. Create a new particle at this position!
                queue.append(newCoords)
                particleCoordinates.append(newCoords)
                nParticle += 1
                grid[newGridCoords[0], newGridCoords[1]] = nParticle
                success = True

            else:
                # There is a conflict. Do NOT create a new particle at this position!
                continue

        if success == False:
            # No new particle could be associated to the currently active particle.
            # Remove current particle from the active queue!
            del queue[idx]

    return (nParticle, np.array(particleCoordinates))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    width = 512
    height = 512

    radius = 10
    max_radius = 3 * radius

    import cv2

    weight_matrix = (
        np.ones([height, width], np.float32) * max_radius
    )  # Should be between 0 and 1
    cv2.circle(weight_matrix, np.array(weight_matrix.shape) // 2, 50, radius, -1)

    # cv2.imshow("Weight Matrix", weight_matrix)
    # cv2.waitKey(0)

    npoints, points = bridson(weight_matrix)
    # points = np.array(poisson_disc_samples(width=width, height=height, r=radius))

    plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()
