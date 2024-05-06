# %%

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform

# %%

image = data.camera()
height = image.shape[0]
width = image.shape[1]


def transFormedImage(image, matrix):
    height = image.shape[0]
    width = image.shape[1]
    M = np.eye(3)
    M[:2, :2] = matrix
    center = (width / 2, height / 2)
    translation_matrix = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    translation_inverse = np.linalg.inv(translation_matrix)
    total = translation_inverse @ M @ translation_matrix
    newImage = transform.warp(image, transform.AffineTransform(np.linalg.inv(total)))
    return newImage


def plotTransformations(tform_matricies, image=data.camera()):
    n_tforms = len(tform_matricies)
    fig, ax = plt.subplots(ncols=n_tforms)
    scaled = transFormedImage(image, np.array([[2 / 3, 0], [0, 2 / 3]]))
    for i in range(n_tforms):
        ax[i].imshow(transFormedImage(scaled, tform_matricies[i]), cmap=plt.cm.gray)
    for a in ax:
        a.axis("off")
        a.set_xlim(0, width)
        a.set_ylim(width, 0)
        a.set_aspect("equal")
    plt.style.use("dark_background")
    plt.tight_layout()
    plt.show()


# %%
def rotate(angle):
    return np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [
                math.sin(angle),
                math.cos(angle),
            ],
        ]
    )


id = np.array([[1, 0], [0, 1]])
alpha = math.pi / 4
rot = rotate(alpha)
sheer = np.array([[1, -0.5], [0, 1]])
plotTransformations([sheer, sheer.T])

# %%
