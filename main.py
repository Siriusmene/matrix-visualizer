# %%

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform

# %%


def scaleMat(scale):
    return np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])


def translationMat(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])


def transformMat(matrix):
    M = np.eye(3)
    M[:2, :2] = matrix
    return M


def transformImage(image, matrix):
    return transform.warp(image, np.linalg.inv(matrix))


def plotTransformations(tform_matricies, image=data.camera()):
    """plot at least 2 transformations"""
    n_tforms = len(tform_matricies)
    mid_x = image.shape[1] / 2
    mid_y = image.shape[0] / 2

    fig, ax = plt.subplots(ncols=min(n_tforms, 2), nrows=n_tforms // 2 + 1)
    ax = np.atleast_2d(ax)
    transTensor = np.stack(
        [tm @ np.array([[1, 1], [-1, 1]]) for tm in tform_matricies], axis=2
    )
    maxim = np.max(np.abs(transTensor))
    scale = 1 / maxim

    transMats = [
        translationMat(mid_x, mid_y)
        @ transformMat(tm)
        @ scaleMat(scale)
        @ translationMat(-mid_x, -mid_y)
        for tm in tform_matricies
    ]
    transformed = [transformImage(image, tm) for tm in transMats]
    if n_tforms % 2 == 1:
        ax[-1, -1].axis("off")

    for i in range(n_tforms):
        a = ax[i // 2][i % 2]
        a.imshow(transformed[i], cmap=plt.cm.gray)
        a.axis("off")
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
sheer = np.array([[1, -1 / 2], [0, 1]])
plotTransformations([sheer])

# %%
