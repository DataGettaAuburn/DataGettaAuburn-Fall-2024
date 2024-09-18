import matplotlib.pyplot as plt
import numpy as np


def heatmap(values: list[tuple[float, float, float]],
            res_x: int,
            res_y: int,
            spr: float = 0.5,
            range_x: tuple = None,
            range_y: tuple = None, 
            norm: bool = False
            ) -> np.ndarray:
    """
    Creates a heatmap from a list of values.

    values: 
        list of values, where each value is a tuple (x, y, w), where x and y
        are the coordinates of the value and w is the weight of the value.
    res_x: 
        resolution of the x axis
    res_y: 
        resolution of the y axis
    spr: 
        the amount of spread to the neighbouring pixels. if not specified, it is set to 0.5.
    range_x: 
        the range of the x inputs (min, max) to include in the heatmap.
        If the x value is outside of this range, it is ignored. If not specified,
        the range is set to the min and max of the data
    range_y: 
        the range of the y inputs (min, max) to include in the heatmap.
        If the y value is outside of this range, it is ignored. If not specified,
        the range is set to the min and max of the data
    norm: 
        whether or not to normalize the heatmap

    return: a res_x by res_y np.ndarray representing the heatmap
    """

    # if the range is not specified, it is set to the min and max of the data
    if range_x is None:
        range_x = (min(values, key=lambda x: x[0])[0], max(values, key=lambda x: x[0])[0])
    if range_y is None:
        range_y = (min(values, key=lambda x: x[1])[1], max(values, key=lambda x: x[1])[1])
    
    heatmap = np.zeros((res_y, res_x))
    scale_x = res_x / (range_x[1] - range_x[0])
    scale_y = res_y / (range_y[1] - range_y[0])
    for x, y, w in values:
        if range_x[0] <= x <= range_x[1] and range_y[0] <= y <= range_y[1]:
            heatmap[int((y - range_y[0]) * scale_y)][int((x - range_x[0]) * scale_x)] += w
    
    # blurs the heatmap using a 5x5 gaussian kernel
    kernel = np.array([spr/6, spr/3, 1-spr, spr/3, spr/6])
    heatmap = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, heatmap)
    heatmap = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, heatmap)

    if norm:
        normalize(heatmap)

    return heatmap

def normalize(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalizes a heatmap.

    heatmap: 
        the heatmap to normalize

    return: a np.ndarray representing the normalized heatmap
    """
    max_val = np.max(heatmap)
    if max_val == 0:
        return heatmap
    return heatmap / max_val

def div_heatmap(h1: np.ndarray, 
                h2: np.ndarray
                ) -> np.ndarray:
    """
    Divides two heatmaps.

    h1: 
        the numerator heatmap
    h2: 
        the denominator heatmap

    return: a 2D list of values representing the quotient heatmap
    """
    if h1.shape != h2.shape:
        raise ValueError("Heatmaps must be the same shape")
    return h1 / (h2 + 1e-9)

def save_to_image(heatmap: np.ndarray, 
                  filename: str, 
                  fig_title: str, 
                  ext: tuple[float, float, float, float],
                  cmap: str = 'Reds'
                  ) -> None:
    """
    Saves a heatmap to an image file.

    heatmap: 
        the heatmap to save
    filename: 
        the name of the file to save the heatmap to
    fig_title: 
        the title of the heatmap
    ext: 
        the extent of the heatmap. This should be a tuple (x_min, x_max, y_min, y_max)
    cmap: 
        the color map to use. possible colors: 
        https://matplotlib.org/stable/users/explain/colors/colormaps.html

    return: None
    """

    plt.imshow(heatmap, cmap=cmap, interpolation='nearest', origin='lower', extent=ext)
    plt.title(fig_title)
    plt.savefig(filename + '.png', bbox_inches='tight')

