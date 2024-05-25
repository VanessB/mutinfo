import numpy

from ..base import CorrelatedNormal
from .. import mapped


def uniform_to_segment(x: numpy.array, min_length: float) -> numpy.array:
    """
    Map a 2D uniformly distributed random vector to a random segment.
    The coordinates of the ends are distributed uniformly, preserving order.
    
    Parameters
    ----------
    x : numpy.array
        Samples from an uniform distribution on [0; 1]^2, shape: (?,2).
    min_length : float, optional
        Minimum length of the segment (must be in [0; 1)).

    Returns
    -------
    coords : numpy.array
        Coordinates of the ends of sampled random segments.
    """
        
    if len(x.shape) != 2 or x.shape[1] != 2:
        raise TypeError("Parameter `x` must have shape (?,2)")
        
    if not (0.0 <= min_length < 1.0):
        raise ValueError("Parameter `min_length` must be in [0.0, 1.0)")

    coords = numpy.empty_like(x)

    # The first number is the left end.
    # It is linearly distributed from `0.0` to `1.0 - min_length`.
    coords[:,0] = (1.0 - min_length) * (1.0 - numpy.sqrt(1.0 - x[:,0]))

    # The last number is the right end.
    # Given the left end (condition), it is uniformly distributed
    # from the left end plus `min_length` to 1.0.
    coords[:,1] = x[:,1] * (1.0 - coords[:,0] - min_length) + coords[:,0] + min_length

    return coords


def uniform_to_rectangle(x: numpy.array, min_width: float=0.0, min_height: float=0.0,
                         max_width: float=1.0, max_height: float=1.0) -> numpy.array:
    """
    Map a 4D uniformly distributed random vector to a random rectangle.
    The coordinates of the corners are distributed uniformly, preserving order.
    
    Parameters
    ----------
    x : numpy.array
        Samples from an uniform distribution on [0; 1]^4, shape: (?,4).
    min_width : float, optional
        Minimum width of the rectangle.
    min_height : float, optional
        Minimum height of the rectangle.
    max_width : float, optional
        Maximum width of the rectangle.
    max_height : float, optional
        Maximum height of the rectangle.

    Returns
    -------
    coords : numpy.array
        Coordinates of the corners of sampled random rectangles.
    """

    if len(x.shape) != 2 or x.shape[1] != 4:
        raise TypeError("Parameter `x` must have shape (?,2)")

    coords = numpy.empty_like(x)
    coords[:,0:2] = uniform_to_segment(x[:,0:2], min_width  / max_width)  * max_width
    coords[:,2:4] = uniform_to_segment(x[:,2:4], min_height / max_height) * max_height

    return coords


def draw_rectangle(coords: numpy.array, image_width: int, image_height: int) -> numpy.array:
    """
    Map coordinates of rectangles to rasterized images of rectangles.
    
    Parameters
    ----------
    coords : numpy.array
        Normalized coordinates of the corners of rectangles, shape: (?,4).
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.
    """

    if len(coords.shape) != 2 or coords.shape[1] != 4:
        raise TypeError("Parameter `coords` must have shape (?,4)")
        
    n_samples = coords.shape[0]

    # Denormalized coordinates.
    denormalized_coords = numpy.empty_like(coords)
    denormalized_coords[:,0:2] = coords[:,0:2] * image_width
    denormalized_coords[:,2:4] = coords[:,2:4] * image_height

    #integer_denormalized_coords = denormalized_coords.astype(int)

    # Grid.
    grid_x = numpy.arange(image_width)
    grid_y = numpy.arange(image_height)

    # Images generation.
    images_x = numpy.clip(numpy.minimum(grid_x[None,...] - denormalized_coords[:,0,None], denormalized_coords[:,1,None] - grid_x[None,...]), 0.0, 1.0)
    images_y = numpy.clip(numpy.minimum(grid_y[None,...] - denormalized_coords[:,2,None], denormalized_coords[:,3,None] - grid_y[None,...]), 0.0, 1.0)
    images = images_x[:,:,None] * images_y[:,None,:]

    return images