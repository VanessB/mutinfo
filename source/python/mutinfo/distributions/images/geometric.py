import numpy
import math


def uniform_to_segment(x: numpy.ndarray, min_length: float=0.0) -> numpy.ndarray:
    """
    Map a 2D uniformly distributed random vector to a random segment.
    The coordinates of the ends are distributed uniformly, preserving order.
    
    Parameters
    ----------
    x : numpy.ndarray
        Samples from an uniform distribution on [0; 1]^2, shape: (?,2).
    min_length : float, optional
        Minimum length of the segment (must lie in [0; 1)).

    Returns
    -------
    coords : numpy.ndarray
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


def uniform_to_rectangle(x: numpy.ndarray, min_size: tuple[int, ...], nax_size: tuple[int, ...]=None) -> numpy.ndarray:
    """
    Map a 2d-dimensional uniformly distributed random vector
    to a random d-dimensional hyperrectangle.
    The coordinates of the corners are distributed uniformly, preserving order.
    
    Parameters
    ----------
    x : numpy.ndarray
        Samples from an uniform distribution on [0; 1]^{2d}, shape: (?,2d).
    min_size : tuple[int, ...]
        Minimum size of the rectangle.
    max_size : tuple[int, ...], optional
        Maximum size of the rectangle.
        Default value: (1.0, ...).

    Returns
    -------
    coords : numpy.ndarray
        Coordinates of the corners of sampled random hyperrectangles.
    """

    if nax_size is None:
        max_size = (1.0,) * len(min_size)
    elif len(min_size) != len(max_size):
        raise ValueError("Expected `min_size` and `max_size` to be of the same length.")
        
    dimensionality = len(min_size)

    if len(x.shape) != 2 or x.shape[1] != 2*dimensionality:
        raise ValueError(f"Expected `x` to be of shape (?,{2*dimensionality})")

    coords = numpy.empty_like(x)
    for axis in range(dimensionality):
        coords[:,2*axis:2*(axis+1)] = uniform_to_segment(x[:,2*axis:2*(axis+1)], min_size[axis] / max_size[axis]) * max_size[axis]

    return coords


def draw_rectangle(coords: numpy.ndarray, image_shape: tuple[int, ...]) -> numpy.ndarray:
    """
    Map coordinates of hyperrectangles to corresponding rasterized images.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Normalized coordinates of the corners of d-dimensional hyperrectangles,
        shape: (?,2d).
    image_shape : tuple[int, ...]
        Image size in pixels/voxels/...

    Returns
    -------
    images : numpy.ndarray
        Rasterized images of hyperrectangles.
    """

    dimensionality = len(image_shape)
    
    if len(coords.shape) != 2 or coords.shape[1] != 2*dimensionality:
        raise ValueError(f"Expected `coords` to be of shape (?,{2*dimensionality})")
        
    n_samples = coords.shape[0]

    # Denormalized coordinates.
    denormalized_coords = numpy.empty_like(coords)
    for axis in range(dimensionality):
        denormalized_coords[:,2*axis:2*(axis+1)] = coords[:,2*axis:2*(axis+1)] * image_shape[axis]

    # Grid.
    grid = [numpy.arange(image_shape[axis]) for axis in range(dimensionality)]

    # Images generation.
    # Oh god, oh no...
    images_projections = [
        numpy.clip(
            numpy.minimum(
                grid[axis][numpy.newaxis,...] - denormalized_coords[:,2*axis,numpy.newaxis],
                denormalized_coords[:,2*axis+1,numpy.newaxis] - grid[axis][numpy.newaxis,...]
            ),
            0.0, 1.0
        ) for axis in range(dimensionality)
    ]

    # Black magic: d-dimensional outer product.
    # Why can't numpy.meshgrid work with 2D arrays?
    images_projections = [
        images_projections[axis][(slice(None),) + (numpy.newaxis,)*axis + (slice(None),) + (numpy.newaxis,)*(dimensionality - axis - 1)]
        for axis in range(dimensionality)
    ]

    #images = numpy.multiply.reduce(images_projections)
    # Broadcasting does not work with numpy.multiply.reduce
    images = math.prod(images_projections)

    return images