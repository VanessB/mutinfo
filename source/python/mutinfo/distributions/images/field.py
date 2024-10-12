import numpy

from collections.abc import Callable


def draw_field(parameters: numpy.ndarray,
               field_function: Callable[[list[numpy.ndarray]], numpy.ndarray],
               grid_shape: tuple[int, ...]) -> numpy.ndarray:
    """
    Uniformly evaluate and draw a field.
    
    Parameters
    ----------
    parameters : numpy.ndarray
        Samples of parameters.
    field_function : Callable[[list[numpy.ndarray]], numpy.ndarray]
        Parametric function defining a field.
    grid_shape : tuple[int, ...]
        Shape of a uniform grid, on which the field is evaluated.

    Returns
    -------
    images : numpy.ndarray
        Multidimensional array of field values on the grid.
    """
    
    grid = numpy.meshgrid(*[numpy.linspace(0.0, 1.0, n_points) for n_points in grid_shape])
    grid = [coordinate[numpy.newaxis,...] for coordinate in grid]
    
    images = field_function(grid, parameters)
                
    return images


def symmetric_gaussian_field(grid: list[numpy.ndarray], parameters: numpy.ndarray, sigma: float=0.2) -> numpy.ndarray:
    """
    Symmetric Gaussian field.

    Parameters
    ----------
    grid : list[numpy.ndarray]
        d-Dimensional grid to evaluate the field on.
    parameters : numpy.ndarray
        Coordinates of the mode, shape: (?,d).
    sigma : float, optional
        Scale parameter.

    Returns
    -------
    images : numpy.ndarray
        Multidimensional array of field values on the grid.
    """

    dimensionality = len(grid)
    parameters = parameters[(...,) + (numpy.newaxis,) * dimensionality]

    return numpy.exp(-0.5 * sum((coordinate - parameters[:,axis])**2 for axis, coordinate in enumerate(grid)) / sigma**2)