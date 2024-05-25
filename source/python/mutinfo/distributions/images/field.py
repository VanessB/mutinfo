import numpy

from ..base import CorrelatedNormal
from .. import mapped


def draw_field(parameters: numpy.array, field_function: callable,
               grid_shape: tuple) -> list:
    """
    Uniformly evaluate and draw a field.
    
    Parameters
    ----------
    parameters : numpy.array
        Samples of parameters.
    field_function : callable
        Parametric function defining a field.
    grid_shape : tuple
        Shape of a uniform grid, on which the field is evaluated.

    Returns
    -------
    images : list(numpy.array)
        Multidimensional array of field values on the grid.
    """
    
    grid = numpy.meshgrid(*[numpy.linspace(0.0, 1.0, n_points) for n_points in grid_shape])
    grid = [coordinate[numpy.newaxis,...] for coordinate in grid]
    
    images = field_function(grid, parameters)
                
    return images


def symmetric_gaussian_field(grid: list, parameters: numpy.array, sigma: float=0.2):
    """
    Symmetric Gaussian field.

    Parameters
    ----------
    grid : list(numpy.array)
        d-Dimensional grid to evaluate the field on.
    parameters : numpy.array
        Coordinates of the mode, shape: (?,d).
    sigma : float, optional
        Scale parameter.
    """

    dimension = len(grid)
    parameters = parameters[(...,) + (numpy.newaxis,) * dimension]

    return numpy.exp(-0.5 * sum((coordinate - parameters[:,axis])**2 for axis, coordinate in enumerate(grid)) / sigma**2)