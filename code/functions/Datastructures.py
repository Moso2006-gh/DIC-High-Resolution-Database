import numpy as np
from typing import TypedDict, TypeAlias, List

# An outline of a cell, represented as a 2D array of coordinates (n_points, 2)
Outline : TypeAlias = np.ndarray

# The centroid of a cell, represented as a 1D array of length 2 (y, x)
Centroid : TypeAlias = np.ndarray

# A shape representation of a cell: the outline minus the centroid
Cell_Shape : TypeAlias = np.ndarray  # shape: (n_points, 2)

# A dictionary representing a single cell
class Cell(TypedDict):
    shape: Cell_Shape    # the cell boundary coordinates
    centroid: Centroid  # the cell center coordinates

# A dictionary representing a cell at a specific frame
class Track_Point(TypedDict):
    frame: int   # frame index
    cell: Cell   # the cell data at this frame

# A list of TrackPoints representing the trajectory of a cell across frames
Track : TypeAlias = List[Track_Point]

# A dictionary representing a cell at a specific frame
class Track_Info_Point(TypedDict):
    frame: int  # The frame in the original tiff sequence from where the shot was extracted
    concentration_top: float  # Top-side concentration measurement of the cell
    concentration_bottom: float  # Bottom-side concentration measurement of the cell
    cell: Cell  # The cell data, including outline and centroid (centroid can be use as coordinates)

# A list of Track_Info_Point dictionaries representing a full cell track file over time
Track_Info : TypeAlias = List[Track_Info_Point]