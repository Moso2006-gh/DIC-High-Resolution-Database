import os
import sys
import math
import json
import pickle
import shutil
import traceback
import contextlib
import numpy as np
import pandas as pd
import trackpy as tp
from tqdm import tqdm
from pathlib import Path
import imageio.v2 as imageio
from typing import List, Tuple
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from .Miscellaneous import convert_np_to_json, angle_between
from .Datastructures import Outline, Centroid, Cell, Track, Cell_Shape, Track_Info

#Contour analysis
def get_outlines(mask: np.ndarray, buffer: int = 50) -> List[Outline]:
    """
    Extract pixel coordinate outlines for labeled regions in a mask, 
    excluding labels that touch the image border within a given buffer.

    Args:
        mask (np.ndarray): 2D labeled mask array where 0 represents background 
            and positive integers represent distinct regions.
        buffer (int): Width (in pixels) of the border zone to exclude from outline extraction.

    Returns:
        List[np.ndarray]: A list of 2D arrays, each containing (row, col) coordinates 
        for the outline pixels of a label that does not touch the excluded border area.

    Process:
        1. Pads the mask to make boundary comparisons easier.
        2. Identifies outline pixels by checking if a pixel differs from any neighbor.
        3. Builds a filter to exclude objects touching a 'buffer' region near edges.
        4. Collects coordinate arrays for each valid label’s outline.
    """
    pad = np.pad(mask, 1, constant_values=0)
    
    center = pad[1:-1, 1:-1]
    left = pad[1:-1,:-2]
    up = pad[:-2, 1:-1]
    right = pad[1:-1, 2:]
    down = pad[2:, 1:-1]
    
    
    filter = (center != left) | (center != up) | (center != right) | (center != down)
    filter &= (center != 0)
    
    y, x = mask.shape
    edge_filter = np.zeros(shape=(y - 2*buffer, x - 2*buffer), dtype=bool)
    edge_filter = np.pad(edge_filter, buffer, constant_values=1)
    
    out_of_bounds = np.unique(mask[edge_filter == 1])

    outlines = []
    
    for label in np.unique(mask):
        if label == 0 or label in out_of_bounds:
            continue
        
        outline = np.column_stack(np.where((mask == label) & filter))
        outlines.append(outline)
    return outlines

def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order a set of (row, col) points in sequence 
    by iteratively following the nearest neighbor in one of 8 directions.

    Args:
        points (Iterable[tuple[int, int]]): Coordinates of points to order, 
            given as (row, col) integer tuples.

    Returns:
        np.ndarray: Array of points ordered as shape (N, 2), 
        where each row is (row, col). Or empty array if it couldnt close the shape
    """
    points_set = set(map(tuple, points))

    mean_y = np.mean([y for y, _ in points])
    candidates = [p for p in points_set if p[0] < mean_y]
    xs = sorted(x for _, x in points_set)
    median_x = xs[len(xs) // 2]

    start_point = min(candidates, key=lambda p: abs(p[1] - median_x))
    
    neighbor_dirs = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]
    
    outlines = [[start_point]]
    seen = {dir: set(start_point) for dir in neighbor_dirs}
    while outlines:
        new_outlines = []
        for outline in outlines:
            last_point = outline[-1]

            for dir in neighbor_dirs:
                forward_point = (last_point[0] + dir[0], last_point[1] + dir[1])

                if forward_point == start_point:
                    if len(outline) > len(points_set) * 0.5:
                        return np.array(outline)
                    
                    for i in range(1, len(outline)):
                        faulty_dir = (outline[i][0] - outline[i - 1][0], outline[i][1] - outline[i - 1][1])
                        if outline[i] in seen[faulty_dir]:
                            seen[faulty_dir].remove(outline[i])
                    
                if forward_point in points_set and forward_point not in outline and forward_point not in seen[dir]:
                    new_outlines.append(outline + [forward_point])
                    seen[dir].add(forward_point)
        outlines = new_outlines
    return np.array([])

def regular_contour_of_points(points: np.ndarray, n: int, order: bool = True) -> np.ndarray:
    """
    Generates a regularly spaced contour of n points along a given set of points.
    Optionally orders the points clockwise before interpolation to ensure consistent traversal.
    """
    if order:
        points = order_points(points)
        
    
    if len(points) < 3:
        return []

    regular_contour = np.zeros((n, 2)) 
    for indx, i in enumerate(np.linspace(0, 1, n, endpoint=False) * len(points)):
        i_n = int(np.floor(i))
        r = i - i_n

        if i_n + 1 >= len(points):
            next_point = points[0]
        else:
            next_point = points[i_n + 1]

        regular_contour[indx] = (1 - r) * points[i_n] + r * next_point
    
    foward_dir = regular_contour[1, :] - regular_contour[0, :]
    perpendicular = np.array([foward_dir[1], -foward_dir[0]])

    vectors = [pt - regular_contour[0, :] for pt in regular_contour[1:]]
    closest_vector = min(vectors, key=lambda x: angle_between(x, perpendicular))

    if np.rad2deg(angle_between(closest_vector, perpendicular)) < 20:
        regular_contour = np.vstack([regular_contour[0], regular_contour[1:][::-1]]) 
    return regular_contour

def get_cells_from_mask(mask: np.ndarray, buffer: int = 50, n: int = 100) -> List[Cell]:
    """
    Extracts cell shapes and centroids from a mask.

    Args:
        mask (np.ndarray): mask.
        buffer (int): Number of pixels to ignore near image edges.
        n (int): Number of points to resample each cell shape to.

    Returns:
        List[Cells]: A list of cells.
            Each cell is a dict with keys:
                'shape'  -> np.ndarray of shape (n, 2)
                'centroid' -> np.ndarray of shape (2,)
    """
    outlines = get_outlines(mask, buffer)

    cells: List[Cell] = []
    for o, outline in enumerate(outlines):
        regular_outline = regular_contour_of_points(outline, n)

        if len(regular_outline) < n:
            print(f"\nFound a faulty cell {o}")
            continue
        
        centroid: Centroid = np.mean(regular_outline, axis=0)
        cell: Cell = {
            "shape": regular_outline - centroid,
            "centroid": centroid
        }
        
        cells.append(cell)
    return cells

def get_cells_per_frame(path_to_masks: Path, buffer: int = 50, n: int = 100) -> List[List[Cell]]:
    """
    Extracts cell shapes and centroids from mask files.

    Args:
        path_to_masks (Path): Directory containing .npy mask files.
        buffer (int): Number of pixels to ignore near image edges.
        n (int): Number of points to resample each cell shape to.

    Returns:
        List[List[Cell]]: A list of frames, where each frame is a list of cells.
            Each cell is a dict with keys:
                'shape'  -> np.ndarray of shape (n, 2)
                'centroid' -> np.ndarray of shape (2,)
    """
    cells_per_frame: List[List[Cell]] = []
    mask_files = list(path_to_masks.glob("*.npy"))
    for file in tqdm(mask_files, desc="🦠 Extracting cells", file=sys.stdout):
        mask = np.load(file)
        cells = get_cells_from_mask(mask, buffer, n)
        cells_per_frame.append(cells)
    return cells_per_frame

def get_cells_per_frame_from_tracks(tracks: List[Track]) -> List[List[Cell]]:
    """
    Organizes cell objects from a list of tracks into a per-frame structure.

    Args:
        tracks (List[Track]): A list of tracks, where each track is a list of
                              trackpoints. Each trackpoint is expected to be a 
                              dictionary containing at least:
                                - "frame": the frame index (int)
                                - "cell": the associated Cell object

    Returns:
        List[List[Cell]]: A list where each element corresponds to a frame and 
                          contains a list of Cell objects present in that frame.
    """
    max_frame = max(trackpoint["frame"] for track in tracks for trackpoint in track)
    cells_per_frame: List[List[Cell]] = [[] for _ in range(max_frame)]
    
    for track in tracks:
        for trackpoint in track:
            frame = trackpoint["frame"]
            cells_per_frame[frame - 1].append(trackpoint["cell"])       
    return cells_per_frame

def get_shape_area_and_max_distance(points: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the area and max distance between two points.

    Args:
        points (np.ndarray): A list of points (n, 2)
                                
    Returns:
        (area, max_distance): The area of the shape enclosed by the points and 
                            the max distance between two points
    """
    hull = ConvexHull(points, qhull_options="QJ")
    hull_pts = points[hull.vertices]
    
    diff = hull_pts[:, None,:] - hull_pts[None,:,:]
    dist_sq = np.sum(diff**2, axis=-1)
    return hull.volume, np.sqrt(np.max(dist_sq))

#Timestep Analysis
def get_cell_tracks(cells_per_frame: List[List[Cell]], max_disp: int = 100, gaps: int = 5) -> List[Track]:
    """
    Links cells across frames to generate tracks.

    Args:
        cells_per_frame (List[List[Dict[str, np.ndarray]]]): List of frames, each containing a list of cells.
            Each cell is a dict with keys 'shape' and 'centroid', both np.ndarray.
        max_disp (int): Maximum allowed displacement between frames for tracking.
        gaps (int): Maximum number of frames a cell can disappear and still be linked.

    Returns:
        List[List[Dict[str, np.ndarray]]]: A list of cell tracks. Each track is a list of dictionaries representing
        a single cell's presence in consecutive frames. Each dictionary contains:
            'frame' (int): The index of the frame in which the cell appears.
            'cell' (dict): A dictionary describing the cell, containing:
                'shape' (np.ndarray): The coordinates of the cell's boundary in that frame.
                'centroid' (np.ndarray): The coordinates of the cell's center point in that frame.
    """
    print("🧭 Tracking cells...")
    rows = []
    for frame_idx, cells in enumerate(cells_per_frame):
        for cell_index, cell in enumerate(cells):
            y,x = cell["centroid"]
            rows.append({"frame": frame_idx, "cell_index": cell_index, "x": x, "y": y})

    df = pd.DataFrame(rows)

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        linked = tp.link_df(df, search_range=max_disp, memory=gaps)

    # Get tracks
    tracks = []
    for particle in np.unique(linked["particle"].to_numpy()):
        track = []
        track_df = linked[linked["particle"] == particle]
        
        if len(track_df) < 10:
            continue
        
        for row in track_df.itertuples():
            frame = row.frame
            cell_index = row.cell_index
            
            track.append({
                "frame": frame,
                "cell": cells_per_frame[frame][cell_index]
            })
        tracks.append(track)
    return tracks

def get_cell_shape_evolution(track: Track) -> List[Cell_Shape]:
    """
    Computes the shape evolution of a cell along its track.

    Args:
        track (Track): A list of TrackPoints, each containing:
            - 'frame' (int): frame index
            - 'cell' (Cell): cell data, including:
                - 'shape' (np.ndarray, shape: n_points x 2): the cell boundary
                - 'centroid' (np.ndarray, shape: 2): the cell center

    Returns:
        List[Cell_Shape]: List of arrays (shape: n_points x 2).
    """
    n = len(track[0]["cell"]["shape"])
    cell_shape_evolution = np.zeros((len(track), n, 2))

    for pos_idx, pos in enumerate(track):
        cell_shape_evolution[pos_idx] = pos["cell"]["shape"]   
    return cell_shape_evolution

def get_elongation_over_time(shape_evolution: List[Cell_Shape]) -> np.ndarray:
    """
    Computes the elongation of a cell over time based on its shape evolution.

    For each frame:
    - Constructs the convex hull of the cell shape.
    - Measures the maximum distance between any two points on the hull (cell length).
    - Estimates an effective diameter from the hull area assuming a circular shape.
    - Computes elongation as the ratio of maximum distance (length) to diameter.

    Args:
        shape_evolution (List[Cell_Shape]): 
            List of arrays representing the cell shape at each frame, 
            centered on the centroid and uniformly resampled.
            Each array has shape (n_points, 2).

    Returns:
        np.ndarray: 1D array of elongation values (length / diameter) for each frame.
                    Frames with insufficient points or hull computation errors are skipped.
    """
    n_frames, _, _ = shape_evolution.shape
    
    areas = np.zeros(n_frames)
    max_dists = np.zeros(n_frames)

    for i, points in enumerate(shape_evolution):
        if len(np.unique(points, axis=0)) < 3:
            continue

        hull = ConvexHull(points, qhull_options="QJ")
        hull_pts = points[hull.vertices]
        
        diff = hull_pts[:, None,:] - hull_pts[None,:,:]
        dist_sq = np.sum(diff**2, axis=-1)
        max_dists[i] = np.sqrt(np.max(dist_sq))
        areas[i] = hull.volume
    
    diameters = 2 * np.sqrt(areas)/math.pi
    safe_diameters = np.where(diameters == 0, np.nan, diameters)
    return max_dists / safe_diameters

def get_elongations_over_time_for_all_cells(tracks: List[Track]) -> List[np.ndarray]:
    """
    Computes elongation over time for all tracked cells.

    For each cell track:
    - Extracts the cell shape evolution across frames.
    - Computes elongation (length / diameter) for each frame.
    
    Args:
        tracks (List[Track]): List of cell tracks

    Returns:
        List[np.ndarray]: A list of 1D arrays, each containing the elongation values 
                          for a single cell across its tracked frames.
    """
    elongations: List[np.ndarray] = []
    for track in tqdm(tracks, desc="📏 Getting elongations", file=sys.stdout):
        cell_shape_evolution: List[Cell_Shape] = get_cell_shape_evolution(track)
        elongation: np.ndarray = get_elongation_over_time(cell_shape_evolution)
        elongations.append(elongation)
    return elongations 

def track_cells_and_get_elongations(path_to_masks: Path, buffer: int = 50, max_disp: int = 100, gaps: int = 5, overwrite: bool = False) -> List[Track]:
    """
    Track cells across frames and save their elongation measurements.

    This function processes a directory of mask files to identify and track individual cells over time.
    It saves both the cell tracks (`tracks.pkl`) and their elongation data (`elongations.txt`) to disk.
    The elongation file includes only tracks with at least 10 frames of data.

    Args:
        path_to_masks (Path): Path to the folder containing mask `.npy` files.
        buffer (int, optional): Border padding to ignore near-frame-edge cells. Default is 50.
        max_disp (int, optional): Maximum allowed displacement between frames when linking cells. Default is 100.
        gaps (int, optional): Maximum number of frame gaps allowed when linking tracks. Default is 5.
        overwrite (bool, optional): If False and outputs already exist, skip processing. Default is False.

    Returns:
        List[Track]: A list of tracked cells across frames.
    """
    if overwrite:
        print("💣 Overwriting tracks")
    
    if (not overwrite) & (Path(path_to_masks).parent / "tracks.pkl").exists():
        print(f"⏩ Not computing tracks for {Path(path_to_masks).parent} because they already exists...")
        return np.load(Path(path_to_masks).parent / "tracks.pkl", allow_pickle=True)
    
    cells_per_frame = get_cells_per_frame(path_to_masks, buffer)
    tracks = get_cell_tracks(cells_per_frame, max_disp, gaps)

    with open(Path(path_to_masks).parent / "tracks.pkl", "wb") as f:
        pickle.dump(tracks, f)

    elongations = get_elongations_over_time_for_all_cells(tracks)
    
    with open(Path(path_to_masks).parent / "elongations.txt", "w") as file:
        for row in elongations:
            line = f"{len(row)}, " + ", ".join(str(item) for item in row)            
            file.write(line + "\n")
    return tracks

def get_cell_shape_evolution_from_track(track: Track) -> List[Cell_Shape]:
    """
    Get the cell shape evolution from the track.
    
    Args:
        track (Track): A list of TrackPoints, each containing:
            - 'frame' (int): frame index
            - 'cell' (Cell): cell data, including:
                - 'shape' (np.ndarray, shape: n_points x 2): the cell boundary
                - 'centroid' (np.ndarray, shape: 2): the cell center
        step (int, optional): number of frames used to calculate the speed

    Returns:
        List of the cell shapes
    """
    return np.array([pos["cell"]["shape"] for pos in track])


def get_track_velocities(track: Track, step: int = 10) -> np.ndarray:
    """Calculate the track velocities

    Args:
        track (Track): A list of TrackPoints, each containing:
            - 'frame' (int): frame index
            - 'cell' (Cell): cell data, including:
                - 'shape' (np.ndarray, shape: n_points x 2): the cell boundary
                - 'centroid' (np.ndarray, shape: 2): the cell center
        step (int, optional): number of frames used to calculate the speed

    Returns:
        np.ndarray: Vector velocities for each frame
    """
    centroids = np.array([pos["cell"]["centroid"] for pos in track])
    
    displacements = centroids[step:] - centroids[:-step]
    velocities = displacements / step
    
    last_val = velocities[-1]
    pad = np.tile(last_val, (step, 1)) 
    velocities = np.vstack([velocities, pad]) 
    return velocities

def get_shape_theta(cell_shape: Cell_Shape, velocity: np.ndarray, south: bool = False) -> Tuple[float, np.ndarray]:
    """
    Compute the orientation angle of a cell shape relative to a reference direction.

    Uses Singular Value Decomposition (SVD) to determine the principal axis of the cell shape
    and calculates the angle between this axis and a fixed horizontal direction (left or right),
    depending on the hemisphere.

    Args:
        cell_shape (Cell_Shape): 2D array (n_points x 2) representing the cell's shape,
                                 with each point as (y, x).
        velocity (np.ndarray): A 2D vector representing the cell's movement direction.
        south (bool, optional): Whether to use the southern hemisphere convention.
                                If False (default), angle is measured from the left (-x axis);
                                if True, from the right (+x axis).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - theta_deg (float): The orientation angle of the shape in degrees [0, 360),
                                 measured relative to the specified reference axis.
            - direction (np.ndarray): A 2D unit vector representing the principal orientation.
    """
    _, _, Vt = np.linalg.svd(np.column_stack((cell_shape[:, 0], cell_shape[:, 1]))) 
    direction = Vt[0]
    if np.dot(direction, velocity) < 0:
        direction = -direction                 

    if not south:
        theta_rad = angle_between(direction, (-1, 0))
    else:
        theta_rad = angle_between(direction, (1, 0))
    theta_deg = np.degrees(theta_rad) % 360
    return theta_deg, direction

def get_cell_theta_over_time(track: Track, tolerance: int = 0.3, south: bool = False) -> np.ndarray:
    """
    Compute cell theta over time for one cell. Discards values if the area of the cell changes by more than a certain tolerance

    Args:
        track (Track)

    Returns:
        np.ndarray (frames, 2): Each subarray  
    """
    frames_and_thetas = []
    
    velocities = get_track_velocities(track)
    last_area = None
    for i, trackpoint in enumerate(track):
        frame = trackpoint['frame']
        cell = trackpoint['cell']
        
        cell_shape = cell["shape"]
        area, _ = get_shape_area_and_max_distance(cell_shape)
        
        if (not last_area) or (not abs(last_area - area) > last_area * tolerance):
            theta, _ = get_shape_theta(cell_shape, velocities[i], south)
            frames_and_thetas.append([frame, theta])
            last_area = area
    return np.array(frames_and_thetas)

def get_point_trayectory_and_instant_change(cell_shape_evolution: List[Cell_Shape], point_index: int):
    """
    Analyze the trajectory and instantaneous change of a specific outline point over time.

    Projects the movement of a selected point on the cell outline onto its principal axis (via PCA),
    computes its relative displacement from the starting position, and calculates the instantaneous change (derivative).

    Args:
        cell_shape_evolution (List[Cell_Shape]): List of cell shapes over time (frames, n_points, 2).
        point_index (int): Index of the outline point to analyze.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Relative displacement of the point over time.
            - Instantaneous change (first derivative) of the displacement.
    """
    point_of_interest = cell_shape_evolution[:, point_index] 
    pca = PCA(n_components=1) 
    X_pca = pca.fit_transform(point_of_interest).flatten() 
    X_relative = abs(X_pca - X_pca[0]) 
    dx = np.diff(X_relative, 1)
    return X_relative, dx

#Saving final entries
def track_to_tiff_files(track: Track, index: int, tif_files: List[Path], output_path: Path, top_con: float, bottom_con: float, y_sup: int, x_sup: int, radius: float, flip: bool) -> Track_Info:
    """
    Process a cell track and generate cropped TIFF images for each frame.
    
    Args:
        track: List of cell positions with centroid and shape information.
        index: Index of the track (used for display purposes).
        tif_files: List of TIFF file paths corresponding to frames.
        output_path: Directory to save cropped images.
        top_con: Concentration at the top of the bounding box.
        bottom_con: Concentration at the bottom of the bounding box.
        y_sup: Maximum y-dimension of the original image.
        x_sup: Maximum x-dimension of the original image.
        radius: Radius around the centroid to crop.
        flip: Whether to vertically flip the crop.
        
    Returns:
        track_info: List of dictionaries containing frame info, 
                    concentration values, and the cell object.
    """
    
    def con_at_coord(cord, top_con, bottom_con, max_cord):
        lerp = cord / max_cord
        return bottom_con * (1 - lerp) + top_con * lerp
    
    def get_bounding_coord(center, radius, sup):
        max, min = center + radius, center - radius
        if sup < max:
            min -= max - sup
            max = sup
        if min < 0:
            max += abs(min)
            min = 0
        return max, min
    
    def moving_average(data, window_size=3):
        series = pd.Series(data)
        return series.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

    output_path.mkdir(parents=True, exist_ok=True)
    
    centroids: List[Centroid] = [position["cell"]["centroid"] for position in track]
    ys, xs = zip(*centroids)
    ys_smoothed = moving_average(np.array(ys))
    xs_smoothed = moving_average(np.array(xs))
    
    track_info: Track_Info = []
    for i, position in enumerate(tqdm(track, f"🏃 Track {index}", file=sys.stdout)):
        frame = position["frame"]
        cell: Cell = position["cell"]
        
        y_center = int(ys_smoothed[i])
        x_center = int(xs_smoothed[i])

        y_max, y_min = get_bounding_coord(y_center, radius, y_sup)
        x_max, x_min = get_bounding_coord(x_center, radius, x_sup)
        
        con_top = con_at_coord(y_max, bottom_con, top_con, y_sup)   
        con_bottom = con_at_coord(y_min, bottom_con, top_con, y_sup)  

        track_info.append({
            "frame": frame,
            "concentration_top": con_top,
            "concentration_bottom": con_bottom,
            "cell": cell
        })
                    
        background_img = imageio.imread(tif_files[frame])
        if not flip:
            cropped = background_img[y_min:y_max, x_min:x_max]
        else:
            cropped = background_img[y_max:y_min:-1, x_min:x_max]
            
            cell["shape"] = np.array([(-pos[0], pos[1]) for pos in cell["shape"]], dtype=float) 
            cell["centroid"][0] = y_sup - cell["centroid"][0]
            con_top, con_bottom = con_bottom, con_top

        output_file = output_path / f"{frame}.tif"
        imageio.imwrite(output_file, cropped.astype(background_img.dtype))
    return track_info
        
def create_data_entries(top_con: float, bottom_con: float, tracks: List[Track], path_to_tifs: Path, output_path: Path, radius: int = 200, flip: bool = False, overwrite: bool = False) -> None:
    """
    Create data entries for cell tracks by processing image data and extracting relevant information.

    This function loads cell tracks from a .npy file and corresponding TIFF images, optionally flips
    the y-coordinates if the top concentration is lower than the bottom, and generates cropped 
    images centered around each cell. It also calculates top and bottom concentrations for each 
    cell position and saves all relevant information for each track as a JSON file.

    Args:
        top_con (float): The maximum concentration value at the top of the image.
        bottom_con (float): The minimum concentration value at the bottom of the image.
        path_to_tracks (Path): Path to the .npy file containing the list of cell tracks.
        path_to_tifs (Path): Path to the folder containing the TIFF images.
        output_path (Path): Path to the folder where tracks will be saved.
        radius (int, optional): Radius (in pixels) around the cell centroid to crop the image. Default is 200.
        flip (bool, optional): Whether to flip the y-coordinates when top_con < bottom_con. Default is False.

    Returns:
        None
    """    
    tif_files = sorted(Path(path_to_tifs).glob("*.tif"))
    y_sup, x_sup = imageio.imread(tif_files[0]).shape

    flip = (top_con < bottom_con) and flip
    if flip:
        print("🔄 Flipping y cord")
    
    if overwrite:
        print("💣 Overwriting db entries")
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    for index, track in (tqdm(enumerate(tracks), f"🗃️ Saving tracks", total=len(tracks), file=sys.stdout)):
        try:
            output_folder = Path(output_path, str(index))
            output_folder.mkdir(parents=True, exist_ok=True)

            if len(list((output_folder / "tiffs").glob("*.tif"))):
                print(f"⏩ Skipping track {index} because already exists...")
                continue

            track_info = track_to_tiff_files(track, index, tif_files, output_folder / "tiffs", top_con, bottom_con, y_sup, x_sup, radius, flip)
            with open(Path(output_folder / "track_data.pkl"), 'wb') as f:
                pickle.dump(track_info, f)

            with open(Path(output_folder) / "track_data.json", 'w') as f:
                json.dump(track_info, f, default=convert_np_to_json)

            if top_con < bottom_con and (not flip):
                theta_over_time = get_cell_theta_over_time(track, south=True)
            else:
                theta_over_time = get_cell_theta_over_time(track)
            
            df = pd.DataFrame(theta_over_time, columns=["frame", "theta"])
            df.to_csv(Path(output_folder) / "theta_over_time.csv", index=False)
        except Exception:
            traceback.print_exc()
            shutil.rmtree(output_folder)
    return
