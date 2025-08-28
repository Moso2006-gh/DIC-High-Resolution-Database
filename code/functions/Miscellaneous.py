import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from .Datastructures import Track_Info, Track
import imageio.v2 as imageio

def create_log(name_of_log : str, path_to_logs : Path) -> None:
    """
    Creates a timestamped log file and redirects all stdout/stderr output 
    to both the console and the log file simultaneously.

    Parameters:
        name_of_log (str): Base name for the log file.
        path_to_logs (Path): Directory where logs will be stored.

    The function ensures the 'logs' subfolder exists, appends a timestamp 
    to the log filename, and replaces sys.stdout/sys.stderr with a Tee 
    object that writes output to both the terminal and the log file.
    """
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()
    Path(path_to_logs).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = path_to_logs / f"{name_of_log}_log_{timestamp}.log"
    log_file = open(log_filename, "w", encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return None

def get_tif_files(path_to_tifs: Path, background: bool, max_length=9000) -> Tuple[np.ndarray, int, int]:
    """
    Retrieve TIFF files and intensity bounds for background plotting.

    Args:
        path_to_tifs (Path): Directory containing TIFF files.
        background (bool): Whether to use background images.
        max_length (int, optional): Maximum number of frames if no background.

    Returns:
        Tuple: (List of TIFF file paths, vmin, vmax) if background is True,
               otherwise (np.zeros array of max_length).
    """
    if background:
        tif_files = sorted(Path(path_to_tifs).glob("*.tif"))
        background_img = imageio.imread(tif_files[0])
        vmin, vmax = np.percentile(background_img, [0.01, 99.9])  # 1st and 99th percentiles
        return tif_files, vmin, vmax
    else:
        return np.zeros(max_length)

def convert_np_to_json(o):
    """
    Convert NumPy types to native Python types for JSON serialization.

    Args:
        o: Object to convert.

    Returns:
        Converted object suitable for JSON serialization.

    Raises:
        TypeError: If the type is not serializable.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, (np.integer,)):
        return int(o)
    elif isinstance(o, (np.floating,)): 
        return float(o)
    elif isinstance(o, (np.bool_)):
        return bool(o)
    elif isinstance(o, (np.str_)):
        return str(o)
    raise TypeError(f"Type {type(o)} not serializable")

def convert_json_to_np(o):
    """
    Recursively convert JSON-compatible objects to NumPy types.

    Args:
        o: Object to convert.

    Returns:
        Object with NumPy types restored.
    """
    if isinstance(o, dict):
        return {k: convert_json_to_np(v) for k, v in o.items()}
    elif isinstance(o, list):
        converted = [convert_json_to_np(item) for item in o]
        try:
            return np.array(converted)
        except Exception:
            return converted
    elif isinstance(o, bool):
        return np.bool_(o)
    elif isinstance(o, int):
        return np.int64(o)
    elif isinstance(o, float):
        return np.float64(o)
    elif isinstance(o, str):
        return np.str_(o)
    else:
        return o

def load_track_from_db(path_to_db_entry: Path) -> Track_Info:
    """
    Load a cell track from a database entry.

    Args:
        path_to_db_entry (Path): Path to the database entry directory.

    Returns:
        Track_Info: Loaded track information as NumPy types.
    """
    with open(Path(path_to_db_entry) / "track_data.json", "r") as f:
        json_data = json.load(f)
    return convert_json_to_np(json_data)

def flip_track(track: Track, max_y: int = 2200) -> Track:
    """
    Flip a cell track vertically.

    Args:
        track (Track): Track to flip.
        max_y (int, optional): Maximum y-value for flipping.

    Returns:
        Track: Vertically flipped track.
    """
    for pos in track:
        pos["cell"]["shape"][:, 0] = -pos["cell"]["shape"][:, 0]
        pos["cell"]["centroid"][0] = max_y - pos["cell"]["centroid"][0]
    return track

def flip_tracks(tracks: List[Track], max_y: int = 2200) -> List[Track]:
    """
    Flip multiple cell tracks vertically.

    Args:
        tracks (List[Track]): List of tracks to flip.
        max_y (int, optional): Maximum y-value for flipping.

    Returns:
        List[Track]: List of flipped tracks.
    """
    for track in tracks:
        track = flip_track(track, max_y)
    return tracks

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the angle (in radians) between two vectors.

    Args:
        u (np.ndarray): First vector.
        v (np.ndarray): Second vector.

    Returns:
        float: Angle in radians between vectors u and v.
    """
    u = np.array(u)
    v = np.array(v)
    
    dot_product = np.dot(u, v)
    norm_product = np.linalg.norm(u) * np.linalg.norm(v)
    
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    
    return np.arccos(cos_theta)