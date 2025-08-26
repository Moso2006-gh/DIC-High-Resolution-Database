import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from .Datastructures import Track_Info

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

def convert_np_to_json(o):
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
    with open(Path(path_to_db_entry) / "track_data.json", "r") as f:
        json_data = json.load(f)
    return convert_json_to_np(json_data)

def angle_between(u, v):
    u = np.array(u)
    v = np.array(v)
    
    dot_product = np.dot(u, v)
    norm_product = np.linalg.norm(u) * np.linalg.norm(v)
    
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    
    return np.arccos(cos_theta)