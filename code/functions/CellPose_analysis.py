import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tifffile import imread
from cellpose import models
from typing import List

def calculate_masks(path_to_tifs: Path, output_dir: Path, model_path: Path, use_gpu: bool, overwrite: bool = False) -> Path:
    """
    Processes a list of .tif image files using a pretrained Cellpose model to 
    generate segmentation masks, saving them as .npy files.

    Args:
        tif_files (list[Path]): Paths to the .tif images to process.
        output_dir (Path): Directory where segmentation masks will be saved.
        model_path (Path or str): Path to the pretrained Cellpose model.
        use_gpu (bool): If True, attempts to use GPU acceleration for processing.
        overwrite (bool, optional): If False (default), skips files that already 
                                     have a corresponding saved mask.
    
    Returns:
        Path: Output path where masks have been saved

    Workflow:
        - Loads the pretrained Cellpose model (GPU if available and requested).
        - Iterates through each .tif file, preserving its relative directory 
          structure inside `output_dir`.
        - For each image:
            - Skips processing if mask exists and `overwrite` is False.
            - Runs the Cellpose model to generate masks.
            - Saves masks as NumPy arrays with `_outlines.npy` suffix.
    """
    if not hasattr(calculate_masks, "_model"):
        if use_gpu:
            print("Trying to use GPU...")
        print("Loading Cellpose model...")
        calculate_masks._model = models.CellposeModel(pretrained_model=model_path, gpu=use_gpu)
    model = calculate_masks._model
    
    if overwrite:
        print("💣 Overwriting masks")
    
    tif_files = list(Path(path_to_tifs).glob("*.tif"))
    for tif_file in tqdm(tif_files, desc=f"🎭 Predicting masks: {output_dir}", file=sys.stdout):
        out_name = tif_file.stem + "_outlines.npy"
        out_path = output_dir / out_name
        if (not overwrite) & out_path.exists():
            continue
        
        img = imread(tif_file)
        masks, _, _ = model.eval(img)

        np.save(out_path, masks)
    return output_dir