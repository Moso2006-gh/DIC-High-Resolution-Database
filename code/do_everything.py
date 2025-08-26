import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import traceback
from pathlib import Path
from functions.Miscellaneous import create_log
from functions.CellPose_analysis import calculate_masks
from functions.DIC_analysis import track_cells_and_get_elongations, create_data_entries

# buffer refers to the edge width that we are neglecting while extracting the outlines
# max_disp is the maximum distance a cell can travel in one frame interval(in pixels)
# gaps is the number of missing frames
# top_con and bottom_con refer to the max and min concentrations in the chamber. 
# For running all positions at once, make sure the concentrations.txt file is inside each position
# radius is the half length of each entry in the dataset (in pixels)
# flip is true if concentration at the top of each image in the dataset is higher than the bottom

def main(path_to_tifs: Path, path_to_masks: Path, path_to_db: Path, overwrite: bool = False,
         model_path: Path = "../vSAM2", use_gpu: bool = True,  
         buffer: int = 50, max_disp: int = 100, gaps: int = 5, 
         top_con: float = 0, bottom_con: float = 0, radius: int = 200, flip: bool = True):
    
    calculate_masks(path_to_tifs, path_to_masks, model_path, use_gpu, overwrite) # step 1: making the mask
    tracks = track_cells_and_get_elongations(path_to_masks, buffer, max_disp, gaps, overwrite) # step2: cell tracking
    create_data_entries(top_con, bottom_con, tracks, path_to_tifs, path_to_db, radius, flip, overwrite)
    print("✅ Done!")    

create_log("trackcells", Path("../logs/"))

mask_root_dir = Path("../masks/")
data_root_dir = Path("../data/")
db_root_dir = Path("../tracks/")
tiff_directiories = [
    "../data/0-0_wt-19_08_2025/"   
]

for directory in tiff_directiories:
    print(f"🌲 Analysing root folder: {directory}")
    for folder in Path(directory).rglob("*"):
        try:
            if (not folder.is_dir()) or ("10x" in folder.parent.name):
                continue

            tif_files = list(folder.glob("*.tif"))
            if not len(tif_files):
                continue

            print(f"📂 Processing folder: {folder}")

            relative_path = folder.relative_to(data_root_dir)
            
            mask_subdir = mask_root_dir / relative_path
            mask_subdir.mkdir(parents=True, exist_ok=True)
            
            db_subdir = (db_root_dir / relative_path).parent
            db_subdir.mkdir(parents=True, exist_ok=True)
            
            if (folder.parent / "concentrations.txt").exists():
                with open(folder.parent / "concentrations.txt", "r") as f:
                    line = f.readline().strip() 
                    bottom_con, top_con = map(float, line.split(','))
            elif (mask_subdir.parent / "concentrations.txt").exists():
                with open(mask_subdir.parent  / "concentrations.txt", "r") as f:
                    line = f.readline().strip() 
                    bottom_con, top_con = map(float, line.split(','))
            else:
                print("❌ Mising concentrations cant analyse folder... \n")
                continue 

            main(folder, mask_subdir, db_subdir, bottom_con=bottom_con, top_con=top_con)
        except Exception:
            print(f"😵 Error when analysing {folder}:")
            traceback.print_exc()
        print("\n")
    print("\n")

# python .\do_everything.py