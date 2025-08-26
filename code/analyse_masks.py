import argparse
import traceback
from pathlib import Path
from functions.Miscellaneous import create_log
from functions.DIC_analysis import track_cells_and_get_elongations, create_data_entries

def main(path_to_masks: Path, path_to_tifs: Path, top_con: float, bottom_con: float, output_path: Path, buffer: int = 50, max_disp: int = 100, gaps: int = 5, overwrite: bool = False):
    tracks = track_cells_and_get_elongations(path_to_masks, buffer, max_disp, gaps, overwrite)
    create_data_entries(top_con, bottom_con, tracks, path_to_tifs, output_path, flip=True)
    print("✅ Done!\n")    

parser = argparse.ArgumentParser()
parser.add_argument("--buffer", help="File to Masks", default=50)
parser.add_argument("--max_disp", help="File to Masks", default=100)
parser.add_argument("--gaps", help="File to Masks", default=5)
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs", default=False)
args = parser.parse_args()

create_log("trackcells", Path("../logs/"))

mask_root_dir = Path("../masks/")
data_root_dir = Path("../data/")
db_root_dir = Path("../tracks/")
mask_directiories = [
    "../data/0-0_wt-19_08_2025/"   
]

for directory in mask_directiories:    
    print(f"🌲 Analysing root folder: {directory}")
    for folder in Path(directory).rglob("*"):
        try:
            if not folder.is_dir():
                continue

            npy_files = list(folder.glob("*.npy"))
            if not len(npy_files):
                continue

            print(f"📂 Processing folder: {folder}")
            relative_path = folder.parent.relative_to(mask_root_dir)
            path_to_tifs = data_root_dir / relative_path / folder.name
            output_path = db_root_dir / relative_path

            with open(folder.parent / "concentrations.txt", "r") as f:
                line = f.readline().strip() 
                bottom_con, top_con = map(float, line.split(',')) 

            main(folder, path_to_tifs, top_con, bottom_con, output_path, args.buffer, args.max_disp, args.gaps, args.overwrite)
        except Exception as e:
            traceback.print_exc()  # prints full traceback
        print("\n")
    print("\n")

# python .\analyse_masks.py
