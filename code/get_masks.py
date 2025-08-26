import argparse
from pathlib import Path
from functions.CellPose_analysis import calculate_masks
from functions.Miscellaneous import create_log

def main(tif_files: list[Path], output_dir: Path, model_path: Path, use_gpu: bool, overwrite: bool = False) -> None:
    calculate_masks(tif_files, output_dir, model_path, use_gpu, overwrite)
    print("✅ Done!")

parser = argparse.ArgumentParser(description="Run Cellpose model on TIFF images")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained Cellpose model")
parser.add_argument("--use_gpu", action="store_true", default=False, help="Use GPU if available")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs", default=False)
args = parser.parse_args()

create_log("process", Path("../logs"))

tiff_directiories = [
    "../data/0-0-oil_25_07_2025/",
    "../data/500-500-oil_23_07_2025/",
    "../data/0-50-oil_17_07_2025/"
]

for directory in tiff_directiories:
    print(f"Analysing folder: {directory.split('/')[-2]}")
    for folder in Path(directory).rglob("*"):
        if (not folder.is_dir()) or ("10x" in folder.parent.name):
            continue

        tif_files = list(folder.glob("*.tif"))
        if not len(tif_files):
            continue

        relative_path = folder.relative_to(args.input_dir)
        output_subdir = args.output_dir / relative_path
        output_subdir.mkdir(parents=True, exist_ok=True)

        main(folder, output_subdir, Path(args.model_path), args.use_gpu, args.overwrite)
        print("\n")


# python .\code\process_data_directory.py --input_dir "..\data" --output_dir "..\masks" --model_path "..\vSAM2" --use_gpu