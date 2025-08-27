import sys
import inspect
import functools
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from typing import List, Callable
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from .Datastructures import Cell, Track, Track_Info_Point, Cell_Shape
from .Miscellaneous import load_track_from_db, get_tif_files, flip_track, flip_tracks
from .DIC_analysis import get_cells_per_frame_from_tracks, get_shape_theta, get_track_velocities

# Helper
def gif(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Bind args/kwargs to the function's signature
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Extract by name (works regardless of positional/keyword call)
        file_name = bound.arguments["file_name"]
        output_path = bound.arguments["output_path"]

        # Call the original function
        images = func(*args, **kwargs)

        # Build path and save
        output_file = Path(output_path) / f"{file_name}.gif"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        imageio.mimsave(output_file, images)

        print(f"🎬 Saved GIF to {output_file}")
        return output_file
    return wrapper

def create_gif_from_tracks_path(func: Callable, path_to_tracks: Path, file_name: str, output_path: Path, root_masks: Path = Path("../masks/"), background: bool = False, root_tifs: Path = Path("../data/"), flip: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    tracks = np.load(Path(path_to_tracks), allow_pickle=True)
    relative = (Path(path_to_tracks).relative_to(Path(root_masks))).parent
    
    if background:
        path_to_tifs = Path(root_tifs) / relative / "Default"
    else:
        path_to_tifs = "."

    return func(tracks, file_name, output_path, path_to_tifs, background, flip, cap, frame, dpi)

def plot_frame(image: np.ndarray, dpi: int) -> None:
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    return


# Mask
def create_mask_frame(mask: Path, frame: int, tif_file: Path =".", background: bool = False, mask_alpha: float = 0.85, dpi: int = 100) -> np.ndarray:
    """
    Create a single RGB image with a labeled mask overlay.

    This function generates a visualization of a mask for a given frame.
    It optionally overlays a grayscale background image from a TIFF file 
    and adjusts transparency of the mask using an alpha channel.

    Args:
        mask (Path): Path to the `.npy` file containing the mask data.
        frame (int): Frame number to be shown in the plot title.
        tif_file (Path, optional): Path to the corresponding TIFF file for the background. Default is current directory.
        background (bool, optional): Whether to overlay the background image. Default is False.
        mask_alpha (float, optional): Transparency level of the mask overlay. Default is 0.85.

    Returns:
        np.ndarray: RGB image (as NumPy array) of the rendered mask visualization.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.set_title(f"Frame {frame}")
    
    mask_data = np.load(mask, allow_pickle=True)

    if background:
        background_img = imageio.imread(tif_file)
        ax.imshow(
            background_img,
            cmap='gray',
            origin='lower',
            vmin=850,
            vmax=2000
        )

    alpha_layer = np.zeros_like(mask_data, dtype=float)
    alpha_layer[mask_data > 0] = mask_alpha

    ax.imshow(
        mask_data,
        cmap="tab20",
        alpha=alpha_layer
    )
    
    ax.set_xlim([0, mask_data.shape[1]])
    ax.set_ylim([mask_data.shape[0], 0]) 
    
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((height, width, 4)) 
    image = image[:,:,:3] 
    plt.close(fig)
    
    return image

@gif
def create_mask_gif(path_to_masks: Path, file_name: str, output_path: Path, path_to_tifs: Path = Path("."), background: bool = False, mask_alpha: int = 0.85, cap: int = 9000, frame: int = None, dpi: int = 100) -> List[np.ndarray]:
    """
    Create a GIF visualizing mask overlays across multiple frames.

    This function generates a GIF from a series of `.npy` mask files, with optional
    background images from TIFF files. You can visualize a single frame or create a 
    full animation, limited by an optional frame cap.

    Args:
        path_to_masks (Path or str): Directory containing `.npy` mask files.
        file_name (str): Output name for the generated GIF (without extension).
        output_path (Path or str): Directory where the GIF will be saved.
        path_to_tifs (Path, optional): Directory containing background TIFF files. Default is current directory.
        background (bool, optional): Whether to overlay background images. Default is False.
        mask_alpha (float, optional): Transparency level for the mask overlays. Default is 0.85.
        frame (int, optional): If specified, only this frame is displayed instead of creating a GIF.
        cap (int, optional): Maximum number of frames to include in the GIF. Default is 900.

    Returns:
        None
    """
    masks = sorted(Path(path_to_masks).glob("*.npy"))
    if background:
        tif_files = sorted(Path(path_to_tifs).glob("*.tif"))
    else:
        tif_files = np.zeros(len(masks))
        
    if frame is not None:
        image = create_mask_frame(masks[frame], frame, tif_files[frame], background, mask_alpha, dpi)
        plot_frame(image, dpi)
        return [image]
    
    images = []
    for frame, mask in tqdm(enumerate(masks[:cap]), desc="🎬 Creating GIF", file=sys.stdout, total=min(len(masks), cap)):        
        image = create_mask_frame(mask, frame, tif_files[frame], background, mask_alpha, dpi)
        images.append(image)

    return images


# Track from raw tiff
def create_track_frame_from_raw_tiffs(track: Track, frame: int, background: bool, tif_file: Path, vmin: int, vmax: int, y_min: int, y_max: int, x_min: int, x_max: int, dpi: int = 100) -> np.ndarray:
    position = track[frame]
    cell = position["cell"]

    centroid = cell["centroid"]
    outline = cell["shape"] + centroid

    fig, ax = plt.subplots(figsize=(6,6), dpi=dpi)
    ax.set_title(f"Frame {position["frame"]}")
    
    ax.scatter(outline[:, 1], outline[:, 0], s=4)
    
    centroids = np.vstack([position["cell"]["centroid"] for position in track[:frame]])
    ax.plot(centroids[:, 1], centroids[:, 0], color="red")
    ax.scatter(centroid[1], centroid[0], c='red', s=4, label='Centroid')
    
    if background:
        background_img = imageio.imread(tif_file)
        cropped = background_img[y_min:y_max, x_min:x_max]
        ax.imshow(
            cropped,
            cmap='gray',
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            vmin=vmin,
            vmax=vmax
        )

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((height, width, 4)) 
    image = image[:,:,:3] 
    plt.close(fig)
    
    return image

@gif
def create_track_gif_from_raw_tiffs(track: Track, file_name: str, output_path: Path, margin: int = 20, path_to_tifs: Path = Path("."), background: bool = False, flip: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    """
    Create a GIF visualizing a single cell track.

    This function generates a GIF showing the shape and centroid of a cell across frames.
    Optionally, it can overlay the corresponding background images from TIFF files. The GIF
    is cropped dynamically based on the cell’s positions with an added margin, ensuring all
    frames have consistent dimensions.

    Args:
        track (Track): A list of cell positions for the track.
        file_name (str): Name of the output GIF file (without extension).
        output_path (Path): Folder where the GIF will be saved.
        margin (int, optional): Extra pixels around the cell for cropping. Default is 20.
        path_to_tifs (Path, optional): Path to the folder containing TIFF images. Default is current directory.
        background (bool, optional): Whether to overlay background images. Default is False.
        cap (int, optional): Cap the max amount of frames of the GIF.
        frame (int, optional): If provided, only display a single frame instead of creating a full GIF.

    Returns:
        None
    """
    tif_files, vmin, vmax = get_tif_files(path_to_tifs, background, len(track[:cap]))
    if flip:
        track = flip_track(track)
    
    all_y = np.concatenate([position["cell"]["shape"][:, 0] + position["cell"]["centroid"][0] for position in track[:cap]])
    all_x = np.concatenate([position["cell"]["shape"][:, 1] + position["cell"]["centroid"][1] for position in track[:cap]])

    y_min, y_max = int(all_y.min() - margin), int(all_y.max() + margin)
    x_min, x_max = int(all_x.min() - margin), int(all_x.max() + margin)

    if frame is not None:
        image = create_track_frame_from_raw_tiffs(track, frame, background, tif_files[frame], vmin, vmax, y_min, y_max, x_min, x_max, dpi)
        plot_frame(image, dpi)
        return [image]

    images = []
    for frame in tqdm(range(min(cap, len(track)), desc="🎬 Creating GIF", file=sys.stdout)):
        image = create_track_frame_from_raw_tiffs(track, frame, background, tif_files[frame], vmin, vmax, y_min, y_max, x_min, x_max, dpi)
        images.append(image)

    return images


# Track from db entry
def create_track_frame_from_db_entry(position: Track_Info_Point, tif_file: Path, background: bool, vmin: int, vmax: int, dpi: int = 100):
    frame = position["frame"]
    cell = position["cell"]

    fig, ax = plt.subplots(figsize=(6,6), dpi=dpi)
    ax.set_title(f"Frame {frame}")
    
    ax.scatter(cell["shape"][:, 1], cell["shape"][:, 0], s=4)
    if background:
        background_img = imageio.imread(tif_file)
        height, width = background_img.shape[:2]

        ax.imshow(
            background_img,
            cmap='gray',
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            extent=[-width/2, width/2, -height/2, height/2]
        )

    ax.invert_yaxis()
    
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((height, width, 4)) 
    image = image[:,:,:3] 
    plt.close(fig)
    
    return image

@gif
def create_track_gif_from_db_entry(entry_path: Path, file_name: str, output_path: Path, background: bool, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    entry_path = Path(entry_path)
    output_path = Path(output_path)

    track = load_track_from_db(entry_path)
    tif_files, vmin, vmax = get_tif_files((entry_path / "tiffs"), len(track[:cap]))

    if frame is not None:
        image = create_track_frame_from_db_entry(track[frame], tif_files[frame], background, vmin, vmax, dpi)
        plot_frame(image)
        return [image]
    
    images = []
    for p, pos in tqdm(enumerate(track[:cap]), desc="🎬 Creating GIF", total=len(track[:cap])):
        image = create_track_frame_from_db_entry(pos, tif_files[p], background, vmin, vmax, dpi)
        images.append(image)
        
    return images
  
   
# Outlines     
def create_outlines_frame(cells: List[Cell], frame_index: int, tif_file: Path, background: bool, vmin: int, vmax: int, dpi: int = 100) -> np.ndarray:
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_title(f"Frame {frame_index}")

    if background:
        background_img = imageio.imread(tif_file)
        ax.imshow(
            background_img,
            cmap='gray',
            origin='lower',
            vmin=vmin,
            vmax=vmax
        )

    for cell in cells:
        outline = np.asarray(cell["shape"]) + np.asarray(cell["centroid"])
        if outline.ndim == 2 and outline.shape[1] >= 2:
            ax.scatter(outline[:, 1], outline[:, 0], s=1, c='cyan')
    
    ax.axis('off') 

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((height, width, 4)) 
    image = image[:,:,:3] 
    plt.close(fig) 
    return image

@gif
def create_outlines_gif_from_cells_per_frame(cells_per_frame: List[List[Cell]], file_name: str, output_path: Path, path_to_tifs: Path = Path("."), background: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    """
    Create a GIF visualizing cell outlines for multiple frames.

    This function generates a GIF where each frame shows the outlines of all cells in that frame.
    Optionally, it can overlay the corresponding background images from TIFF files. Each cell's 
    outline is plotted as a scatter plot. The resulting GIF provides a visual summary of cell 
    positions and shapes across frames.

    Args:
        cells_per_frame (List[List[Cell]]): A list of frames, each containing a list or dict of cells with their outlines.
        file_name (str): Name of the output GIF file (without extension).
        output_path (Path): Folder where the GIF will be saved.
        path_to_tifs (Path, optional): Path to the folder containing TIFF images for background. Default is current directory.
        background (bool, optional): Whether to overlay background images in the GIF. Default is False.
        cap (int, optional): Cap the max amount of frames of the GIF.
        frame (int, optional): If provided, only display a single frame instead of creating a full GIF.

    Returns:
        None
    """
    tif_files, vmin, vmax = get_tif_files(path_to_tifs, background, len(cells_per_frame[:cap]))
        
    if frame is not None:
        image = create_outlines_frame(cells_per_frame[frame], frame, tif_files[frame], background, vmin, vmax, dpi)
        plot_frame(image, dpi)
        return [image]

    images = []
    for frame, cells in enumerate(tqdm(cells_per_frame[:cap], desc="🎬 Creating GIF", file=sys.stdout)):
        image = create_outlines_frame(cells, frame, tif_files[frame], background, vmin, vmax, dpi)
        images.append(image)

    return images

def create_outlines_gif_from_tracks(tracks: List[Track], file_name: str, output_path: Path, path_to_tifs: Path = Path("."), background: bool = False, flip: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    """
    Create a GIF visualizing cell outlines for multiple frames.

    This function generates a GIF where each frame shows the outlines of all cells in that frame.
    Optionally, it can overlay the corresponding background images from TIFF files. Each cell's 
    outline is plotted as a scatter plot. The resulting GIF provides a visual summary of cell 
    positions and shapes across frames.

    Args:
        tracks (List[Track]): A list of frames, each containing a list or dict of cells with their outlines.
        file_name (str): Name of the output GIF file (without extension).
        output_path (Path): Folder where the GIF will be saved.
        path_to_tifs (Path, optional): Path to the folder containing TIFF images for background. Default is current directory.
        background (bool, optional): Whether to overlay background images in the GIF. Default is False.
        cap (int, optional): Cap the max amount of frames of the GIF.
        frame (int, optional): If provided, only display a single frame instead of creating a full GIF.

    Returns:
        None
    """
    if flip:
        tracks = flip_tracks(tracks)
        
    cells_per_frame = get_cells_per_frame_from_tracks(tracks)
    create_outlines_gif_from_cells_per_frame(cells_per_frame[:cap], file_name, output_path, path_to_tifs, background, cap, frame, dpi)

def create_outlines_gif_from_path_to_tracks(path_to_tracks: Path, file_name: str, output_path: Path, root_masks: Path = Path("../masks/"), background: bool = False, root_tifs: Path = Path("../data/"), flip: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100):
    create_gif_from_tracks_path(create_outlines_gif_from_tracks, path_to_tracks, file_name, output_path, root_masks, background, root_tifs, flip, cap, frame, dpi)


# All Tracks
def create_all_tracks_frame(tracks: List[Track], colors: List[plt.cm.tab20], max_frame: int, tif_file: Path, background: bool, vmin: int, vmax: int, dpi: int = 100) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6,6), dpi=dpi)
    for t, track in enumerate(tracks):
        color = colors[t]
        centroids = []
        last_outline = []
        for pos in track:
            frame = pos["frame"]
            if frame <= max_frame:
                centroids.append(pos["cell"]["centroid"])
                last_outline = pos["cell"]["shape"] + pos["cell"]["centroid"]
            else:
                break
        if len(centroids) == len(track) or not len(centroids):
            continue
        
        centroids = np.array(centroids)
        ax.plot(centroids[:, 1], centroids[:, 0], color=color)
        ax.scatter(last_outline[:, 1], last_outline[:, 0], s=0.1, color=color)
        
        last_cy, last_cx = centroids[-1]
        ax.text(
            last_cx, last_cy, str(t),
            fontsize=6, color=color,
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.5)
        )
    
    if background:
        background_img = imageio.imread(tif_file)
        ax.imshow(
            background_img,
            cmap='gray',
            origin='lower',
            vmin=vmin,
            vmax=vmax
        )
    ax.set_axis_off()
    ax.invert_yaxis()
    
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((height, width, 4)) 
    image = image[:,:,:3] 
    plt.close(fig)
    return image

@gif
def create_all_tracks_gif_from_tracks(tracks: List[Track], file_name: str, output_path: Path, path_to_tifs: Path = Path("."), background: bool = False, flip: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    """
    Create a GIF visualizing cell tracks for multiple frames.

    This function generates a GIF where each frame shows the tracks of all visible cells up to that point.
    Optionally, it can overlay the corresponding background images from TIFF files. Each cell's 

    Args:
        tracks (List[Track]): A list of frames, each containing a list or dict of cells with their outlines.
        file_name (str): Name of the output GIF file (without extension).
        output_path (Path): Folder where the GIF will be saved.
        path_to_tifs (Path, optional): Path to the folder containing TIFF images for background. Default is current directory.
        background (bool, optional): Whether to overlay background images in the GIF. Default is False.
        cap (int, optional): Cap the max amount of frames of the GIF.
        frame (int, optional): If provided, only display a single frame instead of creating a full GIF.
    """
    max_frame = max([pos["frame"] for track in tracks for pos in track])
    tif_files, vmin, vmax = get_tif_files(path_to_tifs, background, min(max_frame, cap))
    if flip:
        tracks = flip_tracks(tracks)
        
    colors = [plt.cm.tab20(i % 20) for i in range(len(tracks))]
    
    if frame is not None:
        image = create_all_tracks_frame(tracks, colors, frame, tif_files[frame], background, vmin, vmax, dpi)
        plot_frame(image, dpi)
        return [image]

    images = []
    for frame in tqdm(range(min(max_frame, cap)), desc="🎬 Creating GIF", file=sys.stdout):
        image = create_all_tracks_frame(tracks, colors, frame, tif_files[frame], background, vmin, vmax, dpi)
        images.append(image)
        
    return images

def create_all_tracks_gif_from_path_to_tracks(path_to_tracks: Path, file_name: str, output_path: Path, root_masks: Path = Path("../masks/"), background: bool = False, root_tifs: Path = Path("../data/"), flip: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100):
    create_gif_from_tracks_path(create_all_tracks_gif_from_tracks, path_to_tracks, file_name, output_path, root_masks, background, root_tifs, flip, cap, frame, dpi)


# Shape evolution
def create_shape_evolution_frame(shape_evolution: List[Cell_Shape], n_points: int, num_frames: int, frame_index: int, colors: plt.cm.hsv, dpi: int = 100) -> np.ndarray:
    fig, ax = plt.subplots(dpi=dpi)

    for point in range(n_points):
        start_frame = max(0, frame_index - 20)
        x_vals = shape_evolution[start_frame:frame_index, point, 1]
        y_vals = shape_evolution[start_frame:frame_index, point, 0]
        
        alphas = np.linspace(0.1, 1, len(x_vals))
        
        for i in range(len(x_vals) - 1):
            ax.plot(
                x_vals[i:i+2],  
                y_vals[i:i+2],
                alpha=alphas[i],
                marker='o',
                markersize=2,
                color=colors[i],
                zorder=i
            )


    ax.set_title(f"Evolution of shape (Frame {frame_index})")
    ax.set_xlim(-75, 75)
    ax.set_ylim(-125, 125)
    ax.set_aspect("equal", adjustable="box") 
    ax.set_title(f"Evolution of shape (Frame {frame_index})")
    ax.invert_yaxis()


    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((height, width, 4)) 
    image = image[:,:,:3] 
    plt.close(fig) 

    return image

@gif
def create_shape_evolution_gif(shape_evolution: List[Cell_Shape], file_name: str, output_path: Path, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    """
    Create a GIF showing the temporal evolution of a cell shape.

    This function visualizes how the shape of a cell changes over time. Each frame in the GIF 
    represents the cell shape at that time point, with points along the outline connected in order.
    Fading lines indicate the progression of each point along the shape over previous frames, 
    providing a visual sense of movement and deformation.

    Args:
        shape_evolution (List[Cell_Shape]): A 3D array of shape (num_frames, n_points, 2) representing the coordinates of cell outline points over time.
        file_name (str): Name of the output GIF file (without extension).
        output_path (Path): Folder where the GIF will be saved.
        cap (int, optional): Cap the max amount of frames of the GIF.
        frame (int, optional): If provided, only display a single frame instead of creating a full GIF.
        dpi (int, optional): Density of pixels

    Returns:
        None
    """
    num_frames, n_points, _ = shape_evolution.shape
    images = []
    
    colors = plt.cm.Blues(np.linspace(0, 1, 20))

    if frame is not None:
        image = create_shape_evolution_frame(shape_evolution, n_points, num_frames, frame, colors, dpi)
        plot_frame(image, dpi)
        return [image]

    for frame_index in tqdm(range(1, min(num_frames, cap)), desc="🎬 Creating GIF", file=sys.stdout):
        image = create_shape_evolution_frame(shape_evolution, n_points, num_frames, frame_index, colors, dpi)
        images.append(image) 

    return images

def create_shape_evolution_gif_from_db_entry(entry: Path, file_name: str, output_path: Path, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    track = load_track_from_db(entry)
    cell_shape_evolution = np.array([pos["cell"]["shape"] for pos in track])
    create_shape_evolution_gif(cell_shape_evolution, file_name, output_path, cap, frame, dpi)


# Shape theta
def create_shape_theta_frame(cell_shape: Cell_Shape, velocity: np.ndarray, south: bool = False, dpi: int = 100) -> np.ndarray:
    """
    Generate a single frame showing cell shape, velocity vector, and orientation (theta).
    
    Args:
        cell_shape (np.ndarray): Array of shape (n_points, 2) representing cell shape.
        velocity (np.ndarray): Velocity vector (vx, vy).
        south (bool): Whether gradient arrow should point south or north.

    Returns:
        np.ndarray: RGB image of the frame.
    """
    theta_deg, direction = get_shape_theta(cell_shape, velocity, south)
    slope = direction[0] / direction[1]

    # Plot
    fig, ax = plt.subplots(dpi=dpi)
    ax.scatter(cell_shape[:, 1], cell_shape[:, 0])

    lim = 120
    ax.plot([-lim, lim], [-slope * lim, slope * lim], linewidth=2, label=f"theta {theta_deg:.2f}", c="Red", linestyle="--")

    # Gradient arrow
    if not south:
        ax.arrow(0, 0, 0, -60, head_width=4, head_length=4, fc='green', ec='green')
    else:
        ax.arrow(0, 0, 0, 60, head_width=4, head_length=4, fc='green', ec='green')

    # Velocity arrow
    ax.arrow(0, 0, velocity[1]*20, velocity[0]*20, head_width=4, head_length=4, 
             fc='blue', ec='blue', label="velocity")
    
    
    ax.arrow(0, 0, direction[1]*20, direction[0]*20, head_width=4, head_length=4, 
             fc='orange', ec='orange', label="velocity", zorder=2)

    # Axes
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')  # keeps aspect 1:1 but respects your limits
    ax.invert_yaxis()  


    # Legend
    custom_lines = [
        Line2D([0], [0], color='red', lw=2, linestyle='--'),
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='orange', lw=2),
        Line2D([0], [0], color='green', lw=2)
    ]
    ax.legend(custom_lines, [f"theta {theta_deg:.2f}°", "velocity", "direction", "gradient"])

    # Convert figure → RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))[:,:,:3]

    plt.close(fig)
    return image

@gif
def create_shape_theta_gif_from_track(track: Track, file_name: str, output_path: Path, south: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    """
    Create a GIF visualizing the orientation (theta) of cell shapes over time.

    This function generates a GIF where each frame visualizes the principal orientation of a cell 
    (theta) computed using SVD. The orientation is shown using arrows and overlays based on cell shape.
    It uses velocity vectors to determine consistent directionality. Optionally, a single frame 
    can be visualized instead of generating the full GIF.

    Args:
        track (Track): A list of frames, each containing a cell with its shape and centroid data.
        file_name (str): Name of the output GIF file (without extension).
        output_path (Path): Directory where the resulting GIF will be saved.
        cap (int, optional): Cap the max amount of frames of the GIF.
        frame (int, optional): If provided, only display a single frame instead of creating a full GIF.

    Returns:
        None
    """
    velocities = get_track_velocities(track)
    if frame is not None:
        image = create_shape_theta_frame(track[frame]["cell"]["shape"], velocities[frame], south, dpi)
        plot_frame(image, dpi)
        return [image]
    
    images = []
    for i, pos in tqdm(enumerate(track[:cap]), desc="🎬 Creating GIF", file=sys.stdout, total=len(track[:cap])):
        image = create_shape_theta_frame(pos["cell"]["shape"], velocities[i], south, dpi)
        images.append(image)
        
    return images

def create_shape_theta_gif_from_db_entry(entry: Path, file_name: str, output_path: Path, south: bool = False, cap: int = 9000, frame: int = None, dpi: int = 100) -> None:
    track = load_track_from_db(entry)
    create_shape_theta_gif_from_track(track, file_name, output_path, south, cap, frame, dpi)
    
def plot_point_trayectory_and_instant_change(cell_shape_evolution: List[Cell_Shape], point_index: int) -> np.ndarray: 
    point_of_interest = cell_shape_evolution[:, point_index] 
    pca = PCA(n_components=1) 
    X_pca = pca.fit_transform(point_of_interest).flatten() 
    X_relative = abs(X_pca - X_pca[0]) 
    dx = np.diff(X_relative, 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10,7))  # 1 row, 2 columns

    # Left plot: Trajectory
    axes[0].plot(np.arange(len(X_relative)), X_relative, color='blue', alpha=0.5, linewidth=2, label='Displacement')
    axes[0].scatter(np.arange(len(X_relative)), X_relative, color='red', s=50, label='Point values')
    axes[0].set_title(f"Trajectory of Point {point_index}", fontsize=14)
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Displacement")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Right plot: Derivative
    axes[1].plot(np.arange(len(dx)), dx, color='green', alpha=0.5, linewidth=2, label='Derivative')
    axes[1].scatter(np.arange(len(dx)), dx, color='orange', s=50, label='Instantaneous change')
    axes[1].set_title(f"Instantaneous Change of Point {point_index}", fontsize=14)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Change")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    return X_relative