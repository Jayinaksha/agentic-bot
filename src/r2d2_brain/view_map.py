import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import json
import sys
import os

# --- Semantic Classes ---
SEMANTIC_CLASSES = {
    "unknown": 0,
    "free": 1,
    "occupied": 2,
    "chair": 3,
    "table": 4,
    "door": 5,
    "wall": 6,
    "person": 7,
    "obstacle": 8
}

# Colors for visualization
SEMANTIC_COLORS = {
    0: [0.7, 0.7, 0.7],  # unknown - gray
    1: [1.0, 1.0, 1.0],  # free - white
    2: [0.0, 0.0, 0.0],  # occupied - black
    3: [0.8, 0.2, 0.2],  # chair - red
    4: [0.2, 0.6, 0.8],  # table - blue
    5: [0.8, 0.8, 0.2],  # door - yellow
    6: [0.4, 0.4, 0.4],  # wall - dark gray
    7: [0.9, 0.6, 0.2],  # person - orange
    8: [0.5, 0.0, 0.5],  # obstacle - purple
}

def view_npz_map(npz_file="semantic_map_1757596278.npz"):
    """Load and visualize a semantic map from an NPZ file"""
    
    # Check if file exists
    if not os.path.exists(npz_file):
        # Try looking in map_data directory
        data_dir_path = os.path.join("map_data", npz_file)
        if os.path.exists(data_dir_path):
            npz_file = data_dir_path
        else:
            print(f"Error: File {npz_file} not found.")
            print("Please make sure the file exists in the current directory or in the map_data/ folder.")
            return
    
    # Load the data
    print(f"Loading map from {npz_file}...")
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Print all available keys in the file
        print(f"Available keys in file: {data.files}")
        
        # Extract arrays and metadata
        occupancy_grid = data["occupancy_grid"]
        semantic_grid = data["semantic_grid"]
        
        # Handle objects differently - check if it's a NumPy array or JSON string
        objects = {}
        if "objects" in data:
            objects_data = data["objects"]
            if isinstance(objects_data, np.ndarray):
                print("Objects stored as NumPy array - will display map without object markers")
                # If it's just a placeholder array, initialize an empty objects dict
                objects = {}
            elif isinstance(objects_data, (str, bytes, bytearray)):
                # If it's a JSON string as originally expected
                objects = json.loads(objects_data)
            else:
                print(f"Objects data type: {type(objects_data)}")
        
        resolution = float(data["resolution"])
        origin_x = float(data["origin_x"])
        origin_y = float(data["origin_y"])
        width = int(data["width"])
        height = int(data["height"])
        
        # Print map information
        print(f"\nMap Information:")
        print(f"  Size: {width}x{height} cells")
        print(f"  Resolution: {resolution} meters per cell")
        print(f"  Origin: ({origin_x}, {origin_y})")
        print(f"  Objects: {len(objects)} detected")
        print(f"  Occupancy grid shape: {occupancy_grid.shape}")
        print(f"  Semantic grid shape: {semantic_grid.shape}")
        
        # Check for non-zero values in the maps
        print(f"  Occupancy grid range: {np.min(occupancy_grid)} to {np.max(occupancy_grid)}")
        print(f"  Semantic grid range: {np.min(semantic_grid)} to {np.max(semantic_grid)}")
        
        # Show unique values in each grid
        print(f"  Unique values in occupancy grid: {np.unique(occupancy_grid)}")
        print(f"  Unique values in semantic grid: {np.unique(semantic_grid)}")
        
        # Create a blended map for visualization
        blended_map = np.copy(occupancy_grid)
        semantic_mask = (semantic_grid > SEMANTIC_CLASSES["occupied"])
        blended_map[semantic_mask] = semantic_grid[semantic_mask]
        
        # Create color map
        max_class_value = max(max(SEMANTIC_CLASSES.values()), 
                              np.max(occupancy_grid) if occupancy_grid.size > 0 else 0,
                              np.max(semantic_grid) if semantic_grid.size > 0 else 0)
        colors = [SEMANTIC_COLORS.get(i, [0,0,0]) for i in range(int(max_class_value) + 1)]
        cmap = ListedColormap(colors)
        
        # Visualize maps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot occupancy grid
        im0 = axes[0].imshow(occupancy_grid, cmap=cmap, origin='lower', vmin=0, vmax=max_class_value)
        axes[0].set_title("Occupancy Grid")
        axes[0].grid(True, alpha=0.3)
        
        # Plot semantic grid
        im1 = axes[1].imshow(semantic_grid, cmap=cmap, origin='lower', vmin=0, vmax=max_class_value)
        axes[1].set_title("Semantic Grid")
        axes[1].grid(True, alpha=0.3)
        
        # Plot blended map
        im2 = axes[2].imshow(blended_map, cmap=cmap, origin='lower', vmin=0, vmax=max_class_value)
        axes[2].set_title("Blended Map")
        axes[2].grid(True, alpha=0.3)
        
        # Mark objects on the blended map
        for obj_id, obj in objects.items():
            if "position" in obj:
                obj_x, obj_y = obj["position"]["x"], obj["position"]["y"]
                obj_i = int((obj_y - origin_y) / resolution)
                obj_j = int((obj_x - origin_x) / resolution)
                
                if 0 <= obj_i < height and 0 <= obj_j < width:
                    axes[2].plot(obj_j, obj_i, 'o', color='yellow', markersize=8)
                    
                    # Add label
                    axes[2].text(obj_j, obj_i-5, obj["name"], color='black', 
                             fontsize=8, ha='center', backgroundcolor='white', alpha=0.7)
        
        # Add legend
        legend_elements = []
        for name, value in SEMANTIC_CLASSES.items():
            if value <= max_class_value:
                legend_elements.append(patches.Patch(color=colors[value], label=name))
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements))
        
        # Add colorbar
        fig.colorbar(im2, ax=axes[2], orientation='vertical', shrink=0.8)
        
        # Save figure to file
        plt.savefig("semantic_map_visualization.png")
        print("Saved visualization to semantic_map_visualization.png")
        
        # Show the plot
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the legend
        plt.show()
        
        # List all detected objects
        if objects:
            print("\nDetected Objects:")
            for i, (obj_id, obj) in enumerate(objects.items()):
                pos_str = ""
                if "position" in obj:
                    pos_str = f"at ({obj['position']['x']:.2f}, {obj['position']['y']:.2f})"
                print(f"  {i+1}. {obj.get('name', 'Unknown')} {pos_str}: {obj.get('description', '')}")
        
    except Exception as e:
        print(f"Error loading map: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_npz_map(sys.argv[1])
    else:
        # Use the specific file by default
        view_npz_map()