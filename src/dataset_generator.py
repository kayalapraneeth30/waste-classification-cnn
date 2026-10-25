import os
import numpy as np
from pathlib import Path
from src.config import DATA_DIR, CLASSES

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def create_sample_waste_dataset(output_dir: Path = DATA_DIR / "sample_images", samples_per_class: int = 5):
    """
    Synthesizes sample benchmark images for each waste category:
    - Plastic (blue/red bottles with specular highlights)
    - Paper (white/beige fibrous rectangular sheets and cardboard)
    - Glass (teal/transparent containers with refractive reflections)
    - Metal (silver/metallic cans and foil textures)
    - Organic (green/brown fruit and vegetable shapes)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for category in CLASSES:
        cat_dir = output_dir / category.lower()
        cat_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            width, height = 256, 256
            
            # Base background (neutral tabletop or surface)
            bg_tone = np.random.randint(220, 245)
            arr = np.full((height, width, 3), bg_tone, dtype=np.uint8)

            if category == "Plastic":
                # Colored plastic bottle shape (e.g. translucent blue or crimson)
                color = (np.random.randint(20, 60), np.random.randint(100, 180), np.random.randint(200, 255))
                # Add central cylinder
                y_start, y_end = 40, 210
                x_start, x_end = 80, 176
                arr[y_start:y_end, x_start:x_end] = color
                # Add specular highlight strip
                arr[y_start+10:y_end-10, x_start+15:x_start+25] = (240, 248, 255)

            elif category == "Paper":
                # Cardboard/paper sheet with subtle fibrous grain
                paper_color = (np.random.randint(230, 250), np.random.randint(220, 240), np.random.randint(200, 220))
                arr[50:200, 50:206] = paper_color
                # Add crumpled line artifacts
                noise = np.random.randint(-15, 15, (150, 156, 3))
                arr[50:200, 50:206] = np.clip(arr[50:200, 50:206].astype(int) + noise, 0, 255).astype(np.uint8)

            elif category == "Glass":
                # Clear teal/cyan glass jar silhouette with high reflection
                glass_tint = (160, 220, 215)
                arr[45:215, 75:181] = glass_tint
                # Refraction edge borders
                arr[45:215, 75:82] = (90, 150, 145)
                arr[45:215, 174:181] = (90, 150, 145)
                # Specular white streaks
                arr[55:205, 95:102] = (255, 255, 255)

            elif category == "Metal":
                # Metallic silver soda can with cylindrical shading
                for col in range(80, 176):
                    intensity = int(140 + 95 * np.sin((col - 80) / 96.0 * np.pi))
                    arr[50:206, col] = (intensity, intensity, intensity + 5)
                # Circular metallic lid
                arr[45:55, 85:171] = (190, 190, 195)

            elif category == "Organic":
                # Green/yellow apple or banana fruit shape
                center_y, center_x = 128, 128
                radius = 65
                y_grid, x_grid = np.ogrid[:height, :width]
                dist_from_center = (x_grid - center_x)**2 + (y_grid - center_y)**2
                mask = dist_from_center <= radius**2
                arr[mask] = (np.random.randint(40, 90), np.random.randint(140, 200), np.random.randint(30, 70))
                # Add stem
                arr[50:68, 125:131] = (80, 50, 20)

            # Save image
            img_path = cat_dir / f"{category.lower()}_sample_{i+1}.jpg"
            if PIL_AVAILABLE:
                img = Image.fromarray(arr)
                img.save(img_path, quality=92)
            generated_files.append(img_path)

    print(f"[Dataset] Generated {len(generated_files)} synthetic sample waste images across 5 classes.")
    return generated_files

if __name__ == "__main__":
    create_sample_waste_dataset()
