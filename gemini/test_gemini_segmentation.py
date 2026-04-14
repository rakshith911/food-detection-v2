"""
Test script to get segmentation masks from Gemini API for food images.

This uses Gemini's segmentation mask capabilities to get precise masks
for each food item/ingredient, not just bounding boxes.
"""

try:
    from google import genai
    from google.genai import types
    NEW_GENAI_AVAILABLE = True
except ImportError:
    NEW_GENAI_AVAILABLE = False
    print("⚠ google-genai package not found.")
    print("   Install with: pip install google-genai")
    print("   Or: pip install -r requirements.txt")

from PIL import Image, ImageDraw, ImageColor
import io
import base64
import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional


def get_api_key() -> Optional[str]:
    """Get Gemini API key from environment or file."""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        # Try reading from FoodAI test file
        test_file = Path(__file__).parent.parent / "FoodAI" / "nutrition-video-analysis" / "terraform" / "docker" / "TEST_OPTIMIZATIONS.md"
        if test_file.exists():
            try:
                content = test_file.read_text()
                for line in content.split('\n'):
                    if 'GEMINI_API_KEY=' in line and 'export' in line:
                        api_key = line.split('"')[1] if '"' in line else line.split("'")[1]
                        break
            except Exception as e:
                print(f"⚠ Could not read API key from file: {e}")
    
    return api_key


def parse_json(json_output: str):
    """Parse JSON from markdown code blocks."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            output = json_output.split("```")[0]  # Remove everything after the closing "```"
            return output
    # If no markdown fencing, try to parse directly
    return json_output


def extract_segmentation_masks(
    image_path: str,
    output_dir: str = "segmentation_outputs",
    prompt: Optional[str] = None,
    model: str = "gemini-3-flash-preview"
):
    """
    Extract segmentation masks for food items from an image using Gemini.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save segmentation outputs
        prompt: Custom prompt (default: detects all food items)
        model: Gemini model to use
    """
    if not NEW_GENAI_AVAILABLE:
        raise ImportError(
            "google-genai package is required for segmentation masks.\n"
            "Install with: pip install google-genai\n"
            "Or: pip install -r requirements.txt"
        )
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Please:\n"
            "1. Set GEMINI_API_KEY environment variable, or\n"
            "2. Ensure TEST_OPTIMIZATIONS.md file exists with the key"
        )
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Load and resize image
    im = Image.open(image_path)
    original_size = im.size
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    resized_size = im.size
    
    print(f"📸 Image: {image_path}")
    print(f"   Original size: {original_size}")
    print(f"   Resized size: {resized_size}")
    
    # Default prompt for food detection
    if prompt is None:
        prompt = """
        Analyze this food image and provide segmentation masks for all visible food items and ingredients.
        Output a JSON list of segmentation masks where each entry contains:
        - "box_2d": [y_min, x_min, y_max, x_max] normalized coordinates (0-1000)
        - "mask": base64-encoded PNG segmentation mask
        - "label": descriptive label for the food item/ingredient
        
        Include all visible food items, ingredients, garnishes, sides, sauces, and components.
        Use descriptive labels that identify what each item is.
        """
    
    config = types.GenerateContentConfig(
        top_p=1,
        top_k=1,
        seed=42,
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # Better for object detection
        # Add timeout settings
    )
    
    print(f"\n🤖 Sending to Gemini API...")
    print(f"   Model: {model}")
    print(f"   This may take 30-60 seconds...")
    print(f"   (Press Ctrl+C if it takes too long)")
    
    # Try different models if the requested one fails
    models_to_try = [
        model,  # Try requested model first
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    
    response = None
    used_model = None
    
    for model_name in models_to_try:
        try:
            print(f"   Attempting with model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, im],  # Pillow images can be directly passed
                config=config
            )
            used_model = model_name
            print(f"   ✓ Received response from {model_name}")
            break
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                print(f"   ⚠ Model {model_name} not available, trying next...")
                continue
            else:
                # Other error, raise it
                raise
    
    if response is None:
        # List available models for user
        print(f"\n⚠ None of the tried models worked.")
        print(f"   Tried: {', '.join(models_to_try)}")
        try:
            models = client.models.list()
            available = []
            for m in models:
                if hasattr(m, 'name'):
                    available.append(m.name)
            if available:
                print(f"\n📋 Available models:")
                for m in available[:10]:  # Show first 10
                    print(f"   - {m}")
                print(f"\n   Try: python test_gemini_segmentation.py --image <image> --model <model_name>")
        except:
            pass
        raise ValueError("No available model found. Check your API key and model availability.")
    
    # Process the response
    print(f"✓ Received response from Gemini")
    print(f"   Response length: {len(response.text)} characters")
    
    # Parse JSON response
    json_str = parse_json(response.text)
    items = json.loads(json_str)
    
    print(f"\n📦 Found {len(items)} segmentation masks")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create annotated image with all masks
    annotated_image = im.convert('RGBA')
    colors_list = list(ImageColor.colormap.keys())
    
    # Process each mask
    for i, item in enumerate(items):
            label = item.get("label", f"Item_{i}")
            box = item.get("box_2d", [])
            
            if len(box) != 4:
                print(f"⚠ Skipping {label}: invalid box_2d format")
                continue
            
            # Get bounding box coordinates (normalized by 1000)
            y0 = int(box[0] / 1000 * resized_size[1])
            x0 = int(box[1] / 1000 * resized_size[0])
            y1 = int(box[2] / 1000 * resized_size[1])
            x1 = int(box[3] / 1000 * resized_size[0])
            
            # Skip invalid boxes
            if y0 >= y1 or x0 >= x1:
                print(f"⚠ Skipping {label}: invalid box coordinates")
                continue
            
            print(f"\n   [{i+1}] {label}")
            print(f"       Box: [{x0}, {y0}, {x1}, {y1}]")
            
            # Process mask if available
            mask = None
            if "mask" in item:
                png_str = item["mask"]
                if png_str.startswith("data:image/png;base64,"):
                    # Remove prefix
                    png_str = png_str.removeprefix("data:image/png;base64,")
                    mask_data = base64.b64decode(png_str)
                    mask = Image.open(io.BytesIO(mask_data))
                    
                    # Resize mask to match bounding box
                    mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
                    print(f"       Mask size: {mask.size}")
                else:
                    print(f"       ⚠ Invalid mask format")
            
            # Create overlay for this mask
            overlay = Image.new('RGBA', resized_size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Get color for this item
            color_name = colors_list[i % len(colors_list)]
            rgb_color = ImageColor.getrgb(color_name)
            
            # Draw bounding box
            overlay_draw.rectangle([x0, y0, x1, y1], outline=rgb_color, width=3)
            
            # Draw mask overlay if available
            if mask:
                mask_array = np.array(mask.convert('L'))  # Convert to grayscale
                overlay_color = (*rgb_color, 100)  # Semi-transparent
                
                for y in range(y0, min(y1, resized_size[1])):
                    for x in range(x0, min(x1, resized_size[0])):
                        mask_y = y - y0
                        mask_x = x - x0
                        if 0 <= mask_y < mask_array.shape[0] and 0 <= mask_x < mask_array.shape[1]:
                            if mask_array[mask_y, mask_x] > 128:  # Threshold for mask
                                overlay_draw.point((x, y), fill=overlay_color)
            
            # Draw label
            label_text = label
            try:
                font = ImageDraw.ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            except:
                font = ImageDraw.ImageFont.load_default()
            
            # Get text size
            bbox_text = overlay_draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Draw label background
            label_bg = [x0, y0 - text_height - 4, x0 + text_width + 8, y0]
            overlay_draw.rectangle(label_bg, fill=(255, 255, 255))
            
            # Draw label text
            overlay_draw.text((x0 + 4, y0 - text_height - 2), label_text, fill=(0, 0, 0), font=font)
            
            # Composite overlay onto annotated image
            annotated_image = Image.alpha_composite(annotated_image, overlay)
            
            # Save individual mask if available
            if mask:
                mask_filename = f"{label.replace(' ', '_')}_{i}_mask.png"
                mask_path = os.path.join(output_dir, mask_filename)
                mask.save(mask_path)
                print(f"       💾 Saved mask: {mask_filename}")
    
    # Save annotated image
    annotated_filename = f"{Path(image_path).stem}_segmented.png"
    annotated_path = os.path.join(output_dir, annotated_filename)
    annotated_image.convert('RGB').save(annotated_path)
    print(f"\n💾 Annotated image saved to: {annotated_path}")
    
    # Save JSON results
    json_filename = f"{Path(image_path).stem}_masks.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(items, f, indent=2)
    print(f"💾 JSON results saved to: {json_path}")
    
    return {
        "masks": items,
        "annotated_image": annotated_path,
        "json_output": json_path
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract segmentation masks from food images using Gemini")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="segmentation_outputs",
        help="Directory to save outputs (default: segmentation_outputs)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (default: detects all food items)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use. Will try alternatives if not available."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("=" * 60)
        print("Available Gemini Models")
        print("=" * 60)
        api_key = get_api_key()
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                models = client.models.list()
                print("\nAvailable models:")
                for m in models:
                    if hasattr(m, 'name'):
                        print(f"  - {m.name}")
            except Exception as e:
                print(f"⚠ Error listing models: {e}")
        else:
            print("\n⚠ No API key found. Cannot list models.")
        sys.exit(0)
    
    print("=" * 60)
    print("Gemini Food Image Segmentation")
    print("=" * 60)
    
    extract_segmentation_masks(
        image_path=args.image,
        output_dir=args.output_dir,
        prompt=args.prompt,
        model=args.model
    )
    
    print("\n✓ Segmentation complete!")
