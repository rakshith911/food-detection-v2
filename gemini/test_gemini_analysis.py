"""
Test script to send Food-101 images to Gemini API for detailed ingredient and micronutrient analysis.

This will test on a single image first to see what Gemini returns.
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor
import google.generativeai as genai
from typing import Dict, List, Optional
import random

# Try to import the newer genai client for structured output
try:
    from google import genai as genai_new
    from google.genai.types import (
        GenerateContentConfig,
        HarmBlockThreshold,
        HarmCategory,
        HttpOptions,
        Part,
        SafetySetting,
    )
    from pydantic import BaseModel
    NEW_GENAI_AVAILABLE = True
except ImportError:
    NEW_GENAI_AVAILABLE = False
    # Don't print warning here, will check when needed


def list_available_models(api_key: Optional[str] = None):
    """
    List all available Gemini models.
    
    Args:
        api_key: Gemini API key
    
    Returns:
        List of available model names
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("⚠ No API key provided, cannot list models")
        return []
    
    genai.configure(api_key=api_key)
    
    try:
        models = genai.list_models()
        model_names = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                # Extract just the model name (remove 'models/' prefix if present)
                name = m.name
                if name.startswith('models/'):
                    name = name[7:]  # Remove 'models/' prefix
                model_names.append(name)
                # Also keep full name
                model_names.append(m.name)
        return model_names
    except Exception as e:
        print(f"⚠ Could not list models: {e}")
        return []


def setup_gemini(api_key: Optional[str] = None):
    """
    Setup Gemini API with the API key.
    
    Args:
        api_key: Gemini API key. If None, tries to get from environment or file.
    
    Returns:
        Configured Gemini model
    """
    # Try to get API key from various sources
    if not api_key:
        # Try environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # Try reading from FoodAI test file
        if not api_key:
            test_file = Path(__file__).parent.parent / "FoodAI" / "nutrition-video-analysis" / "terraform" / "docker" / "TEST_OPTIMIZATIONS.md"
            if test_file.exists():
                try:
                    content = test_file.read_text()
                    # Extract API key from export statement
                    for line in content.split('\n'):
                        if 'GEMINI_API_KEY=' in line and 'export' in line:
                            api_key = line.split('"')[1] if '"' in line else line.split("'")[1]
                            break
                except Exception as e:
                    print(f"⚠ Could not read API key from file: {e}")
    
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Please:\n"
            "1. Set GEMINI_API_KEY environment variable, or\n"
            "2. Pass api_key parameter, or\n"
            "3. Ensure TEST_OPTIMIZATIONS.md file exists with the key"
        )
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # First, try to list available models to see what's actually available
    try:
        available_models = list_available_models(api_key)
        print(f"📋 Available models: {', '.join(available_models[:5])}..." if len(available_models) > 5 else f"📋 Available models: {', '.join(available_models)}")
    except Exception as e:
        print(f"⚠ Could not list models: {e}")
        available_models = []
    
    # Try different model names (in order of preference)
    # Note: Model names may vary by API version
    model_names_to_try = [
        'gemini-pro-vision',      # Legacy vision model (most stable)
        'gemini-1.5-flash',       # Newer fast model  
        'gemini-1.5-pro',         # Newer capable model
        'gemini-pro',             # Text-only model (won't work for images)
    ]
    
    # Also try with 'models/' prefix (some API versions require this)
    model_names_with_prefix = [f'models/{name}' for name in model_names_to_try if name != 'gemini-pro']
    model_names_to_try = model_names_to_try + model_names_with_prefix
    
    # If we have available models, prioritize those
    if available_models:
        # Add available models to the front of the list
        model_names_to_try = available_models[:3] + model_names_to_try
    
    model = None
    model_name = None
    last_error = None
    
    for name in model_names_to_try:
        try:
            model = genai.GenerativeModel(
                name,
                generation_config={"top_p": 1, "top_k": 1, "seed": 42},
            )
            model_name = name
            print(f"✓ Gemini API configured with {name} (key: {api_key[:10]}...)")
            break
        except Exception as e:
            last_error = e
            continue
    
    if model is None:
        error_msg = f"Could not initialize any Gemini model.\n"
        error_msg += f"Tried: {', '.join(model_names_to_try[:5])}...\n"
        if last_error:
            error_msg += f"Last error: {last_error}\n"
        error_msg += "Please run with --list-models to see available models."
        raise ValueError(error_msg)
    
    return model


def analyze_food_image(
    image_path: Path,
    model,
    include_micronutrients: bool = True,
    include_ingredients: bool = True,
    include_bounding_boxes: bool = True
) -> Dict:
    """
    Analyze a food image with Gemini to get detailed ingredients and micronutrients.
    
    Args:
        image_path: Path to the image file
        model: Gemini model instance
        include_micronutrients: Whether to request micronutrient information
        include_ingredients: Whether to request ingredient breakdown
    
    Returns:
        Dictionary with analysis results
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    print(f"\n📸 Analyzing image: {image_path.name}")
    print(f"   Size: {image.size}")
    print(f"   Mode: {image.mode}")
    
    # Get image dimensions for bounding box normalization
    img_width, img_height = image.size
    
    # Build prompt
    prompt_parts = [
        "Analyze this food image in detail. Provide a comprehensive analysis including:",
        "",
        "1. MAIN DISH/FOOD ITEM:",
        "   - Primary food name",
        "   - Cuisine type",
        "   - Cooking method",
        "",
        "2. VISIBLE INGREDIENTS WITH LOCATIONS:",
        "   - List all visible ingredients/components",
        "   - Include garnishes, sides, sauces, etc.",
        "   - Estimate quantities if possible",
    ]
    
    if include_bounding_boxes:
        prompt_parts.extend([
            "",
            "   IMPORTANT: For each visible food item/ingredient, provide bounding box coordinates.",
            f"   Image dimensions: {img_width} x {img_height} pixels.",
            "   Provide bounding boxes in format: [x_min, y_min, x_max, y_max] where:",
            "   - x_min, y_min: top-left corner coordinates",
            "   - x_max, y_max: bottom-right corner coordinates",
            "   - All coordinates are in pixels (0 to image width/height)",
            "",
        ])
    
    if include_ingredients:
        prompt_parts.extend([
            "3. INGREDIENT BREAKDOWN:",
            "   - Detailed list of all ingredients",
            "   - Hidden ingredients (e.g., in sauces, marinades)",
            "   - Cooking oils/fats used",
            "",
        ])
    
    if include_micronutrients:
        prompt_parts.extend([
            "4. NUTRITIONAL INFORMATION (for a typical serving):",
            "   - Calories",
            "   - Macronutrients (protein, carbs, fats)",
            "   - Micronutrients (vitamins, minerals)",
            "   - Fiber content",
            "",
        ])
    
    prompt_parts.extend([
        "5. ADDITIONAL NOTES:",
        "   - Any allergens present",
        "   - Dietary restrictions (vegetarian, vegan, gluten-free, etc.)",
        "   - Health considerations",
        "",
        "Please format the response as structured JSON with the following keys:",
        "- main_food_item",
        "- cuisine_type",
        "- cooking_method",
        "- visible_ingredients (list of objects, each with: name, bounding_box [x_min, y_min, x_max, y_max], estimated_quantity)",
        "- ingredient_breakdown (detailed list)",
        "- nutritional_info (object with calories, macros, micronutrients)",
        "- allergens (list)",
        "- dietary_tags (list)",
        "- additional_notes",
        "",
        "Example format for visible_ingredients:",
        '[{"name": "pizza slice", "bounding_box": [100, 50, 300, 250], "estimated_quantity": "1 slice"},',
        ' {"name": "olives", "bounding_box": [150, 100, 200, 150], "estimated_quantity": "5-6 olives"}]',
    ])
    
    prompt = "\n".join(prompt_parts)
    
    print("\n🤖 Sending to Gemini API...")
    # Get model name from model object if possible
    model_name_str = getattr(model, '_model_name', 'gemini-1.5-flash')
    print(f"   Model: {model_name_str}")
    
    try:
        # Send image and prompt to Gemini
        response = model.generate_content([prompt, image])
        
        # Get response text
        response_text = response.text
        print(f"\n✓ Received response from Gemini")
        print(f"   Response length: {len(response_text)} characters")
        
        # Try to parse JSON if present
        result = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "raw_response": response_text,
            "parsed": False
        }
        
        # Try to extract JSON from response
        try:
            # Look for JSON block in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                # Try to find JSON object directly
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end] if json_start >= 0 else None
            
            if json_str:
                parsed_data = json.loads(json_str)
                result.update(parsed_data)
                result["parsed"] = True
                print("   ✓ Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            print(f"   ⚠ Could not parse JSON: {e}")
            print("   (Response may be in natural language format)")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error calling Gemini API: {e}")
        raise


# Pydantic model for bounding boxes (for new API)
if NEW_GENAI_AVAILABLE:
    class BoundingBox(BaseModel):
        """Represents a bounding box with normalized coordinates and label."""
        box_2d: list[int]  # [y_min, x_min, y_max, x_max] normalized by 1000
        label: str


def analyze_with_structured_boxes(
    image_path: Path,
    api_key: str
) -> Dict:
    """
    Analyze image using new Google GenAI API with structured bounding box output.
    
    Args:
        image_path: Path to image file
        api_key: Gemini API key
    
    Returns:
        Dictionary with bounding boxes and analysis
    """
    if not NEW_GENAI_AVAILABLE:
        raise ImportError(
            "New Google GenAI client not available.\n"
            "Install with: pip install google-genai"
        )
    
    # Create client
    client = genai_new.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))
    
    # Configure for structured output
    config = GenerateContentConfig(
        system_instruction="""
        Analyze this food image and return bounding boxes for all visible food items and ingredients.
        Return bounding boxes as an array with labels.
        Never return masks. Limit to 25 objects.
        If an object is present multiple times, give each object a unique label
        according to its distinct characteristics (colors, size, position, etc.).
        Include all visible ingredients, garnishes, sides, and components.
        """,
        temperature=0.5,
        top_p=1,
        top_k=1,
        seed=42,
        safety_settings=[
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ],
        response_mime_type="application/json",
        response_schema=list[BoundingBox],
    )
    
    # Read image file as bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Create Part from bytes (for local files)
    # Note: Part.from_bytes might not exist, try Part.from_data or Part inline
    try:
        image_part = Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
    except AttributeError:
        # Alternative: use inline data
        image_part = Part(
            inline_data={
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        )
    
    prompt = """
    Analyze this food image and detect all visible food items and ingredients.
    Output bounding boxes for each item with descriptive labels.
    Include main dishes, sides, garnishes, sauces, and all visible components.
    Coordinates should be normalized [y_min, x_min, y_max, x_max] scaled by 1000.
    """
    
    print(f"\n🤖 Using new Google GenAI API with structured output...")
    print(f"   Model: gemini-2.5-flash")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, prompt],
            config=config,
        )
        
        bounding_boxes = response.parsed if hasattr(response, 'parsed') else []
        
        print(f"✓ Received {len(bounding_boxes)} bounding boxes")
        
        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "bounding_boxes": [bbox.model_dump() for bbox in bounding_boxes],
            "raw_response": response.text if hasattr(response, 'text') else str(response),
            "parsed": True,
            "api_version": "new"
        }
        
    except Exception as e:
        print(f"❌ Error with new API: {e}")
        raise


def plot_bounding_boxes(
    image_path: Path,
    analysis_result: Dict,
    output_path: Optional[Path] = None
) -> Path:
    """
    Plot bounding boxes on the image based on Gemini analysis results.
    
    Args:
        image_path: Path to the original image
        analysis_result: Dictionary with analysis results containing visible_ingredients
        output_path: Path to save annotated image (default: adds _annotated suffix)
    
    Returns:
        Path to saved annotated image
    """
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Try to get a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        try:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font = None
            font_small = None
    
    # Extract bounding boxes from results
    visible_ingredients = []
    
    # Check if using new API format
    if analysis_result.get("api_version") == "new":
        # New API returns bounding_boxes with normalized coordinates
        bounding_boxes = analysis_result.get("bounding_boxes", [])
        for bbox_data in bounding_boxes:
            # Convert from normalized [y_min, x_min, y_max, x_max] * 1000 to pixel coordinates
            box_2d = bbox_data.get("box_2d", [])
            if len(box_2d) == 4:
                y_min_norm, x_min_norm, y_max_norm, x_max_norm = box_2d
                # These are normalized by 1000, so divide by 1000 then multiply by image size
                img_width, img_height = image.size
                x_min = int(x_min_norm / 1000 * img_width)
                y_min = int(y_min_norm / 1000 * img_height)
                x_max = int(x_max_norm / 1000 * img_width)
                y_max = int(y_max_norm / 1000 * img_height)
                
                visible_ingredients.append({
                    "name": bbox_data.get("label", "Unknown"),
                    "bounding_box": [x_min, y_min, x_max, y_max]
                })
    elif analysis_result.get("parsed"):
        # Try to get from parsed data (old API format)
        visible_ingredients = analysis_result.get("visible_ingredients", [])
    else:
        # Try to parse from raw response
        raw_response = analysis_result.get("raw_response", "")
        # Simple extraction - look for JSON-like structures
        import re
        # Look for bounding box patterns in the response
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(bbox_pattern, raw_response)
        if matches:
            # Try to extract ingredient names near bounding boxes
            for i, match in enumerate(matches):
                x_min, y_min, x_max, y_max = map(int, match)
                visible_ingredients.append({
                    "name": f"Ingredient {i+1}",
                    "bounding_box": [x_min, y_min, x_max, y_max]
                })
    
    if not visible_ingredients:
        print("⚠ No bounding boxes found in analysis results")
        print("   The response may not contain bounding box coordinates")
        return image_path
    
    print(f"\n📦 Found {len(visible_ingredients)} items with bounding boxes")
    
    # Generate colors for each ingredient (use PIL's colormap)
    colors_list = list(ImageColor.colormap.keys())
    colors = [colors_list[i % len(colors_list)] for i in range(len(visible_ingredients))]
    
    # Draw bounding boxes
    for i, ingredient in enumerate(visible_ingredients):
        name = ingredient.get("name", f"Item {i+1}")
        bbox = ingredient.get("bounding_box", [])
        
        if len(bbox) != 4:
            print(f"⚠ Skipping {name}: invalid bounding box format {bbox}")
            continue
        
        x_min, y_min, x_max, y_max = bbox
        color = colors[i]
        
        # Validate coordinates
        img_width, img_height = image.size
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))
        
        # Draw rectangle (convert color name to RGB if needed)
        if isinstance(color, str):
            rgb_color = ImageColor.getrgb(color)
        else:
            rgb_color = color
        
        draw.rectangle([x_min, y_min, x_max, y_max], outline=rgb_color, width=4)
        
        # Draw label background
        label_text = name
        if font:
            bbox_text = draw.textbbox((0, 0), label_text, font=font_small or font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        else:
            text_width = len(label_text) * 6
            text_height = 12
        
        # Draw label background rectangle (use white/light background for black text)
        label_bg = [x_min, y_min - text_height - 4, x_min + text_width + 4, y_min]
        # Use white background with slight transparency effect, or light color
        draw.rectangle(label_bg, fill=(255, 255, 255))  # White background
        
        # Draw label text (black for better readability)
        draw.text((x_min + 8, y_min - text_height + 2), label_text, fill=(0, 0, 0), font=font_small or font)
        
        print(f"   [{i+1}] {name}: [{x_min}, {y_min}, {x_max}, {y_max}]")
    
    # Save annotated image
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
    
    image.save(output_path)
    print(f"\n💾 Annotated image saved to: {output_path}")
    
    return output_path


def test_single_image(image_path: Optional[Path] = None):
    """
    Test Gemini analysis on a single image.
    
    Args:
        image_path: Path to image. If None, picks first image from food101_images folder.
    """
    print("=" * 60)
    print("Gemini Food Image Analysis Test")
    print("=" * 60)
    
    # Find an image if not provided
    if not image_path:
        images_dir = Path(__file__).parent / "food101_images"
        
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                "Please run extract_images.py first to extract images."
            )
        
        # Find first image
        image_files = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.jpeg")) + list(images_dir.rglob("*.png"))
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {images_dir}")
        
        image_path = image_files[0]
        print(f"\n📁 Using first available image: {image_path}")
    
    # Try new API first (if available), fallback to old API
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
            except:
                pass
    
    if NEW_GENAI_AVAILABLE and api_key:
        print("\n🚀 Using new Google GenAI API with structured bounding box output...")
        try:
            result = analyze_with_structured_boxes(image_path, api_key)
        except Exception as e:
            print(f"⚠ New API failed: {e}")
            print("   Falling back to old API...")
            model = setup_gemini(api_key)
            result = analyze_food_image(image_path, model)
    else:
        if not NEW_GENAI_AVAILABLE:
            print("\n⚠ New Google GenAI client not available.")
            print("   Install with: pip install google-genai")
            print("   Using old API instead...")
        # Setup Gemini (old API)
        model = setup_gemini(api_key)
        # Analyze image
        result = analyze_food_image(image_path, model)
    
    # Display results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    if result.get("parsed"):
        print("\n📋 Structured Data:")
        print(json.dumps({k: v for k, v in result.items() if k not in ["raw_response", "image_path", "image_name", "parsed"]}, indent=2))
    else:
        print("\n📝 Raw Response:")
        print(result["raw_response"])
    
    # Save results
    output_file = Path(__file__).parent / f"gemini_analysis_{Path(image_path).stem}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot bounding boxes if available
    try:
        annotated_image_path = plot_bounding_boxes(image_path, result)
        print(f"📊 Bounding boxes visualized on: {annotated_image_path}")
    except Exception as e:
        print(f"⚠ Could not plot bounding boxes: {e}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gemini API on Food-101 images")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (default: first image in food101_images/)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available Gemini models and exit"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("=" * 60)
        print("Available Gemini Models")
        print("=" * 60)
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # Try to get from file
            test_file = Path(__file__).parent.parent / "FoodAI" / "nutrition-video-analysis" / "terraform" / "docker" / "TEST_OPTIMIZATIONS.md"
            if test_file.exists():
                try:
                    content = test_file.read_text()
                    for line in content.split('\n'):
                        if 'GEMINI_API_KEY=' in line and 'export' in line:
                            api_key = line.split('"')[1] if '"' in line else line.split("'")[1]
                            break
                except:
                    pass
        
        if api_key:
            models = list_available_models(api_key)
            if models:
                print("\nModels supporting generateContent:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("\n⚠ Could not retrieve model list")
        else:
            print("\n⚠ No API key found. Cannot list models.")
            print("   Provide --api-key or set GEMINI_API_KEY environment variable")
        sys.exit(0)
    
    # Set API key if provided
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key
    
    # Get image path
    image_path = Path(args.image) if args.image else None
    
    # Run test
    try:
        result = test_single_image(image_path)
        print("\n✓ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
