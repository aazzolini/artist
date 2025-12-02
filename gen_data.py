import svgwrite
import cairosvg
import cairosvg.parser
import cairosvg.surface
import numpy as np
import cairo
from PIL import Image
import random
import string
from gpu_canvas import GPUCanvas

CANVAS_SIZE = 256


def generate_letter_svg(letter, x, y, font_size=64, font_family="Arial", rotation=0, filename=None, png_filename=None):
    """
    Construct an SVG data structure in memory rendering a single letter
    at a given position, size, orientation, and font.
    Uses svgwrite for SVG construction.

    Args:
        letter (str): Single character to render.
        x (float): x-coordinate of the letter position.
        y (float): y-coordinate of the letter position.
        font_size (float): Size of the letter.
        font_family (str): Font family to use.
        rotation (float): Rotation angle in degrees (counterclockwise).
        filename (str, optional): If provided, saves the SVG to this file.
        png_filename (str, optional): If provided, saves the PNG to this file.

    Returns:
        tuple: (svgwrite.Drawing, np.array) - SVG drawing object and numpy array
    """

    # Fixed canvas size so downstream canvases stay aligned across references
    width = height = CANVAS_SIZE
    dwg = svgwrite.Drawing(size=(width, height))

    # Add white background rectangle
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    # Center the text on the canvas (or use x, y if provided and different from 0,0)
    if x == 0 and y == 0:
        center_x = width / 2
        center_y = height / 2
    else:
        center_x = x
        center_y = y
    
    # Create text element - use explicit y position instead of dominant-baseline
    # Adjust y to account for font baseline (typically ~0.75 of font_size from top)
    text_y = center_y + font_size * 0.35  # Approximate centering for most fonts
    
    # Use sans-serif as primary font to avoid fallback rendering issues
    # If specific font not available, use generic sans-serif
    safe_font = font_family if font_family in ["Arial", "Helvetica", "sans-serif"] else "sans-serif"
    text_style = f"font-family: {safe_font}; font-size: {font_size}px; fill: black; stroke: none;"
    
    if rotation == 0:
        # Simple case: no rotation
        text = dwg.text(
            letter,
            insert=(center_x, text_y),
            font_size=font_size,
            font_family=safe_font,
            text_anchor="middle",
            fill="black",
            stroke="none",
            style=text_style
        )
        dwg.add(text)
    else:
        # Use group for rotation
        g = dwg.g(transform="translate({},{}) rotate({})".format(center_x, text_y, rotation))
        text = dwg.text(
            letter,
            insert=(0, 0),
            font_size=font_size,
            font_family=safe_font,
            text_anchor="middle",
            fill="black",
            stroke="none",
            style=text_style
        )
        g.add(text)
        dwg.add(g)
    
    # Save the SVG file if filename is provided
    if filename:
        dwg.saveas(filename)
    
    # Convert SVG string to PNG (in memory or to file)
    svg_string = dwg.tostring()
    
    # Use svg2png directly to convert to PNG bytes, then to numpy array
    # This avoids potential issues with PNGSurface
    from io import BytesIO
    
    # Convert to PNG with explicit background
    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), 
                                  output_width=width, 
                                  output_height=height,
                                  background_color="white")
    
    # Convert PNG bytes to PIL Image, then to numpy array
    img = Image.open(BytesIO(png_bytes))
    img_array = np.array(img)
    
    # Ensure RGBA format (svg2png might return RGB)
    if img_array.shape[2] == 3:
        # Add alpha channel (fully opaque)
        alpha = np.ones((height, width, 1), dtype=img_array.dtype) * 255
        img_array = np.concatenate([img_array, alpha], axis=2)
    
    # Convert to RGB for saving (some viewers have issues with RGBA)
    img_rgb = img_array[:, :, :3].copy()
    
    # Save PNG file if png_filename is provided
    if png_filename:
        # Save as RGB to avoid any viewer issues with RGBA
        img = Image.fromarray(img_rgb, 'RGB')
        img.save(png_filename)
    
    return dwg, img_array


def array_to_ascii(img_array, chars=" .:-=+*#%@", invert=False, scale_width=1.0):
    """
    Convert a numpy image array to ASCII art.
    
    Args:
        img_array: numpy array with shape (height, width, 4) RGBA or (height, width, 3) RGB
        chars: string of characters from lightest to darkest (or vice versa)
        invert: if True, invert the brightness mapping
        scale_width: scale factor for width (to account for character aspect ratio)
    
    Returns:
        str: ASCII art representation
    """
    # Extract alpha channel if present, otherwise use RGB
    if img_array.shape[2] == 4:
        # Use alpha channel to determine visibility, convert RGB to grayscale
        alpha = img_array[:, :, 3].astype(np.float32) / 255.0
        rgb = img_array[:, :, :3].astype(np.float32)
        # Convert RGB to grayscale using standard weights
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
        # Combine with alpha
        intensity = gray * alpha
    else:
        # Just RGB, convert to grayscale
        rgb = img_array[:, :, :3].astype(np.float32)
        intensity = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    
    if invert:
        intensity = 1.0 - intensity
    
    # Scale width to account for character aspect ratio (typically ~2:1)
    if scale_width != 1.0:
        new_width = int(intensity.shape[1] * scale_width)
        # Simple downsampling by taking every Nth pixel
        if scale_width < 1.0:
            step = int(1.0 / scale_width)
            intensity = intensity[:, ::step]
        # Simple upsampling by repeating pixels
        elif scale_width > 1.0:
            intensity = np.repeat(intensity, int(scale_width), axis=1)
    
    # Map intensity to character indices
    char_indices = (intensity * (len(chars) - 1)).astype(np.int32)
    char_indices = np.clip(char_indices, 0, len(chars) - 1)
    
    # Create ASCII art string
    ascii_lines = []
    for row in char_indices:
        ascii_lines.append(''.join(chars[i] for i in row))
    
    return '\n'.join(ascii_lines)


def generate_random_letters(
    num_letters=None,
    available_fonts=None,
    letters_to_use=None,
    font_size_range=(32, 96),
    rotation_range=(0, 360)
):
    """
    Generator that yields random letter pairs (svg, img_array) with metadata.
    Generates indefinitely (or up to num_letters if specified). Never saves files.
    
    Args:
        num_letters: Number of letters to generate (None for infinite)
        available_fonts: List of font names to choose from
        letters_to_use: String of letters to choose from
        font_size_range: Tuple of (min, max) font sizes
        rotation_range: Tuple of (min, max) rotation angles in degrees
    
    Yields:
        tuple: (svg, img_array, metadata_dict) where metadata contains:
            - letter: the letter character
            - font_family: font used
            - font_size: size used
            - rotation: rotation angle
            - position: (x, y) coordinates
            - index: generation index
    """
    if available_fonts is None:
        available_fonts = ["Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana", "Georgia", "sans-serif"]
    if letters_to_use is None:
        letters_to_use = string.ascii_uppercase
    
    i = 0
    while num_letters is None or i < num_letters:
        # Random parameters
        letter = random.choice(letters_to_use)
        font_size = random.randint(font_size_range[0], font_size_range[1])
        rotation = random.uniform(rotation_range[0], rotation_range[1])
        font_family = random.choice(available_fonts)
        
        # Random position with a margin so most of the glyph stays inside the canvas
        canvas_size = CANVAS_SIZE
        # Estimate a half-diagonal generous enough to cover rotation and font metrics
        est_half_diag = font_size * 0.9
        # Keep margin large enough for coverage but not larger than half the canvas
        max_margin = canvas_size / 2 - 1
        margin = min(est_half_diag, max_margin)
        x = random.uniform(margin, canvas_size - margin)
        y = random.uniform(margin, canvas_size - margin)
        
        # Generate the letter (never saves files)
        svg, img_array = generate_letter_svg(
            letter, x, y,
            font_size=font_size,
            font_family=font_family,
            rotation=rotation,
            filename=None,
            png_filename=None
        )
        
        # Create metadata
        metadata = {
            'letter': letter,
            'font_family': font_family,
            'font_size': font_size,
            'rotation': rotation,
            'position': (x, y),
            'index': i
        }
        
        yield svg, img_array, metadata
        i += 1


if __name__ == "__main__":
    # Generate letters indefinitely, optionally save in outer loop
    print("Generating letters indefinitely (memory only)...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Use the generator - runs indefinitely, never saves
    save_interval = None #  100  # Save every 100th letter (optional, set to None to never save)
    filename_prefix = "letter"
    
    for svg, img_array, metadata in generate_random_letters():
        i = metadata['index']
        letter = metadata['letter']
        
        # Print every generated letter
        print(f"[{i}] Generated '{letter}' - Font: {metadata['font_family']}, "
              f"Size: {metadata['font_size']}, Rotation: {metadata['rotation']:.1f}°, "
              f"Position: ({metadata['position'][0]:.1f}, {metadata['position'][1]:.1f})")
        
        # Optionally save files in the outer loop
        if save_interval is not None and i % save_interval == 0:
            filename = f"{filename_prefix}_{i:06d}_{letter}.svg"
            png_filename = f"{filename_prefix}_{i:06d}_{letter}.png"
            # Save by calling generate_letter_svg again with filenames, or save directly
            svg.saveas(filename)
            img = Image.fromarray(img_array[:, :, :3], 'RGB')
            img.save(png_filename)
            print(f"  [SAVED] {filename}, {png_filename}")
        
        # Show progress every 1000 letters
        if i > 0 and i % 1000 == 0:
            saved_count = (i // save_interval) if save_interval else 0
            print(f"\n--- Generated {i} letters so far (saved {saved_count} files) ---\n")
        
        # Show ASCII art for first letter only
        if i == 0:
            print(f"\n  ASCII Art (first letter):")
            ascii_art = array_to_ascii(img_array, invert=True, scale_width=0.5)
            ascii_lines = ascii_art.split('\n')
            for line in ascii_lines[:15]:
                if line.strip():
                    print(f"  {line}")
            print()
        
        # Convert letter to GPU canvas and compare with a blank canvas
        if i < 5:  # Do this for first 5 letters as example
            # Convert letter numpy array to GPU canvas
            letter_canvas = GPUCanvas.from_array(img_array[:, :, :3])  # Take RGB channels
            
            # Create a blank white canvas for comparison
            blank_canvas = GPUCanvas(letter_canvas.width, letter_canvas.height)
            
            # Compute loss
            loss = letter_canvas.compute_loss(blank_canvas)
            print(f"  Loss vs blank canvas: {loss.item():.2f}")
