"""
GPU Canvas for drawing operations using PyTorch.
"""

import torch
from PIL import Image


class GPUCanvas:
    """
    GPU canvas for drawing operations.
    Canvas starts white and supports clear and add_rectangle operations.
    """

    def __init__(self, width, height, device=None):
        """
        Initialize a GPU canvas.

        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
        """
        self.width = int(width)
        self.height = int(height)

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize canvas as white (RGB, values 0-255)
        # Shape: (height, width, 3) for RGB
        self.canvas = (
            torch.ones(
                (self.height, self.width, 3), dtype=torch.uint8, device=self.device
            )
            * 255
        )

    @classmethod
    def from_array(cls, img_array, device=None):
        """
        Create a GPUCanvas from a numpy array.

        Args:
            img_array: numpy array of shape (height, width, 3) or (height, width, 4) RGB/RGBA
                      Values should be 0-255
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)

        Returns:
            GPUCanvas instance
        """
        if len(img_array.shape) != 3:
            raise ValueError(
                f"Expected 3D array (H, W, C), got shape {img_array.shape}"
            )

        height, width = img_array.shape[0], img_array.shape[1]

        # Auto-detect device if not specified
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Create canvas instance
        canvas = cls.__new__(cls)
        canvas.width = int(width)
        canvas.height = int(height)
        canvas.device = device

        # Convert array to tensor
        if img_array.shape[2] == 4:
            # RGBA - take only RGB channels
            img_array = img_array[:, :, :3]
        elif img_array.shape[2] != 3:
            raise ValueError(
                f"Expected 3 or 4 channels (RGB/RGBA), got {img_array.shape[2]} channels"
            )

        # Convert to tensor and move to device
        canvas.canvas = torch.from_numpy(img_array).to(device).to(torch.uint8)

        return canvas

    def clear(self):
        """Clear the canvas (reset to white)."""
        self.canvas.fill_(255)

    def add_rectangle(self, x1, y1, x2, y2, color=(0, 0, 0)):
        """
        Add a rectangle to the canvas.

        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            color: RGB color tuple (0-255) for the rectangle, default is black (0, 0, 0)
        """
        # Convert to integers and clamp to canvas bounds
        x1 = max(0, min(int(x1), self.width))
        y1 = max(0, min(int(y1), self.height))
        x2 = max(0, min(int(x2), self.width))
        y2 = max(0, min(int(y2), self.height))

        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Convert color to tensor
        color_tensor = torch.tensor(color, dtype=torch.uint8, device=self.device)

        # Draw rectangle (fill the region)
        self.canvas[y1:y2, x1:x2, :] = color_tensor

    def to_numpy(self):
        """
        Convert canvas to numpy array (CPU).

        Returns:
            numpy array of shape (height, width, 3) with RGB values 0-255
        """
        return self.canvas.cpu().numpy()

    def to_pil(self):
        """
        Convert canvas to PIL Image.

        Returns:
            PIL Image object
        """
        return Image.fromarray(self.to_numpy(), "RGB")

    def save(self, filename):
        """
        Save canvas to file.

        Args:
            filename: Output filename (PNG, JPEG, etc.)
        """
        self.to_pil().save(filename)

    def compute_loss(self, other):
        """
        Compute MSE loss between this canvas and another canvas.

        Args:
            other: Another GPUCanvas instance to compare against

        Returns:
            MSE loss value as a tensor (on the same device as canvas)

        Raises:
            ValueError: If canvases have different dimensions
        """
        if self.width != other.width or self.height != other.height:
            raise ValueError(
                f"Canvas dimensions must match: ({self.width}, {self.height}) vs ({other.width}, {other.height})"
            )

        # Ensure both canvases are on the same device
        other_canvas = other.canvas.to(self.device)

        # Convert to float for loss computation
        self_float = self.canvas.float()
        other_float = other_canvas.float()

        # Mean Squared Error
        loss = torch.mean((self_float - other_float) ** 2)

        return loss
