#!/usr/bin/env python3
"""
Unit test for timestamp watermark functionality.
"""

import os
import tempfile
from PIL import Image
from verl.utils.video_frame_cache import add_timestamp_watermark


def test_add_timestamp_watermark():
    """Test that watermark is added correctly to an image."""
    # Create a test image
    img = Image.new('RGB', (640, 480), color='blue')

    # Add watermark
    watermarked_img = add_timestamp_watermark(
        img,
        timestamp=12.0,
        position="top_left",
        font_size=24,
    )

    # Verify output
    assert watermarked_img is not None
    assert watermarked_img.size == img.size
    assert watermarked_img.mode == 'RGB'

    # Verify the watermarked image is different from original
    # (by comparing pixel values in the watermark region)
    original_pixel = img.getpixel((15, 15))  # Inside watermark area
    watermarked_pixel = watermarked_img.getpixel((15, 15))

    # The pixels should be different due to the watermark
    assert original_pixel != watermarked_pixel, "Watermark should modify pixels"

    print("test_add_timestamp_watermark PASSED")


def test_watermark_positions():
    """Test all watermark positions."""
    img = Image.new('RGB', (640, 480), color='green')

    positions = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for position in positions:
        watermarked_img = add_timestamp_watermark(
            img,
            timestamp=42.0,
            position=position,
            font_size=24,
        )
        assert watermarked_img is not None
        assert watermarked_img.size == img.size
        print(f"  Position {position}: OK")

    print("test_watermark_positions PASSED")


def test_watermark_formats():
    """Test watermark on different image formats."""
    # RGB image
    rgb_img = Image.new('RGB', (640, 480), color='red')
    watermarked_rgb = add_timestamp_watermark(rgb_img, timestamp=5.0)
    assert watermarked_rgb.mode == 'RGB'

    # RGBA image
    rgba_img = Image.new('RGBA', (640, 480), color=(255, 0, 0, 255))
    watermarked_rgba = add_timestamp_watermark(rgba_img, timestamp=5.0)
    # Note: output mode depends on input mode

    print("test_watermark_formats PASSED")


def test_save_watermarked_image():
    """Test saving watermarked image to file."""
    img = Image.new('RGB', (640, 480), color='yellow')

    watermarked_img = add_timestamp_watermark(
        img,
        timestamp=123.0,
        position="top_left",
        font_size=32,
    )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name

    try:
        watermarked_img.save(temp_path, "JPEG", quality=95)

        # Reload and verify
        reloaded = Image.open(temp_path)
        assert reloaded.size == img.size

        print(f"test_save_watermarked_image PASSED (saved to {temp_path})")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    test_add_timestamp_watermark()
    test_watermark_positions()
    test_watermark_formats()
    test_save_watermarked_image()
    print("\nAll tests PASSED!")
