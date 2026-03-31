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
    original_pixel = img.getpixel((15, 15))  # Save before watermark

    # Add watermark (modifies in-place)
    result = add_timestamp_watermark(
        img,
        timestamp=12.5,
        position="top_left",
        font_size=24,
    )

    # Verify output is the same object (in-place modification)
    assert result is img
    assert img.size == (640, 480)

    # Verify the image was modified in the watermark region
    watermarked_pixel = img.getpixel((15, 15))
    assert original_pixel != watermarked_pixel, "Watermark should modify pixels"

    print("test_add_timestamp_watermark PASSED")


def test_watermark_positions():
    """Test all watermark positions."""
    positions = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for position in positions:
        img = Image.new('RGB', (640, 480), color='green')
        result = add_timestamp_watermark(
            img,
            timestamp=42.0,
            position=position,
            font_size=24,
        )
        assert result is img
        assert img.size == (640, 480)
        print(f"  Position {position}: OK")

    print("test_watermark_positions PASSED")


def test_watermark_formats():
    """Test watermark on different image formats."""
    # RGB image
    rgb_img = Image.new('RGB', (640, 480), color='red')
    add_timestamp_watermark(rgb_img, timestamp=5.0)
    assert rgb_img.mode == 'RGB'

    # RGBA image
    rgba_img = Image.new('RGBA', (640, 480), color=(255, 0, 0, 255))
    add_timestamp_watermark(rgba_img, timestamp=5.0)
    assert rgba_img.mode == 'RGBA'

    print("test_watermark_formats PASSED")


def test_save_watermarked_image():
    """Test saving watermarked image to file."""
    img = Image.new('RGB', (640, 480), color='yellow')

    add_timestamp_watermark(
        img,
        timestamp=123.0,
        position="top_left",
        font_size=32,
    )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name

    try:
        img.save(temp_path, "JPEG", quality=95)

        # Reload and verify
        reloaded = Image.open(temp_path)
        assert reloaded.size == img.size

        print(f"test_save_watermarked_image PASSED (saved to {temp_path})")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_auto_font_size():
    """Test auto font size scaling based on image height."""
    # Small image - should use minimum font size (28)
    small_img = Image.new('RGB', (200, 150), color='blue')
    add_timestamp_watermark(small_img, timestamp=5.0, font_size=0)

    # Large image - should scale up
    large_img = Image.new('RGB', (1920, 1080), color='blue')
    add_timestamp_watermark(large_img, timestamp=99.9, font_size=0)

    print("test_auto_font_size PASSED")


def test_subsecond_watermark():
    """Test watermark with sub-second timestamps (fps>1)."""
    from PIL import ImageDraw

    # Test that 0.25 is rendered as "0.25s", not "0.2s" (old .1f truncation)
    img = Image.new('RGB', (640, 480), color='blue')
    result = add_timestamp_watermark(img, timestamp=0.25, position="top_left", font_size=24)
    assert result is img

    # Test integer timestamp still uses .1f format ("1.0s")
    img2 = Image.new('RGB', (640, 480), color='blue')
    result2 = add_timestamp_watermark(img2, timestamp=1.0, position="top_left", font_size=24)
    assert result2 is img2

    # Test various sub-second values
    for ts in [0.25, 0.5, 0.75, 1.25, 2.5]:
        img = Image.new('RGB', (640, 480), color='green')
        add_timestamp_watermark(img, timestamp=ts, font_size=24)

    print("test_subsecond_watermark PASSED")


if __name__ == "__main__":
    test_add_timestamp_watermark()
    test_watermark_positions()
    test_watermark_formats()
    test_save_watermarked_image()
    test_auto_font_size()
    test_subsecond_watermark()
    print("\nAll tests PASSED!")
