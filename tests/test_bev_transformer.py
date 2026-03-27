"""Regression tests for BEV polygon normalization and point ordering."""

import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane_mapping.bird_eye_view import BirdEyeViewTransformer


def test_order_points_stable_for_trapezoid():
    """Shuffled trapezoid points should be ordered TL, TR, BR, BL."""
    shuffled = np.array(
        [
            [900, 500],
            [200, 100],
            [100, 500],
            [800, 100],
        ],
        dtype=np.float32,
    )

    transformer = BirdEyeViewTransformer(source_polygon=shuffled, bev_width=400, bev_height=600)

    expected = np.array(
        [
            [200, 100],
            [800, 100],
            [900, 500],
            [100, 500],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(transformer.source_points, expected)


def test_source_polygon_normalization_deduplicates_and_rounds():
    """Input polygon should be rounded and duplicate closing point removed."""
    raw = np.array(
        [
            [100.4, 200.6],
            [300.2, 210.8],
            [500.5, 450.1],
            [120.7, 440.9],
            [100.4, 200.6],  # closing duplicate
        ],
        dtype=np.float32,
    )

    transformer = BirdEyeViewTransformer(source_polygon=raw, bev_width=400, bev_height=600)

    expected_polygon = np.array(
        [
            [100, 201],
            [300, 211],
            [500, 450],
            [121, 441],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(transformer.source_polygon, expected_polygon)


def test_homography_matrix_is_finite_after_normalization():
    """Homography should remain valid for a typical road trapezoid."""
    polygon = np.array(
        [
            [420.0, 180.0],
            [860.0, 180.0],
            [1040.0, 640.0],
            [260.0, 640.0],
        ],
        dtype=np.float32,
    )

    transformer = BirdEyeViewTransformer(source_polygon=polygon, bev_width=500, bev_height=700)

    assert np.isfinite(transformer.transform_matrix).all()
    assert np.isfinite(transformer.inverse_matrix).all()


