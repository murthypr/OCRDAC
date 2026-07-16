"""Automatic preprocessing detection for OCRDAC.

Analyzes page images to determine whether median filtering is needed
before OCR, based on brightness, contrast, variance, stripe artifacts,
and edge strength.
"""

import math


def _computeStripeScore(gray_pixels, width, height):
    """Compute a horizontal stripe score using row-wise analysis.

    Scanner stripes are rows where ALL pixels are shifted darker/lighter,
    making them internally uniform (low within-row variance). Text rows
    have a mix of dark text and white background (high within-row variance).

    We find rows that deviate significantly from the overall mean, then
    check if those deviant rows are internally uniform. A high fraction
    of uniform deviant rows indicates stripe artifacts.
    """
    if height < 3:
        return 0.0

    row_means = []
    row_variances = []
    for y in range(height):
        row_start = y * width
        row_end = row_start + width
        row = gray_pixels[row_start:row_end]
        row_mean = sum(row) / width
        row_means.append(row_mean)
        row_var = sum((p - row_mean) ** 2 for p in row) / width
        row_variances.append(row_var)

    overall_mean = sum(row_means) / len(row_means)
    if overall_mean == 0:
        return 0.0

    overall_row_var = sum(row_variances) / len(row_variances)

    # Find rows that deviate significantly from the overall mean
    deviation_threshold = 20
    deviant_rows = 0
    uniform_deviant_rows = 0

    for i in range(height):
        if abs(row_means[i] - overall_mean) > deviation_threshold:
            deviant_rows += 1
            # Stripe rows are uniform within the row (low variance)
            # compared to the average row variance. A row with text has
            # high within-row variance because it mixes dark and light.
            if overall_row_var > 0:
                if row_variances[i] < overall_row_var * 0.3:
                    uniform_deviant_rows += 1
            else:
                # All rows are perfectly uniform (e.g., solid-color stripes).
                # If a row has zero within-row variance, it's uniform.
                if row_variances[i] == 0:
                    uniform_deviant_rows += 1

    if deviant_rows == 0:
        return 0.0

    # Score: what fraction of deviant rows are uniform (stripe-like)?
    # High ratio = stripes; low ratio = text/content
    uniformity_ratio = uniform_deviant_rows / deviant_rows

    # Also weight by the fraction of total rows that are deviant
    deviant_fraction = deviant_rows / height

    # Stripe score: high if many deviant rows are uniform (not text)
    # Scale by deviant fraction so it's not triggered by a single outlier
    stripe_score = uniformity_ratio * min(deviant_fraction * 4, 1.0)

    return stripe_score


def _computeEdgeStrength(gray_pixels, width, height):
    """Compute edge strength using a simplified Sobel-like operator.

    Returns the mean edge magnitude across the image. Strong edges indicate
    clean text; weak/absent edges suggest poor scan quality.
    """
    if width < 3 or height < 3:
        return 0.0

    total_magnitude = 0.0
    count = 0

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            idx = y * width + x

            # Simplified Sobel: horizontal and vertical gradients
            gx = (-gray_pixels[(y-1)*width + (x-1)]
                   + gray_pixels[(y-1)*width + (x+1)]
                   - 2 * gray_pixels[y*width + (x-1)]
                   + 2 * gray_pixels[y*width + (x+1)]
                   - gray_pixels[(y+1)*width + (x-1)]
                   + gray_pixels[(y+1)*width + (x+1)])

            gy = (-gray_pixels[(y-1)*width + (x-1)]
                   - 2 * gray_pixels[(y-1)*width + x]
                   - gray_pixels[(y-1)*width + (x+1)]
                   + gray_pixels[(y+1)*width + (x-1)]
                   + 2 * gray_pixels[(y+1)*width + x]
                   + gray_pixels[(y+1)*width + (x+1)])

            magnitude = math.sqrt(gx * gx + gy * gy)
            total_magnitude += magnitude
            count += 1

    return total_magnitude / max(count, 1)


def detect_preprocessing_needed(image):
    """Analyze a page image and decide if median preprocessing is needed.

    Args:
        image: A PIL.Image object (any mode; will be converted to grayscale).

    Returns:
        tuple: (needs_preprocessing: bool, reason: str)
            reason is one of: "low_contrast", "stripe_artifacts",
            "uneven_background", "high_noise", or "none".
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = list(gray.getdata())
    total_pixels = len(pixels)

    if total_pixels == 0:
        return False, "none"

    # Mean brightness
    mean_brightness = sum(pixels) / total_pixels

    # Contrast (max - min)
    min_val = min(pixels)
    max_val = max(pixels)
    contrast = max_val - min_val

    # Pixel variance
    variance = sum((p - mean_brightness) ** 2 for p in pixels) / total_pixels

    # Stripe score
    stripe_score = _computeStripeScore(pixels, width, height)

    # Edge strength (sample subset for performance on large images)
    if total_pixels > 500000:
        # Downsample for speed
        step = max(1, int(math.sqrt(total_pixels // 500000)))
        sampled = pixels[::step]
        sw = width // step
        sh = height // step
        edge_strength = _computeEdgeStrength(sampled, sw, sh)
    else:
        edge_strength = _computeEdgeStrength(pixels, width, height)

    # Decision logic based on thresholds (from spec):
    # low_contrast, stripes_present, uneven_background trigger preprocessing.
    # high_noise (variance > 500) is computed but not used in the decision
    # because clean scans with text naturally have high variance from the
    # dark-on-white bimodal pixel distribution.
    low_contrast = contrast < 40
    stripes_present = stripe_score > 0.15
    uneven_background = 120 < mean_brightness < 200

    if low_contrast:
        return True, "low_contrast"
    if stripes_present:
        return True, "stripe_artifacts"
    if uneven_background:
        return True, "uneven_background"

    return False, "none"
