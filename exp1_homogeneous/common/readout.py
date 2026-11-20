"""Physical-coordinate bin averaging and predeclared profile diagnostics."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

READOUT_VERSION = "pixelwise_N0_then_y_mean_then_x_bin/v1"


def load_run(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def bin_average(x, values, edges):
    """Integrate piecewise-constant voxel samples over physical bins."""
    x = np.asarray(x, dtype=float)
    dx = float(np.median(np.diff(x)))
    if not np.allclose(np.diff(x), dx):
        raise ValueError("Uniform voxel centers required")
    source_edges = np.r_[x - dx / 2, x[-1] + dx / 2]
    if edges[0] < source_edges[0] - 1e-8 or edges[-1] > source_edges[-1] + 1e-8:
        raise ValueError("Requested profile lies outside MC domain")
    integral = np.r_[0, np.cumsum(values, dtype=float) * dx]
    at_edges = np.interp(edges, source_edges, integral)
    return np.diff(at_edges) / np.diff(edges)


def native_profile(channel, background):
    cm = json.loads(str(channel["metadata_json"]))
    bm = json.loads(str(background["metadata_json"]))
    for field in ("geometry", "optics", "mc"):
        if cm["config"][field] != bm["config"][field]:
            raise ValueError(f"Channel/N0 {field} mismatch")
    if not np.array_equal(channel["x_mm"], background["x_mm"]):
        raise ValueError("Channel and N0 coordinates differ")
    denominator = background["R_xy"].astype(float)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Invalid N0 denominator")
    if channel["R_xy"].shape != background["R_xy"].shape:
        raise ValueError("Channel and N0 surface dimensions differ")
    ratio_xy = channel["R_xy"].astype(float) / denominator
    return ratio_xy.mean(axis=1)


def profile(channel, background, width=32.0, points=128):
    ratio = native_profile(channel, background)
    edges = np.linspace(-width / 2, width / 2, points + 1)
    x128 = (edges[:-1] + edges[1:]) / 2
    y128 = bin_average(channel["x_mm"], ratio, edges)
    if not np.isfinite(y128).all():
        raise ValueError("Non-finite profile")
    return x128, y128.astype(np.float32)


def diagnostics(x, ratio, sigma_mm=0.25):
    # Symmetry and smoothing are diagnostic only, never a silent training transform.
    contrast = 1 - np.asarray(ratio, dtype=float)
    contrast = (contrast + contrast[::-1]) / 2
    dx = float(np.median(np.diff(x)))
    smoothed = gaussian_filter1d(contrast, sigma_mm / dx)
    central = np.abs(x) <= 0.5
    peak = float(np.max(smoothed[central]))
    fwhm = None
    if peak > 0:
        pos = np.flatnonzero(x >= 0)
        hit = np.flatnonzero(smoothed[pos] <= peak / 2)
        if len(hit) and hit[0] > 0:
            i = pos[hit[0]]
            a, b = smoothed[i - 1], smoothed[i]
            crossing = x[i - 1] + (a - peak / 2) / (a - b) * (x[i] - x[i - 1])
            fwhm = float(2 * crossing)
    return {"central_contrast": peak, "fwhm_mm": fwhm,
            "edge_abs_contrast": float(np.mean(np.abs(smoothed[np.abs(x) >= 14])))}


def compare(reference, candidate, rtol=0.1, absolute_tolerance=0.005):
    peak_error = abs(candidate["central_contrast"] - reference["central_contrast"])
    relative = peak_error / max(abs(reference["central_contrast"]), 1e-12)
    width_error = None
    if reference["fwhm_mm"] is not None and candidate["fwhm_mm"] is not None:
        width_error = abs(candidate["fwhm_mm"] - reference["fwhm_mm"]) / reference["fwhm_mm"]
    weak = reference["central_contrast"] < absolute_tolerance / rtol
    peak_pass = peak_error <= absolute_tolerance if weak else relative <= rtol
    width_pass = width_error is not None and width_error <= rtol
    return {"contrast_absolute_difference": peak_error, "contrast_relative_difference": relative,
            "fwhm_relative_difference": width_error, "weak_signal": weak,
            "pass": bool(peak_pass and width_pass),
            "rule": "absolute contrast tolerance for weak signals; relative contrast otherwise; relative FWHM for both"}
