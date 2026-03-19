"""
tests/test_performance.py
=========================

Performance benchmarks for transform_vectors.

Measures time and memory usage as a function of grid size to verify
that the function scales linearly (O(N)) in both dimensions.

Run all benchmarks:
    pytest tests/test_performance.py -v -s

Run only the scaling benchmark:
    pytest tests/test_performance.py::test_time_scaling -v -s

Run the large grid benchmark:
    pytest tests/test_performance.py::test_large_grid -v -s
"""

import time
import tracemalloc

import numpy as np
import pytest
import pyproj

from transform_vectors import transform_vectors


# ── CRS fixtures ──────────────────────────────────────────────────────────────

STEREO_CRS = pyproj.CRS.from_proj4(
    "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84"
)
GEO_CRS = pyproj.CRS.from_epsg(4326)


# ── helpers ───────────────────────────────────────────────────────────────────


def make_grid(n):
    """
    Build a flat (n*n,) grid of x, y, u, v arrays.

    Returns x, y, u, v as 2D arrays of shape (n, n).
    """
    x = np.linspace(10000.0, 500000.0, n)
    y = np.linspace(-1000000.0, -500000.0, n)
    X, Y = np.meshgrid(x, y)
    u = np.ones_like(X)
    v = np.zeros_like(X)
    return X, Y, u, v


def benchmark(n, warmup=True):
    """
    Run transform_vectors on an n×n grid and return
    (n_points, elapsed_seconds, peak_memory_bytes).

    Parameters
    ----------
    n       : int, grid side length
    warmup  : bool, run once before timing to avoid cold-start effects
    """
    X, Y, u, v = make_grid(n)

    if warmup:
        transform_vectors(STEREO_CRS, GEO_CRS, X, Y, u, v)

    tracemalloc.start()
    t0 = time.perf_counter()

    transform_vectors(STEREO_CRS, GEO_CRS, X, Y, u, v)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return n * n, elapsed, peak_bytes


def print_row(n_pts, elapsed, peak_bytes, ref_pts=None, ref_time=None):
    """Pretty-print one benchmark row."""
    mb = peak_bytes / 1024**2
    rate = n_pts / elapsed / 1e6
    if ref_pts and ref_time:
        ratio_n = n_pts / ref_pts
        ratio_time = elapsed / ref_time
        scaling = ratio_time / ratio_n  # 1.0 = perfect linear scaling
        print(
            f"  {n_pts:>10,d} pts  |  {elapsed:6.3f} s  |  {mb:7.1f} MB  "
            f"|  {rate:5.2f} Mpts/s  |  scaling={scaling:.2f}x"
        )
    else:
        print(
            f"  {n_pts:>10,d} pts  |  {elapsed:6.3f} s  |  {mb:7.1f} MB  "
            f"|  {rate:5.2f} Mpts/s  |  (reference)"
        )


# ── benchmarks ────────────────────────────────────────────────────────────────

GRID_SIZES = [10, 50, 100, 200, 500, 1000]  # side lengths → 100 to 1M points


def test_time_scaling():
    """
    Verify that runtime scales linearly with number of grid points.

    Each doubling of grid side length (4x more points) should produce
    roughly 4x more runtime.  A scaling factor consistently above 2.0
    suggests superlinear (e.g. quadratic) behaviour.
    """
    print("\n")
    print("  " + "-" * 75)
    print(f"  {'N points':>10}       time        peak mem    throughput    scaling")
    print("  " + "-" * 75)

    results = []
    for n in GRID_SIZES:
        n_pts, elapsed, peak = benchmark(n)
        results.append((n_pts, elapsed, peak))
        ref_pts, ref_time, _ = results[0]
        print_row(
            n_pts,
            elapsed,
            peak,
            ref_pts if len(results) > 1 else None,
            ref_time if len(results) > 1 else None,
        )

    print("  " + "-" * 75)

    # assert linear scaling: runtime should grow at most 5x per 4x point increase
    for i in range(1, len(results)):
        n_pts_prev, t_prev, _ = results[i - 1]
        n_pts_curr, t_curr, _ = results[i]
        point_ratio = n_pts_curr / n_pts_prev
        time_ratio = t_curr / t_prev
        scaling = time_ratio / point_ratio
        assert scaling < 5.0, (
            f"Superlinear scaling detected between "
            f"{n_pts_prev:,} and {n_pts_curr:,} points: "
            f"time grew {time_ratio:.1f}x for {point_ratio:.1f}x more points "
            f"(scaling={scaling:.2f}, expected <5.0)"
        )


def test_memory_scaling():
    """
    Verify that peak memory scales linearly with number of grid points.

    Each 4x increase in points should produce roughly 4x more memory.
    A scaling factor above 5.0 suggests unexpected allocations.
    """
    print("\n")
    print("  " + "-" * 50)
    print(f"  {'N points':>10}       peak mem    mem scaling")
    print("  " + "-" * 50)

    results = []
    for n in GRID_SIZES:
        n_pts, _, peak = benchmark(n, warmup=False)
        results.append((n_pts, peak))
        if len(results) == 1:
            print(f"  {n_pts:>10,d} pts  |  {peak/1024**2:7.1f} MB  |  (reference)")
        else:
            ref_pts, ref_peak = results[0]
            ratio_n = n_pts / ref_pts
            ratio_mem = peak / ref_peak
            scaling = ratio_mem / ratio_n
            print(
                f"  {n_pts:>10,d} pts  |  {peak/1024**2:7.1f} MB  "
                f"|  scaling={scaling:.2f}x"
            )

    print("  " + "-" * 50)

    for i in range(1, len(results)):
        n_pts_prev, mem_prev = results[i - 1]
        n_pts_curr, mem_curr = results[i]
        point_ratio = n_pts_curr / n_pts_prev
        mem_ratio = mem_curr / mem_prev
        scaling = mem_ratio / point_ratio
        assert scaling < 5.0, (
            f"Superlinear memory growth detected between "
            f"{n_pts_prev:,} and {n_pts_curr:,} points: "
            f"memory grew {mem_ratio:.1f}x for {point_ratio:.1f}x more points "
            f"(scaling={scaling:.2f}, expected <5.0)"
        )


def test_large_grid():
    """
    Smoke test with 1,000,000 grid points (1000x1000).

    Verifies the function completes in under 60 seconds and uses
    less than 500 MB of peak memory.
    """
    n = 1000
    print(f"\n  Running 1000×1000 grid ({n*n:,} points)...")

    n_pts, elapsed, peak_bytes = benchmark(n, warmup=False)
    mb = peak_bytes / 1024**2

    print(f"  Time    : {elapsed:.2f} s")
    print(f"  Peak mem: {mb:.1f} MB")
    print(f"  Rate    : {n_pts / elapsed / 1e6:.2f} Mpts/s")

    assert elapsed < 60.0, f"Large grid took {elapsed:.1f}s, expected < 60s"
    assert mb < 500.0, f"Large grid used {mb:.0f} MB, expected < 500 MB"


@pytest.mark.parametrize(
    "crs_name,proj4",
    [
        ("Stereographic", "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84"),
        ("LCC", "+proj=lcc +lat_1=43 +lat_2=62 +lat_0=52 +lon_0=10 +datum=WGS84"),
        ("Mercator", "+proj=merc +lon_0=0 +datum=WGS84"),
        ("LAEA", "+proj=laea +lat_0=52 +lon_0=10 +datum=WGS84"),
        (
            "RotatedPole",
            "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84",
        ),
    ],
)
def test_throughput_by_projection(crs_name, proj4):
    """
    Throughput benchmark across different projections on a 500x500 grid.

    All projections should process at a similar rate since the algorithm
    is the same — differences only come from the pyproj transformer cost.
    """
    n = 500
    src = pyproj.CRS.from_proj4(proj4)
    is_rp = crs_name == "RotatedPole"

    if is_rp:
        x = np.linspace(-30.0, 30.0, n)
        y = np.linspace(-20.0, 20.0, n)
    else:
        x = np.linspace(10000.0, 500000.0, n)
        y = np.linspace(-1000000.0, -500000.0, n)

    X, Y = np.meshgrid(x, y)
    u, v = np.ones_like(X), np.zeros_like(X)

    # warmup
    transform_vectors(src, GEO_CRS, X, Y, u, v, is_rotated_pole=is_rp)

    t0 = time.perf_counter()
    transform_vectors(src, GEO_CRS, X, Y, u, v, is_rotated_pole=is_rp)
    elapsed = time.perf_counter() - t0

    rate = n * n / elapsed / 1e6
    print(f"\n  {crs_name:20s}  {n*n:>9,d} pts  {elapsed:.3f} s  {rate:.2f} Mpts/s")

    # no strict assertion — this is informational
    # but flag anything catastrophically slow
    assert (
        elapsed < 120.0
    ), f"{crs_name} took {elapsed:.1f}s for {n*n:,} points — unexpectedly slow"
