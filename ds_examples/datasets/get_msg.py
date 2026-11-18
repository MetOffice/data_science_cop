#!/usr/bin/env python
"""
Retrieve SEVIRI all-sky data from MetDB (MSGASR) and produce full-disc imagery.

Three output modes are available:
  rgb          Dust RGB composite from IR channels.
  single       Greyscale full-disc for one numbered channel.
  all-channels Greyscale full-disc for every channel in a single figure.

Dust RGB recipe (EUMETSAT):
  Red:   BT(12.0µm) - BT(10.8µm),  range -4 to 2 K
  Green: BT(10.8µm) - BT(8.7µm),   range 0 to 15 K
  Blue:  BT(10.8µm),               range 261 to 289 K

MSGASR data is super-obbed (fewer obs, larger footprint ~48km segments).
The replicated group CSR_REPS contains (CSR_CNTR_FRQY, BRGTS_TMPR) × 10 channels.
First 3 channels (visible/NIR) are always missing; IR channels 4-10 have valid BT.

Usage:
    python get_msg.py                               # RGB, latest available
    python get_msg.py --time 202606091200           # RGB, specific time
    python get_msg.py --mode single --channel 5    # channel 5 greyscale
    python get_msg.py --mode all-channels          # all channels overview
"""

import argparse
import datetime
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import netCDF4
from scipy.interpolate import griddata

import metdb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTACT = "david.rundle@metoffice.gov.uk"
SUBTYPE = "MSGASR"  # MSGASR = all-sky radiance (super-obbed, replicated channels)

# Dust RGB uses three IR channels identified by central frequency.
CHANNELS = {
    "8.7": {"wavelength_um": 8.70, "freq_hz": 3.0e8 / 8.70e-6},
    "10.8": {"wavelength_um": 10.80, "freq_hz": 3.0e8 / 10.80e-6},
    "12.0": {"wavelength_um": 12.00, "freq_hz": 3.0e8 / 12.00e-6},
}

# The three channels needed for the Dust RGB
RGB_CHANNELS = ("8.7", "10.8", "12.0")

# Number of replicated channel slots in MSGASR
N_CHANNELS = 10

# Satellite filter: WMO ID 57 = Meteosat-10 at 0° (prime service)
SATELLITE_ID = 57

# Geostationary grid settings
# MSG SEVIRI full disc spans approximately ±5,568,748 m from the sub-satellite point.
GEO_EXTENT = 5.5e6  # metres from centre
GEO_NPIX = 800      # number of grid pixels along each axis


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_time(time_str):
    """Parse a user-supplied time string into a datetime object."""
    for fmt in ("%Y%m%d%H%M", "%Y-%m-%d %H:%M", "%Y%m%d/%H%MZ"):
        try:
            return datetime.datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse time '{time_str}'. Use YYYYMMDDHHmm or YYYY-MM-DD HH:MM"
    )


def build_keywords(dt=None):
    """
    Build MetDB keyword list for a single image slot.

    If dt is None, request the most recent hour of data (latest available).
    Otherwise request a 15-minute window centred on the given time.
    """
    if dt is None:
        now = datetime.datetime.now(datetime.UTC)
        end_time = now
        start_time = now - datetime.timedelta(minutes=60)
    else:
        start_time = dt - datetime.timedelta(minutes=30)
        end_time = dt + datetime.timedelta(minutes=30)

    keywords = [
        f"START TIME {start_time:%Y%m%d/%H%MZ} ",
        f"END TIME {end_time:%Y%m%d/%H%MZ} ",
    ]
    return keywords


def identify_channel_indices(obs, target_channels):
    """
    Identify which replication indices correspond to the desired channels
    by matching central frequency values.

    Returns a dict mapping channel key -> replication index (0-based).
    """
    # Get frequencies from the first valid observation
    sample = obs[0]
    freqs = np.ma.filled(sample["CSR_CNTR_FRQY"].astype(np.float64), 0.0)

    channel_map = {}
    for key, info in target_channels.items():
        target_freq = info["freq_hz"]
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        channel_map[key] = idx

    return channel_map


def grid_data_geo(lats, lons, values, extent=GEO_EXTENT, n_pixels=GEO_NPIX):
    """
    Interpolate scattered observations onto a regular grid in native
    geostationary (x, y) coordinates using Delaunay triangulation (linear
    interpolation).  Produces a smooth, gap-free image within the convex
    hull of the observations.

    Returns x_centres, y_centres, gridded_values (2D array).
    """
    geo_crs = ccrs.Geostationary(central_longitude=0.0)
    plate_crs = ccrs.PlateCarree()

    # Transform lat/lon to geostationary x, y (metres)
    xy = geo_crs.transform_points(plate_crs, lons, lats)
    x = xy[:, 0]
    y = xy[:, 1]

    # Filter out points that could not be projected (inf values at limb)
    proj_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)

    x_valid = x[proj_valid]
    y_valid = y[proj_valid]
    v_valid = values[proj_valid]

    # Build the regular output grid
    x_centres = np.linspace(-extent, extent, n_pixels)
    y_centres = np.linspace(-extent, extent, n_pixels)

    # If no valid points, return an all-NaN grid
    if len(v_valid) == 0:
        grid = np.full((n_pixels, n_pixels), np.nan)
        return x_centres, y_centres, grid

    xi, yi = np.meshgrid(x_centres, y_centres)

    # Interpolate using linear (Delaunay) then fill remaining with nearest
    grid = griddata(
        (x_valid, y_valid), v_valid, (xi, yi), method='linear'
    )

    return x_centres, y_centres, grid


def normalise_channel(data, lower=None, upper=None):
    """Normalise data to 0-1 range, clipping at given percentiles."""
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return np.zeros_like(data)
    if lower is None:
        lower = np.percentile(valid, 1)
    if upper is None:
        upper = np.percentile(valid, 99)
    normed = (data - lower) / (upper - lower)
    return np.clip(normed, 0, 1)


def plot_channels(x_grid, y_grid, channel_grids, valid_dt, png_file):
    """
    Plot one or more channels as greyscale full-disc images.

    channel_grids: list of (ch_num, wavelength_um, grid_2d)
    """
    geo_crs = ccrs.Geostationary(central_longitude=0.0)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    n = len(channel_grids)
    if n == 1:
        fig = plt.figure(figsize=(8, 8), facecolor="black")
        axes = [fig.add_subplot(1, 1, 1, projection=geo_crs)]
        title_fontsize = 12
        coast_width = 0.5
    else:
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig = plt.figure(figsize=(ncols * 4, nrows * 4 + 0.6), facecolor="black")
        axes = [
            fig.add_subplot(nrows, ncols, i + 1, projection=geo_crs)
            for i in range(n)
        ]
        title_fontsize = 9
        coast_width = 0.3

    for ax, (ch_num, wavelength_um, grid) in zip(axes, channel_grids):
        ax.set_global()
        ax.set_facecolor("black")

        normed = normalise_channel(grid)
        normed = np.where(np.isnan(normed), 0.0, normed)

        ax.pcolormesh(
            x_mesh, y_mesh, normed,
            cmap="gray", transform=geo_crs,
            shading="auto", vmin=0, vmax=1,
        )
        ax.coastlines(resolution="110m", color="yellow", linewidth=coast_width)

        label = (
            f"Ch {ch_num}  {wavelength_um:.3f} µm"
            if not np.isnan(wavelength_um)
            else f"Ch {ch_num}"
        )
        ax.set_title(label, fontsize=title_fontsize, color="white", pad=3)

    if n > 1:
        fig.suptitle(
            f"All Channels  {valid_dt:%Y-%m-%d %H:%M}Z",
            fontsize=13, color="white", y=1.0,
        )
        plt.subplots_adjust(wspace=0.05, hspace=0.25)
    else:
        current_title = axes[0].get_title()
        axes[0].set_title(
            f"{current_title}\n{valid_dt:%Y-%m-%d %H:%M}Z",
            fontsize=title_fontsize, color="white",
        )

    plt.savefig(png_file, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  Image saved: {png_file}")


def save_netcdf(nc_file, lats, lons, valid_dt, wavelengths_um, frequencies_hz, bt_data):
    """
    Save brightness temperature data to NetCDF.

    bt_data: 2D array (obs x channel) or 1D array (obs,) for single channel.
    """
    if bt_data.ndim == 1:
        bt_data = bt_data[:, np.newaxis]

    n_obs, n_channels = bt_data.shape

    with netCDF4.Dataset(nc_file, "w", format="NETCDF4") as ds:
        ds.title = "MSG SEVIRI Brightness Temperature Data"
        ds.source = f"MetDB ({SUBTYPE})"
        ds.history = f"Created {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M}Z"
        ds.validity_time = f"{valid_dt:%Y-%m-%dT%H:%M:%SZ}"

        ds.createDimension("obs", n_obs)
        ds.createDimension("channel", n_channels)

        v = ds.createVariable("latitude", "f4", ("obs",), zlib=True)
        v.units = "degrees_north"
        v[:] = lats

        v = ds.createVariable("longitude", "f4", ("obs",), zlib=True)
        v.units = "degrees_east"
        v[:] = lons

        v = ds.createVariable("channel_wavelength_um", "f4", ("channel",))
        v.units = "microns"
        v[:] = wavelengths_um

        v = ds.createVariable("channel_frequency_hz", "f8", ("channel",))
        v[:] = frequencies_hz

        v = ds.createVariable(
            "brightness_temperature", "f8", ("obs", "channel"),
            zlib=True, fill_value=-9999.0,
        )
        v.units = "K"
        v[:] = np.where(np.isnan(bt_data), -9999.0, bt_data)

    print(f"  NetCDF saved: {nc_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create a Dust RGB from MSG SEVIRI all-sky data."
    )
    parser.add_argument(
        "--time", "-t",
        type=str,
        default=None,
        help="Validity time (YYYYMMDDHHmm or 'YYYY-MM-DD HH:MM'). "
             "If omitted, retrieves the latest available data.",
    )
    parser.add_argument(
        "--output-nc", "-o",
        type=str,
        default=None,
        help="Output NetCDF filename. Default: msg_rgb_YYYYMMDDHHMMZ.nc",
    )
    parser.add_argument(
        "--output-png", "-p",
        type=str,
        default=None,
        help="Output PNG filename. Default: msg_rgb_YYYYMMDDHHMMZ.png",
    )
    parser.add_argument(
        "--contact",
        type=str,
        default=CONTACT,
        help="Contact email for MetDB request.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["rgb", "single", "all-channels"],
        default="rgb",
        help="Output mode: 'rgb' (Day Natural Colour composite, default), "
             "'single' (greyscale for one channel), "
             "'all-channels' (greyscale overview of every channel).",
    )
    parser.add_argument(
        "--channel", "-c",
        type=int,
        default=1,
        metavar="N",
        help=f"Channel number (1–{N_CHANNELS}) used with --mode single. Default: 1",
    )
    parser.add_argument(
        "--no-netcdf",
        action="store_true",
        default=False,
        help="Disable NetCDF output (all modes save NetCDF by default).",
    )
    args = parser.parse_args()

    # Determine request time
    if args.time is not None:
        req_time = parse_time(args.time)
        print(f"Requesting data for: {req_time:%Y-%m-%d %H:%M}Z")
    else:
        req_time = None
        print("Requesting latest available data...")

    # Build the MetDB request
    keywords = build_keywords(req_time)
    elements = [
        "YEAR", "MNTH", "DAY", "HOUR", "MINT",
        "LTTD", "LNGD", "STLT_IDNY",
        (("CSR_CNTR_FRQY", "BRGTS_TMPR"), N_CHANNELS),
    ]

    print(f"Retrieving {SUBTYPE} data from MetDB...")
    print(f"  Keywords: {keywords}")

    obs = metdb.obs(
        args.contact,
        SUBTYPE,
        keywords=keywords,
        elements=elements,
    )

    if obs is None or len(obs) == 0:
        print("ERROR: No observations returned from MetDB.", file=sys.stderr)
        sys.exit(1)

    print(f"  Retrieved {len(obs)} observations.")

    # Filter to keep only the prime satellite at 0° (Meteosat-10, ID 57)
    sat_ids = np.ma.filled(obs["STLT_IDNY"].astype(np.int32), -1)
    sat_mask = sat_ids == SATELLITE_ID
    if sat_mask.sum() == 0:
        print(f"ERROR: No observations for satellite ID {SATELLITE_ID}.", file=sys.stderr)
        print(f"  Available IDs: {np.unique(sat_ids).tolist()}", file=sys.stderr)
        sys.exit(1)
    obs = obs[sat_mask]
    print(f"  Filtered to satellite {SATELLITE_ID}: {len(obs)} observations.")

    # Extract coordinate and time data
    lats = np.ma.filled(obs["LTTD"].astype(np.float64), np.nan)
    lons = np.ma.filled(obs["LNGD"].astype(np.float64), np.nan)
    years = np.ma.filled(obs["YEAR"].astype(np.int32), 0)
    months = np.ma.filled(obs["MNTH"].astype(np.int32), 0)
    days = np.ma.filled(obs["DAY"].astype(np.int32), 0)
    hours = np.ma.filled(obs["HOUR"].astype(np.int32), 0)
    minutes = np.ma.filled(obs["MINT"].astype(np.int32), 0)

    # Determine the representative validity time from the data
    valid_dt = datetime.datetime(
        int(np.median(years)), int(np.median(months)), int(np.median(days)),
        int(np.median(hours)), int(np.median(minutes))
    )
    print(f"  Data validity time: {valid_dt:%Y-%m-%d %H:%M}Z")

    # Extract channel frequencies and wavelengths (used by all modes)
    all_freqs = np.ma.filled(obs[0]["CSR_CNTR_FRQY"].astype(np.float64), 0.0)
    all_wavelengths_um = [
        float(3.0e8 / f * 1e6) if f > 1.0 else float("nan")
        for f in all_freqs
    ]
    print(f"  Channel wavelengths (µm): {[f'{w:.3f}' for w in all_wavelengths_um]}")

    time_str = f"{valid_dt:%Y%m%d%H%M}Z"
    mode = args.mode

    # --- Mode: single channel ---
    if mode == "single":
        ch_num = args.channel
        if not (1 <= ch_num <= N_CHANNELS):
            print(f"ERROR: --channel must be 1\u2013{N_CHANNELS}", file=sys.stderr)
            sys.exit(1)
        ch_idx = ch_num - 1
        bt = np.ma.filled(obs["BRGTS_TMPR"][:, ch_idx].astype(np.float64), np.nan)
        print(f"  Gridding channel {ch_num} ({all_wavelengths_um[ch_idx]:.3f} \u00b5m)...")
        print(f"    Valid BT values: {int(np.sum(np.isfinite(bt)))}/{len(bt)}")
        x_grid, y_grid, grid = grid_data_geo(lats, lons, bt)

        if not args.no_netcdf:
            nc_file = args.output_nc or f"msg_ch{ch_num:02d}_{time_str}.nc"
            save_netcdf(
                nc_file, lats, lons, valid_dt,
                [all_wavelengths_um[ch_idx]], [float(all_freqs[ch_idx])], bt,
            )

        png_file = args.output_png or f"msg_ch{ch_num:02d}_{time_str}.png"
        plot_channels(
            x_grid, y_grid,
            [(ch_num, all_wavelengths_um[ch_idx], grid)],
            valid_dt, png_file,
        )
        print("Done.")
        return

    # --- Mode: all channels ---
    if mode == "all-channels":
        print("  Gridding all channels...")

        channel_grids = []
        x_grid = y_grid = None
        for ch_idx in range(N_CHANNELS):
            bt = np.ma.filled(obs["BRGTS_TMPR"][:, ch_idx].astype(np.float64), np.nan)
            n_valid_bt = int(np.sum(np.isfinite(bt)))
            wl = all_wavelengths_um[ch_idx]
            print(f"    Ch {ch_idx+1:2d} ({wl:8.3f} µm): "
                  f"BT valid={n_valid_bt:>8d}/{len(bt)}, "
                  f"BT range=[{np.nanmin(bt) if n_valid_bt else 'N/A'}, "
                  f"{np.nanmax(bt) if n_valid_bt else 'N/A'}]")

            _x, _y, grid = grid_data_geo(lats, lons, bt)
            if x_grid is None:
                x_grid, y_grid = _x, _y
            channel_grids.append((ch_idx + 1, wl, grid))

        if not args.no_netcdf:
            nc_file = args.output_nc or f"msg_all_channels_{time_str}.nc"
            bt_all = np.ma.filled(obs["BRGTS_TMPR"].astype(np.float64), np.nan)
            save_netcdf(
                nc_file, lats, lons, valid_dt,
                all_wavelengths_um, all_freqs.tolist(), bt_all,
            )

        png_file = args.output_png or f"msg_all_channels_{time_str}.png"
        plot_channels(x_grid, y_grid, channel_grids, valid_dt, png_file)
        print("Done.")
        return

    # --- Mode: RGB (Dust) ---
    channel_map = identify_channel_indices(obs, CHANNELS)
    print(f"  Channel index mapping: {channel_map}")

    brightness_temp = {}
    for key in RGB_CHANNELS:
        idx = channel_map[key]
        brightness_temp[key] = np.ma.filled(obs["BRGTS_TMPR"][:, idx].astype(np.float64), np.nan)

    print("Gridding data...")
    grids = {}
    x_grid = y_grid = None
    for key in RGB_CHANNELS:
        x_grid, y_grid, grids[key] = grid_data_geo(lats, lons, brightness_temp[key])

    # Build the Dust RGB composite (EUMETSAT recipe)
    # Red:   BT(12.0) - BT(10.8),  range [-4, 2] K
    # Green: BT(10.8) - BT(8.7),   range [0, 15] K
    # Blue:  BT(10.8),             range [261, 289] K
    diff_r = grids["12.0"] - grids["10.8"]
    diff_g = grids["10.8"] - grids["8.7"]
    bt_b = grids["10.8"]

    r = np.clip((diff_r - (-4.0)) / (2.0 - (-4.0)), 0, 1)
    g = np.clip((diff_g - 0.0) / (15.0 - 0.0), 0, 1)
    b = np.clip((bt_b - 261.0) / (289.0 - 261.0), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)

    no_data = (
        np.isnan(grids["8.7"]) | np.isnan(grids["10.8"]) |
        np.isnan(grids["12.0"])
    )
    rgb[no_data] = 0

    if not args.no_netcdf:
        nc_file = args.output_nc or f"msg_dust_{time_str}.nc"
        bt_data = np.column_stack([brightness_temp[k] for k in RGB_CHANNELS])
        save_netcdf(
            nc_file, lats, lons, valid_dt,
            [CHANNELS[k]["wavelength_um"] for k in RGB_CHANNELS],
            [CHANNELS[k]["freq_hz"] for k in RGB_CHANNELS],
            bt_data,
        )

    # --- Create the RGB image ---
    png_file = args.output_png or f"msg_dust_{time_str}.png"
    print(f"Creating RGB image: {png_file}")

    geo_crs = ccrs.Geostationary(central_longitude=0.0)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=geo_crs)
    ax.set_global()

    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    ax.pcolormesh(
        x_mesh, y_mesh, rgb,
        transform=geo_crs,
        shading="auto",
    )
    ax.coastlines(resolution="110m", color="cyan", linewidth=0.5)
    ax.set_title(
        f"MSG SEVIRI Dust RGB\n"
        f"R: BT(12.0)-BT(10.8), G: BT(10.8)-BT(8.7), B: BT(10.8)\n"
        f"{valid_dt:%Y-%m-%d %H:%M}Z",
        fontsize=11, color="white",
    )

    plt.savefig(png_file, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  Image saved: {png_file}")
    print("Done.")


if __name__ == "__main__":
    main()
