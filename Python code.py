import numpy as np
import openeo
from openeo.processes import mean

# 1. Connect to Copernicus Data Space Ecosystem
connection = openeo.connect("https://openeo.dataspace.copernicus.eu")

# In VS Code, authenticate_oidc opens a local browser tab to log in directly
connection.authenticate_oidc()

# 2. Coordinates for all 5 study areas
sites = [
    {"name": "Veerapandi", "lat": 9.966636, "lon": 77.429217},
    {"name": "Jagannathapuram", "lat": 13.261128, "lon": 80.168365},
    {"name": "Cuddalore", "lat": 11.844412, "lon": 79.735178},
    {"name": "Viswanthapuram", "lat": 11.805000, "lon": 79.659944},
    {"name": "Peddamungalachedu", "lat": 16.495961, "lon": 77.894172},
]

# Comparison time windows
time_windows = {
    "before": ["2020-06-01", "2020-06-30"],
    "after": ["2025-06-01", "2025-06-30"],
}


def get_roi_polygon(lat: float, lon: float, buffer: float = 0.0005) -> dict:
    """Builds a GeoJSON polygon bounding box around a center point."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - buffer, lat - buffer],
                [lon + buffer, lat - buffer],
                [lon + buffer, lat + buffer],
                [lon - buffer, lat + buffer],
                [lon - buffer, lat - buffer],
            ]
        ],
    }


def analyze_period(conn, lat: float, lon: float, dates: list, roi: dict):
    """Loads bands B03 (Green), B04 (Red), B08 (NIR), computes indices, and aggregates across ROI."""
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={
            "west": lon - 0.002,
            "south": lat - 0.002,
            "east": lon + 0.002,
            "north": lat + 0.002,
        },
        temporal_extent=dates,
        bands=["B03", "B04", "B08"],
    )

    b3 = cube.band("B03")
    b4 = cube.band("B04")
    b8 = cube.band("B08")

    # Spectral index formulas
    ndvi_cube = (b8 - b4) / (b8 + b4)
    ndwi_cube = (b3 - b8) / (b3 + b8)

    # Spatial aggregation over ROI
    ndvi_raw = ndvi_cube.aggregate_spatial(geometries=roi, reducer=mean).execute()
    ndwi_raw = ndwi_cube.aggregate_spatial(geometries=roi, reducer=mean).execute()

    # Parse openEO response: {timestamp: [[value]]}
    ndvi_clean = [
        val[0][0]
        for val in ndvi_raw.values()
        if val and val[0] and val[0][0] is not None
    ]
    ndwi_clean = [
        val[0][0]
        for val in ndwi_raw.values()
        if val and val[0] and val[0][0] is not None
    ]

    mean_ndvi = float(np.mean(ndvi_clean)) if ndvi_clean else float("nan")
    mean_ndwi = float(np.mean(ndwi_clean)) if ndwi_clean else float("nan")

    return mean_ndvi, mean_ndwi


def main():
    print("Starting automated Sentinel-2 analysis...")

    for site in sites:
        name, lat, lon = site["name"], site["lat"], site["lon"]
        roi = get_roi_polygon(lat, lon)

        print(f"\nProcessing: {name} ({lat}, {lon})")

        try:
            ndvi_before, ndwi_before = analyze_period(
                connection, lat, lon, time_windows["before"], roi
            )
            ndvi_after, ndwi_after = analyze_period(
                connection, lat, lon, time_windows["after"], roi
            )

            ndvi_delta = ndvi_after - ndvi_before
            ndwi_delta = ndwi_after - ndwi_before

            print(f"\n{'='*30}")
            print(f"Results for: {name.upper()}")
            print(f"{'='*30}")
            print(f"Before (2020) NDVI : {ndvi_before:.4f}")
            print(f"After  (2025) NDVI : {ndvi_after:.4f}")
            print(f"NDVI Difference    : {ndvi_delta:+.4f}")
            print(
                f"NDVI Status        : {'Increased' if ndvi_delta > 0 else 'Decreased'}"
            )
            print("-" * 30)
            print(f"Before (2020) NDWI : {ndwi_before:.4f}")
            print(f"After  (2025) NDWI : {ndwi_after:.4f}")
            print(f"NDWI Difference    : {ndwi_delta:+.4f}")
            print(
                f"NDWI Status        : {'Increased' if ndwi_delta > 0 else 'Decreased'}"
            )
            print(f"{'='*30}\n")

        except Exception as e:
            print(f"Error processing {name}: {e}")


if __name__ == "__main__":
    main()
