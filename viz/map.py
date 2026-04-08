"""
viz/map.py — Plotly 3D orthographic globe flight risk map for AeroSignal.

Renders a dark animated globe with a great-circle flight arc, airport markers,
per-region risk indicators, a risk overlay circle, and an animated plane trace.
Returns a Plotly Figure or an HTML string for embedding in Streamlit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import numpy as np
import plotly.graph_objects as go

from rag.signals import AIRPORT_COORDS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION_CENTERS: dict[str, tuple[float, float]] = {
    "Gulf":                  (26.0,  50.5),
    "Eastern Europe":        (50.0,  30.0),
    "Russia":                (60.0,  60.0),
    "Eastern Mediterranean": (36.0,  33.0),
    "Pakistan":              (30.0,  69.0),
    "India":                 (22.0,  80.0),
    "Southeast Asia":        (10.0, 110.0),
    "East Asia":             (35.0, 125.0),
    "North Africa":          (25.0,  15.0),
    "Central Asia":          (45.0,  65.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_risk_color(score: float) -> str:
    """
    Return the hex colour for a given 0-100 risk score.

    Args:
        score: Risk score in range 0-100.

    Returns:
        One of: #00C897 (teal), #F5A623 (amber),
                #E8593C (coral red), #C0392B (deep red).
    """
    if score <= 25:
        return "#00C897"
    elif score <= 50:
        return "#F5A623"
    elif score <= 75:
        return "#E8593C"
    else:
        return "#C0392B"


def _risk_label(score: float) -> str:
    """Return Low / Moderate / High / Very High for a score."""
    if score <= 25:   return "Low"
    elif score <= 50: return "Moderate"
    elif score <= 75: return "High"
    else:             return "Very High"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """
    Convert a hex colour string to an rgba() CSS string.

    Args:
        hex_color: Hex string, e.g. "#F5A623".
        alpha:     Opacity in range 0-1.

    Returns:
        RGBA string, e.g. "rgba(245,166,35,0.08)".
    """
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def interpolate_arc(
    start: tuple[float, float],
    end: tuple[float, float],
    steps: int = 50,
) -> tuple[list[float], list[float]]:
    """
    Interpolate great-circle arc points between two (lat, lon) positions.

    Uses spherical linear interpolation (slerp) via numpy linspace for
    accurate placement on the Earth's surface.

    Args:
        start: (lat, lon) of the departure point in degrees.
        end:   (lat, lon) of the arrival point in degrees.
        steps: Number of interpolation points (default 50).

    Returns:
        Tuple of (lats, lons) — two lists of floats in degrees.
    """
    def to_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
        lr = math.radians(lat_deg)
        gr = math.radians(lon_deg)
        return np.array([
            math.cos(lr) * math.cos(gr),
            math.cos(lr) * math.sin(gr),
            math.sin(lr),
        ])

    p1 = to_xyz(start[0], start[1])
    p2 = to_xyz(end[0],   end[1])
    dot = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
    omega = math.acos(dot)

    lats: list[float] = []
    lons: list[float] = []
    for t in np.linspace(0, 1, steps):
        if abs(omega) < 1e-10:
            pt = p1
        else:
            pt = (math.sin((1 - t) * omega) * p1 +
                  math.sin(t * omega) * p2) / math.sin(omega)
        lat = math.degrees(math.asin(float(np.clip(pt[2], -1.0, 1.0))))
        lon = math.degrees(math.atan2(float(pt[1]), float(pt[0])))
        lats.append(lat)
        lons.append(lon)

    return lats, lons


def _circle_points(
    center_lat: float,
    center_lon: float,
    radius_deg: float = 8.0,
    n: int = 72,
) -> tuple[list[float], list[float]]:
    """
    Generate lat/lon points forming an approximate circle around a centre.

    Args:
        center_lat:  Centre latitude in degrees.
        center_lon:  Centre longitude in degrees.
        radius_deg:  Approximate radius in degrees (default 8 ~= 890 km).
        n:           Number of perimeter points (default 72).

    Returns:
        Tuple of (lats, lons).
    """
    theta = np.linspace(0, 2 * np.pi, n)
    lats = [center_lat + radius_deg * np.cos(t) for t in theta]
    lons = [center_lon + radius_deg * np.sin(t) for t in theta]
    return lats, lons


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_map(origin: str, destination: str, risk_result: dict) -> go.Figure:
    """
    Build a Plotly orthographic globe figure for a scored flight route.

    Traces (bottom to top):
      1. Glow arc     — wide, low-opacity Scattergeo line
      2. Core arc     — narrow, high-opacity Scattergeo line
      3. Risk circle  — filled semi-transparent circle over riskiest region
      4. Waypoints    — coloured dots for regions with score > 0
      5. Airports     — origin/destination markers with IATA labels
      6. Plane marker — animated along the arc via fig.frames

    Args:
        origin:      Departure IATA code, e.g. "YYZ".
        destination: Arrival IATA code, e.g. "DXB".
        risk_result: Dict from score_route() with keys: score, label,
                     riskiest_region, waypoints, breakdown.

    Returns:
        Plotly Figure with animation frames and Play/Pause buttons.
    """
    origin_coords = AIRPORT_COORDS.get(origin)
    dest_coords   = AIRPORT_COORDS.get(destination)

    score     = risk_result.get("score", 0.0)
    label     = risk_result.get("label", "Unknown")
    riskiest  = risk_result.get("riskiest_region", "Unknown")
    breakdown = risk_result.get("breakdown", {})
    arc_color = get_risk_color(score)

    if origin_coords and dest_coords:
        # Midpoint between airports — e.g. YYZ(-79.63)+DXB(55.36) = lon -12, lat 34
        center_lat = round((origin_coords[0] + dest_coords[0]) / 2, 1)
        center_lon = round((origin_coords[1] + dest_coords[1]) / 2, 1)
    else:
        center_lat, center_lon = 20.0, 0.0

    traces: list[go.BaseTraceType] = []

    # --- Arc traces ---
    arc_lats: list[float] = []
    arc_lons: list[float] = []
    if origin_coords and dest_coords:
        arc_lats, arc_lons = interpolate_arc(
            (origin_coords[0], origin_coords[1]),
            (dest_coords[0],   dest_coords[1]),
            steps=50,
        )
        # Outer glow
        traces.append(go.Scattergeo(
            lat=arc_lats, lon=arc_lons,
            mode="lines",
            line=dict(width=6, color=arc_color),
            opacity=0.15,
            showlegend=False,
            hoverinfo="skip",
            name="arc_glow",
        ))
        # Core line
        traces.append(go.Scattergeo(
            lat=arc_lats, lon=arc_lons,
            mode="lines",
            line=dict(width=2, color=arc_color),
            opacity=0.8,
            showlegend=False,
            hoverinfo="skip",
            name="arc_core",
        ))

    # --- Risk circle over riskiest region ---
    riskiest_center = REGION_CENTERS.get(riskiest)
    if riskiest_center:
        circ_lats, circ_lons = _circle_points(
            riskiest_center[0], riskiest_center[1]
        )
        traces.append(go.Scattergeo(
            lat=circ_lats, lon=circ_lons,
            mode="lines",
            line=dict(width=1, color=arc_color),
            fill="toself",
            fillcolor=_hex_to_rgba(arc_color, 0.08),
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
            name="risk_circle",
        ))

    # --- Waypoint markers (score > 0 only) ---
    wp_lats:   list[float] = []
    wp_lons:   list[float] = []
    wp_texts:  list[str]   = []
    wp_colors: list[str]   = []
    wp_hover:  list[str]   = []

    for region, region_score in breakdown.items():
        if region_score <= 0:
            continue
        center = REGION_CENTERS.get(region)
        if not center:
            continue
        # Skip markers that overlap the destination airport marker
        if dest_coords:
            dist = math.sqrt(
                (center[0] - dest_coords[0]) ** 2 +
                (center[1] - dest_coords[1]) ** 2
            )
            if dist < 5.0:
                continue
        wp_lats.append(center[0])
        wp_lons.append(center[1])
        wp_texts.append(f"{region}: {region_score:.0f}")
        wp_colors.append(get_risk_color(region_score))
        wp_hover.append(
            f"<b>{region}</b><br>"
            f"Score: {region_score:.0f}/100<br>"
            f"Risk: {_risk_label(region_score)}"
        )

    if wp_lats:
        traces.append(go.Scattergeo(
            lat=wp_lats, lon=wp_lons,
            mode="markers+text",
            marker=dict(
                size=8,
                color=wp_colors,
                symbol="circle",
                line=dict(width=1.5, color="white"),
            ),
            text=wp_texts,
            textposition="top right",
            textfont=dict(color="white", size=9),
            hovertext=wp_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
            name="waypoints",
        ))

    # --- Airport markers ---
    if origin_coords and dest_coords:
        traces.append(go.Scattergeo(
            lat=[origin_coords[0], dest_coords[0]],
            lon=[origin_coords[1], dest_coords[1]],
            mode="markers+text",
            marker=dict(
                size=10,
                color=arc_color,
                symbol="circle",
                line=dict(width=2, color="white"),
            ),
            text=[origin, destination],
            textposition=["top center", "top center"],
            textfont=dict(color="white", size=11, family="monospace"),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
            name="airports",
        ))

    # --- Plane marker (initial position, updated by frames) ---
    plane_lat = arc_lats[0] if arc_lats else center_lat
    plane_lon = arc_lons[0] if arc_lons else center_lon
    traces.append(go.Scattergeo(
        lat=[plane_lat], lon=[plane_lon],
        mode="markers",
        marker=dict(
            size=8,
            color="white",
            symbol="circle",
            line=dict(width=1, color=arc_color),
        ),
        hoverinfo="skip",
        showlegend=False,
        name="plane",
    ))
    plane_trace_idx = len(traces) - 1

    fig = go.Figure(data=traces)

    # --- Animation frames ---
    fig.frames = [
        go.Frame(
            data=[go.Scattergeo(
                lat=[arc_lats[i]], lon=[arc_lons[i]],
                mode="markers",
                marker=dict(
                    size=8,
                    color="white",
                    symbol="circle",
                    line=dict(width=1, color=arc_color),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="plane",
            )],
            traces=[plane_trace_idx],
            name=str(i),
        )
        for i in range(len(arc_lats))
    ]

    # --- Layout ---
    fig.update_layout(
        paper_bgcolor="#060610",
        plot_bgcolor="#060610",
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        geo=dict(
            showland=True,      landcolor="#1a2744",
            showocean=True,     oceancolor="#060d1f",
            showlakes=False,
            showcountries=True, countrycolor="#1e2a3a",
            showframe=False,
            bgcolor="#060610",
            projection_type="orthographic",
            projection_rotation=dict(
                lon=center_lon,
                lat=center_lat,
                roll=0,
            ),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.02,
            x=0.5,
            xanchor="center",
            yanchor="top",
            bgcolor="#1a1a2e",
            bordercolor=arc_color,
            font=dict(color="white", size=11),
            buttons=[
                dict(
                    label="\u25b6 Fly",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 80, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }],
                ),
                dict(
                    label="\u23f8 Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                ),
            ],
        )],
    )

    return fig


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def build_map_html(origin: str, destination: str, risk_result: dict) -> str:
    """
    Build the globe map and return it as an embeddable HTML string.

    Args:
        origin:      Departure IATA code, e.g. "YYZ".
        destination: Arrival IATA code, e.g. "DXB".
        risk_result: Dict from score_route().

    Returns:
        HTML string for use with st.components.v1.html() or saved to file.
    """
    fig = build_map(origin, destination, risk_result)
    return fig.to_html(full_html=False, include_plotlyjs=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_result = {
        "route": "YYZ-DXB",
        "score": 35.0,
        "label": "Moderate",
        "riskiest_region": "Gulf",
        "waypoints": ["Eastern Europe", "Russia", "Gulf"],
        "breakdown": {"Eastern Europe": 0.0, "Russia": 0.0, "Gulf": 35.0},
    }
    html = build_map_html("YYZ", "DXB", sample_result)
    with open("test_map.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Globe map saved — open test_map.html in browser")
