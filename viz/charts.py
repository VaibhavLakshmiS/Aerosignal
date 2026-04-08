"""
viz/charts.py — Plotly chart builders for the AeroSignal dashboard.

Provides three chart functions used by app.py: forecast_chart (7-day risk
and fare timeline), oil_chart (30-day Brent crude area chart), and
cascade_chart (horizontal bar chart of hub demand/fare impacts).
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BG     = "#0a0a0f"
_PANEL  = "#0f0f1a"
_GRID   = "#1e1e2e"
_TEXT   = "#ffffff"
_ACCENT = "#E8593C"
_PURPLE = "#a78bfa"


def _score_color(score: float) -> str:
    if score <= 25:   return "#00C897"
    elif score <= 50: return "#F5A623"
    elif score <= 75: return "#E8593C"
    else:             return "#C0392B"


def forecast_chart(forecast: list[dict]) -> go.Figure:
    """
    Build a dual-axis 7-day risk score bar + projected fare line chart.

    Args:
        forecast: Output list from forecast_route().
    """
    dates     = [d["date"]           for d in forecast]
    scores    = [d["score"]          for d in forecast]
    fares     = [d["projected_fare"] for d in forecast]
    colors    = [_score_color(s)     for s in scores]
    has_fares = any(f is not None for f in fares)

    fig = make_subplots(specs=[[{"secondary_y": True}]]) if has_fares else go.Figure()

    fig.add_trace(
        go.Bar(
            x=dates, y=scores,
            marker_color=colors,
            name="Risk Score",
            hovertemplate="<b>%{x}</b><br>Risk: %{y}/100<extra></extra>",
        ),
        **({"secondary_y": False} if has_fares else {}),
    )

    if has_fares:
        fig.add_trace(
            go.Scatter(
                x=dates, y=fares,
                mode="lines+markers",
                line=dict(color=_PURPLE, width=2),
                marker=dict(size=6, color=_PURPLE),
                name="Projected Fare (USD)",
                hovertemplate="<b>%{x}</b><br>Fare: $%{y:.0f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Projected Fare (USD)",
            secondary_y=True,
            gridcolor=_GRID,
            color=_PURPLE,
        )

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        font=dict(color=_TEXT),
        legend=dict(bgcolor=_PANEL, bordercolor=_GRID, font=dict(color=_TEXT)),
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(gridcolor=_GRID, color=_TEXT),
        yaxis=dict(title="Risk Score /100", gridcolor=_GRID, color=_TEXT, range=[0, 100]),
        height=320, bargap=0.3,
    )
    return fig


def oil_chart(prices: list[dict]) -> go.Figure:
    """
    Build a 30-day Brent crude close-price area chart.

    Args:
        prices: Output list from fetch_oil_prices().
    """
    sorted_prices = sorted(prices, key=lambda p: p["date"])
    dates  = [str(p["date"]) for p in sorted_prices]
    closes = [p["close"]     for p in sorted_prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(232,89,60,0.08)",
        line=dict(color=_ACCENT, width=2),
        name="Brent Crude (USD/bbl)",
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}/bbl<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        font=dict(color=_TEXT),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor=_GRID, color=_TEXT),
        yaxis=dict(gridcolor=_GRID, color=_TEXT, title="USD / bbl"),
        showlegend=False, height=300,
    )
    return fig


def cascade_chart(cascades: list[dict]) -> go.Figure:
    """
    Build a grouped horizontal bar chart of cascade hub impacts.

    Args:
        cascades: Output list from detect_cascade_risk().
    """
    hub_names = [c["affected_hub_name"]   for c in cascades]
    demand    = [c["demand_increase_pct"] for c in cascades]
    fares     = [c["fare_impact_pct"]     for c in cascades]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=hub_names, x=demand, orientation="h",
        name="Demand increase %",
        marker_color="#F5A623",
        hovertemplate="<b>%{y}</b><br>Demand: +%{x}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=hub_names, x=fares, orientation="h",
        name="Fare impact %",
        marker_color=_ACCENT,
        hovertemplate="<b>%{y}</b><br>Fares: +%{x}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        font=dict(color=_TEXT),
        barmode="group",
        legend=dict(bgcolor=_PANEL, bordercolor=_GRID, font=dict(color=_TEXT)),
        margin=dict(l=120, r=20, t=20, b=40),
        xaxis=dict(gridcolor=_GRID, color=_TEXT, title="% Impact"),
        yaxis=dict(gridcolor=_GRID, color=_TEXT),
        height=max(260, len(cascades) * 55),
    )
    return fig
