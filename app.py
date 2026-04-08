"""
app.py — Streamlit entry point for the AeroSignal dashboard.

Premium dark-themed single-page app for geopolitical flight risk intelligence.
Orchestrates the agent pipeline (rag/chain.py), risk scoring (rag/signals.py),
and visualisations (viz/map.py, viz/charts.py) into a cohesive UI for
travellers and analysts.
"""

import html
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import streamlit.components.v1 as components

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="AeroSignal",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #0a0a0f;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}
.stMarkdown, p, h1, h2, h3 {
    color: #ffffff !important;
}
.stTextInput input {
    background-color: #16161f !important;
    border: 1px solid #2a2a3a !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
.stButton button {
    background: linear-gradient(135deg, #E8593C, #C0392B) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 12px !important;
}
[data-testid="stMetric"] {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 16px;
}
.block-container {
    padding-top: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-bottom: 1px solid #1e1e2e;
}
.stTabs [data-baseweb="tab"] {
    color: #555555;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #E8593C !important;
    border-bottom: 2px solid #E8593C !important;
}
.streamlit-expanderHeader {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
}
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Imports after CSS (avoids import-time flicker on slow machines) ───────────
from data.fetch_events import fetch_events
from data.fetch_flights import fetch_flights
from data.fetch_prices import fetch_oil_prices, get_oil_trend
from rag.chain import run_agent
from rag.signals import (
    AIRPORT_COORDS, get_waypoints, score_route,
    forecast_route, get_best_booking_day,
)
from viz.charts import cascade_chart, forecast_chart, oil_chart
from viz.map import build_map_html, get_risk_color


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score <= 25:   return "#00C897"
    elif score <= 50: return "#F5A623"
    elif score <= 75: return "#E8593C"
    else:             return "#C0392B"


def _recommendation(score: float) -> tuple[str, str]:
    """Return (message, hex_color) for the recommendation banner."""
    if score <= 25:
        return (
            "Route appears safe. No significant geopolitical risk signals detected.",
            "#00C897",
        )
    elif score <= 50:
        return (
            "Fly with awareness. Monitor the developing situation before booking.",
            "#F5A623",
        )
    elif score <= 75:
        return (
            "High risk detected. Consider alternatives or delay booking until the situation improves.",
            "#E8593C",
        )
    else:
        return (
            "Very high risk. Strong recommendation to avoid this route or reroute through safer airspace.",
            "#C0392B",
        )


def _card(label: str, value: str, sub: str = "") -> str:
    """Return HTML for a styled metric card."""
    sub_html = (
        f'<p style="color:#6b7280;font-size:12px;margin:2px 0 0 0">{sub}</p>'
        if sub else ""
    )
    return (
        f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:10px;'
        f'padding:16px;text-align:center">'
        f'<p style="color:#6b7280;font-size:11px;margin:0;text-transform:uppercase;'
        f'letter-spacing:0.07em">{label}</p>'
        f'<p style="color:#ffffff;font-size:22px;font-weight:600;margin:4px 0 0 0">{value}</p>'
        f'{sub_html}</div>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<h1 style="color:#ffffff;font-size:28px;font-weight:700;margin:0">AeroSignal</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280;font-size:13px;margin:4px 0 16px 0">'
        'Geopolitical Flight Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    origin_input      = st.text_input("Origin",      placeholder="YYZ (Toronto)")
    destination_input = st.text_input("Destination", placeholder="DXB (Dubai)")
    travel_date       = st.date_input("Travel date")
    clicked           = st.button("Analyze Route")

    if "data" in st.session_state:
        st.divider()
        _d    = st.session_state["data"]
        _rr   = _d["risk_result"]
        _tr   = _d["trend"]
        _sign = "+" if _tr["price_change_pct"] >= 0 else ""
        st.metric("Risk Score",  f"{_rr['score']:.0f}/100")
        st.metric("Oil Price",   f"${_tr['current_price']:.2f}/bbl")
        st.metric("30d Trend",   f"{_sign}{_tr['price_change_pct']:.1f}%")
        st.metric("News Events", str(_d["event_count"]))


# ── Analysis trigger ──────────────────────────────────────────────────────────

if clicked:
    origin      = origin_input.strip().upper()
    destination = destination_input.strip().upper()

    if not origin or not destination:
        st.error("Please enter both an origin and destination IATA code.")

    elif origin not in AIRPORT_COORDS or destination not in AIRPORT_COORDS:
        unknown = origin if origin not in AIRPORT_COORDS else destination
        st.error(
            f"Airport code **{unknown}** not recognised. "
            "Please use a valid IATA code (e.g. YYZ, DXB)."
        )

    else:
        with st.spinner("Agent analysing route..."):
            try:
                prices = fetch_oil_prices()
                trend  = get_oil_trend(prices)

                flights      = fetch_flights(origin, destination, travel_date.strftime("%Y-%m-%d"))
                current_fare = flights[0]["price_usd"] if flights else None

                query = (
                    f"Analyse the geopolitical risk for flying {origin} to {destination}. "
                    "Include 7-day forecast and cascade risks."
                )
                agent_response = run_agent(query)

                # Fetch events for meaningful chart scores and sidebar event count
                waypoints  = get_waypoints(origin, destination)
                all_events: list[dict] = []
                for region in waypoints:
                    try:
                        all_events.extend(fetch_events(region))
                    except Exception:
                        pass

                risk_result = score_route(origin, destination, all_events, trend, prices)
                forecast    = forecast_route(origin, destination, risk_result, prices, current_fare)

                map_html = build_map_html(origin, destination, risk_result)
                f_chart  = forecast_chart(forecast)
                o_chart  = oil_chart(prices)
                c_chart  = (
                    cascade_chart(risk_result["cascade_risks"])
                    if risk_result["has_cascade"] else None
                )

                st.session_state["data"] = {
                    "origin":         origin,
                    "destination":    destination,
                    "risk_result":    risk_result,
                    "forecast":       forecast,
                    "trend":          trend,
                    "prices":         prices,
                    "agent_response": agent_response,
                    "map_html":       map_html,
                    "f_chart":        f_chart,
                    "o_chart":        o_chart,
                    "c_chart":        c_chart,
                    "current_fare":   current_fare,
                    "event_count":    len(all_events),
                }

            except Exception as e:
                st.warning(
                    f"Analysis partially complete — some data may be unavailable. {e}"
                )


# ── Main area ─────────────────────────────────────────────────────────────────

if "data" not in st.session_state:

    # ── Hero (pre-analysis) ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 48px 20px">
        <h1 style="font-size:48px;font-weight:700;color:#ffffff;
                   margin:0;line-height:1.15">
            Should you book<br>this flight today?
        </h1>
        <p style="color:#6b7280;font-size:18px;margin:24px auto 0 auto;
                  max-width:520px;line-height:1.65">
            AeroSignal analyses geopolitical events, oil markets and airspace
            risk to tell you the optimal time to book.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    feature_cards = [
        (
            "7-Day Forecast",
            "Risk timeline with projected fare movement, oil momentum, "
            "and best booking day — updated on every analysis.",
        ),
        (
            "Cascade Detection",
            "Second-order hub impacts. When Gulf airspace is disrupted, "
            "Istanbul and Frankfurt fares spike. We surface that.",
        ),
        (
            "AI Agent",
            "Live Groq Llama 3.3 reasoning with GDELT news, Brent crude "
            "data, and historical route scores.",
        ),
    ]
    for col, (title, desc) in zip([c1, c2, c3], feature_cards):
        with col:
            st.markdown(
                f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;'
                f'border-radius:12px;padding:24px;min-height:130px">'
                f'<p style="color:#E8593C;font-size:13px;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin:0">{title}</p>'
                f'<p style="color:#9ca3af;font-size:14px;margin:12px 0 0 0;'
                f'line-height:1.55">{desc}</p></div>',
                unsafe_allow_html=True,
            )

else:

    # ── Full analysis display ────────────────────────────────────────────────
    d           = st.session_state["data"]
    origin      = d["origin"]
    destination = d["destination"]
    rr          = d["risk_result"]
    score       = rr["score"]
    color       = _score_color(score)
    rec_text, rec_color = _recommendation(score)
    best_day    = get_best_booking_day(d["forecast"])
    peak_day    = max(d["forecast"], key=lambda x: x["score"])
    trend       = d["trend"]

    # Hero result card
    fare_str = f"${d['current_fare']:.0f}" if d["current_fare"] else "N/A"
    st.markdown(
        f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:16px;'
        f'padding:28px 32px;margin-bottom:16px">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:16px">'
        f'<div>'
        f'<p style="color:#6b7280;font-size:12px;margin:0;text-transform:uppercase;'
        f'letter-spacing:0.08em">Route Analysis</p>'
        f'<h2 style="color:#ffffff;font-size:32px;font-weight:700;margin:4px 0">'
        f'{origin} &rarr; {destination}</h2>'
        f'<p style="color:#9ca3af;font-size:14px;margin:4px 0 0 0">'
        f'Riskiest region: <span style="color:{color};font-weight:600">'
        f'{rr["riskiest_region"]}</span>'
        f'&nbsp;&bull;&nbsp;Current fare: {fare_str}</p>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<p style="font-size:56px;font-weight:700;color:{color};margin:0;line-height:1">'
        f'{score:.0f}</p>'
        f'<p style="color:{color};font-size:14px;font-weight:600;margin:0">{rr["label"]} Risk</p>'
        f'<p style="color:#6b7280;font-size:12px;margin:2px 0 0 0">out of 100</p>'
        f'</div></div>'
        f'<div style="margin-top:16px;padding:12px 16px;background:{rec_color}1a;'
        f'border-left:3px solid {rec_color};border-radius:6px">'
        f'<p style="color:{rec_color};margin:0;font-size:14px;font-weight:500">'
        f'{rec_text}</p></div></div>',
        unsafe_allow_html=True,
    )

    # Four metric columns
    mc1, mc2, mc3, mc4 = st.columns(4)
    best_fare_str = f"${best_day['projected_fare']:.0f}" if best_day["projected_fare"] else "—"

    if d["current_fare"] and best_day["projected_fare"]:
        diff       = best_day["projected_fare"] - d["current_fare"]
        impact_str = f"+${diff:.0f}" if diff >= 0 else f"-${abs(diff):.0f}"
    else:
        impact_str = "N/A"

    oil_sign = "+" if trend["price_change_pct"] >= 0 else ""

    with mc1:
        st.markdown(_card("Best Booking Day", best_day["date"], best_fare_str), unsafe_allow_html=True)
    with mc2:
        st.markdown(_card("Peak Risk Day", peak_day["date"], f"{peak_day['score']:.0f}/100"), unsafe_allow_html=True)
    with mc3:
        st.markdown(_card("Fare Delta vs Today", impact_str), unsafe_allow_html=True)
    with mc4:
        st.markdown(
            _card(
                "Oil Trend (30d)",
                f"{oil_sign}{trend['price_change_pct']:.1f}%",
                f"${trend['current_price']:.2f}/bbl",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_globe, tab_forecast, tab_oil, tab_cascade, tab_ai = st.tabs([
        "  Globe  ", "  7-Day Forecast  ", "  Oil Market  ",
        "  Cascade Risk  ", "  AI Analysis  ",
    ])

    with tab_globe:
        components.html(d["map_html"], height=450)

    with tab_forecast:
        st.plotly_chart(d["f_chart"], use_container_width=True)
        st.markdown(
            '<p style="color:#6b7280;font-size:11px;margin:8px 0 4px 0;'
            'text-transform:uppercase;letter-spacing:0.06em">Day-by-day breakdown</p>',
            unsafe_allow_html=True,
        )
        for day in d["forecast"]:
            sc    = _score_color(day["score"])
            fare  = f"${day['projected_fare']:.0f}" if day["projected_fare"] else "—"
            arrow = "↑" if day["trend"] == "rising" else ("↓" if day["trend"] == "falling" else "→")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 12px;border-bottom:1px solid #1e1e2e">'
                f'<span style="color:#9ca3af;font-size:13px;min-width:100px">{day["date"]}</span>'
                f'<span style="color:{sc};font-weight:600;min-width:75px">{day["score"]:.1f}/100</span>'
                f'<span style="color:{sc};font-size:12px;min-width:90px">{day["label"]}</span>'
                f'<span style="color:#9ca3af;font-size:12px;min-width:65px">{fare}</span>'
                f'<span style="color:#6b7280;font-size:12px">'
                f'{arrow} {day["oil_change_pct"]:+.1f}% oil</span></div>',
                unsafe_allow_html=True,
            )

    with tab_oil:
        st.plotly_chart(d["o_chart"], use_container_width=True)
        oc1, oc2, oc3 = st.columns(3)
        p = trend
        with oc1:
            st.markdown(_card("Current Price", f"${p['current_price']:.2f}", "USD / bbl"), unsafe_allow_html=True)
        with oc2:
            sign = "+" if p["price_change_pct"] >= 0 else ""
            st.markdown(_card("30d Change", f"{sign}{p['price_change_pct']:.2f}%"), unsafe_allow_html=True)
        with oc3:
            direction = "Rising" if p["is_rising"] else "Falling"
            dir_color = "#E8593C" if p["is_rising"] else "#00C897"
            st.markdown(
                f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:10px;'
                f'padding:16px;text-align:center">'
                f'<p style="color:#6b7280;font-size:11px;margin:0;text-transform:uppercase;'
                f'letter-spacing:0.07em">Direction</p>'
                f'<p style="color:{dir_color};font-size:22px;font-weight:600;margin:4px 0 0 0">'
                f'{direction}</p></div>',
                unsafe_allow_html=True,
            )

    with tab_cascade:
        if rr["has_cascade"]:
            st.markdown(
                '<div style="padding:12px 16px;background:#E8593C1a;border-left:3px solid #E8593C;'
                'border-radius:6px;margin-bottom:16px">'
                '<p style="color:#E8593C;font-weight:600;margin:0;font-size:14px">'
                'Cascade Risk Detected</p>'
                '<p style="color:#9ca3af;font-size:13px;margin:4px 0 0 0">'
                'Disruptions on this route may drive secondary demand and fare increases '
                'at the following hub airports.</p></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(d["c_chart"], use_container_width=True)
            for c in rr["cascade_risks"]:
                with st.expander(
                    f"{c['affected_hub_name']} ({c['affected_hub']})  —  "
                    f"demand +{c['demand_increase_pct']}%  /  fares +{c['fare_impact_pct']}%"
                ):
                    st.markdown(f"**Trigger region:** {c['trigger_region']} (score {c['trigger_score']:.0f}/100)")
                    st.markdown(f"**Severity:** {c['severity']}")
                    st.markdown(f"**Why:** {c['reason']}")
                    st.markdown(f"*{c['confidence']}*")
        else:
            st.markdown(
                '<div style="padding:20px;background:#00C8971a;border:1px solid #00C897;'
                'border-radius:10px;text-align:center;margin-top:16px">'
                '<p style="color:#00C897;font-weight:600;font-size:16px;margin:0">'
                'No cascade risk detected</p>'
                '<p style="color:#9ca3af;font-size:13px;margin:8px 0 0 0">'
                'No airspace region on this route scores above the cascade threshold (50/100).</p>'
                '</div>',
                unsafe_allow_html=True,
            )

    with tab_ai:
        safe_response = html.escape(d["agent_response"])
        st.markdown(
            f'<div style="background:#080810;border:1px solid #1e1e2e;border-radius:10px;'
            f'padding:20px 24px;font-family:monospace;font-size:13px;color:#d1d5db;'
            f'line-height:1.75;white-space:pre-wrap;word-wrap:break-word">'
            f'{safe_response}</div>',
            unsafe_allow_html=True,
        )
