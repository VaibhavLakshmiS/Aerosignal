"""
app.py — Streamlit entry point for the AeroSignal dashboard.

Premium dark-themed single-page app for geopolitical flight risk intelligence.
Orchestrates the agent pipeline (rag/chain.py), risk scoring (rag/signals.py),
and visualisations (viz/map.py, viz/charts.py) into a cohesive UI for
travellers and analysts.
"""

import html
import re
import sys
import time
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
from data.fetch_flights import fetch_flights, AIRPORT_FALLBACKS
from data.fetch_prices import fetch_oil_prices, get_oil_trend
from rag.chain import run_agent
from rag.signals import (
    AIRPORT_COORDS, get_waypoints, score_route,
    forecast_route, get_best_booking_day,
)
from viz.charts import cascade_chart, forecast_chart, oil_chart
from viz.map import build_map_html, get_risk_color


@st.cache_data(ttl=60)
def get_cached_events(region: str) -> list:
    """Fetch events with a 60s Streamlit cache to limit GDELT hammering."""
    return fetch_events(region, days=14)


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


def _render_agent_response(response: str, score: float) -> None:
    """Render agent response with highlighted numbers and a styled recommendation box."""
    sc = _score_color(score)

    def _highlight(text: str) -> str:
        text = html.escape(text)
        text = re.sub(
            r'\$(\d+(?:\.\d+)?)',
            r'<span style="color:#F5A623;font-weight:600">$\1</span>', text,
        )
        text = re.sub(
            r'(\d+(?:\.\d+)?)\s*%',
            r'<span style="color:#F5A623;font-weight:600">\1%</span>', text,
        )
        text = re.sub(
            r'(\d+(?:\.\d+)?)\s*/\s*100',
            rf'<span style="color:{sc};font-weight:600">\1/100</span>', text,
        )
        return text

    _REC_KEYWORDS = ("recommend", "book", "avoid", "consider", "verdict",
                     "conclusion", "summary", "strong")

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', response) if p.strip()]
    if not paragraphs:
        paragraphs = [response.strip()]

    for para in paragraphs:
        first_word = para.split()[0].lower().rstrip(':') if para.split() else ""
        is_rec = first_word in _REC_KEYWORDS or "recommendation:" in para.lower()

        if is_rec:
            st.markdown(
                f'<div style="margin:12px 0;padding:14px 18px;background:{sc}1a;'
                f'border-left:3px solid {sc};border-radius:6px">'
                f'<p style="color:#6b7280;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:0.08em;margin:0 0 6px 0">Recommendation</p>'
                f'<p style="color:{sc};margin:0;font-size:14px;font-weight:500;'
                f'line-height:1.65">{_highlight(para)}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#080810;border:1px solid #1e1e2e;border-radius:8px;'
                f'padding:14px 18px;margin:6px 0;font-family:monospace;font-size:13px;'
                f'color:#d1d5db;line-height:1.75;white-space:pre-wrap;word-wrap:break-word">'
                f'{_highlight(para)}</div>',
                unsafe_allow_html=True,
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
        st.metric("Risk Score", f"{_rr['score']:.0f}/100")
        st.metric("Oil Price",  f"${_tr['current_price']:.2f}/bbl")
        st.metric("30d Trend",  f"{_sign}{_tr['price_change_pct']:.1f}%")


# ── Rate limiting ─────────────────────────────────────────────────────────────

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = 0

# ── Analysis trigger ──────────────────────────────────────────────────────────

if clicked:
    time_since_last = time.time() - st.session_state.last_analysis_time
    if time_since_last < 30:
        st.warning(
            f"Please wait {30 - int(time_since_last)} seconds "
            "before analyzing another route."
        )
        st.stop()
    if st.session_state.analysis_count >= 10:
        st.error("Session limit reached. Please refresh the page.")
        st.stop()

    st.session_state.analysis_count += 1
    st.session_state.last_analysis_time = time.time()

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
        with st.spinner("Loading route data..."):
            try:
                prices = fetch_oil_prices()
                trend  = get_oil_trend(prices)
            except Exception:
                prices = []
                trend  = {
                    "current_price": 0.0, "price_change_pct": 0.0,
                    "is_rising": False, "prices": [],
                }

            flights      = []
            fare_note    = None
            current_fare = None
            try:
                flights = fetch_flights(origin, destination, travel_date.strftime("%Y-%m-%d"))
                if not flights and origin in AIRPORT_FALLBACKS:
                    fallback = AIRPORT_FALLBACKS[origin]
                    flights  = fetch_flights(fallback, destination, travel_date.strftime("%Y-%m-%d"))
                    if flights:
                        fare_note = (
                            f"No direct fares found for {origin}. "
                            f"Showing fares from {fallback} as reference."
                        )
                current_fare = flights[0]["price_usd"] if flights else None
            except Exception:
                pass

            try:
                risk_result = score_route(origin, destination, [], trend, prices)
            except Exception as e:
                st.error(f"Risk scoring failed: {e}")
                st.stop()

            try:
                forecast = forecast_route(origin, destination, risk_result, prices, current_fare)
            except Exception:
                forecast = []

            try:
                map_html = build_map_html(origin, destination, risk_result)
            except Exception:
                map_html = "<p style='color:#555;padding:20px'>Map unavailable</p>"

            f_chart = c_chart = o_chart = None
            try:
                f_chart = forecast_chart(forecast) if forecast else None
            except Exception:
                pass
            try:
                o_chart = oil_chart(prices) if prices else None
            except Exception:
                pass
            try:
                c_chart = cascade_chart(risk_result["cascade_risks"]) if risk_result.get("has_cascade") else None
            except Exception:
                pass

            st.session_state["data"] = {
                "origin":         origin,
                "destination":    destination,
                "risk_result":    risk_result,
                "forecast":       forecast,
                "trend":          trend,
                "prices":         prices,
                "agent_response": None,
                "events":         None,
                "map_html":       map_html,
                "f_chart":        f_chart,
                "o_chart":        o_chart,
                "c_chart":        c_chart,
                "current_fare":   current_fare,
                "fare_note":      fare_note,
            }
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

if "data" not in st.session_state:

    # ── Hero (pre-analysis) ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 48px 20px">
        <h1 style="font-size:48px;font-weight:700;color:#ffffff;
                   margin:0;line-height:1.15">
            Is this route<br>risky right now?
        </h1>
        <p style="color:#6b7280;font-size:18px;margin:24px auto 0 auto;
                  max-width:560px;line-height:1.65">
            AeroSignal analyzes geopolitical events, oil markets and airspace
            corridors to assess flight route risk — giving you intelligence
            before you book.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    feature_cards = [
        (
            "Route Risk Score",
            "Scores your route 0-100 based on live news, oil prices and "
            "airspace data across the full flight corridor.",
        ),
        (
            "Cascade Detection",
            "Detects second-order hub airport impacts when primary routes "
            "are disrupted by geopolitical events.",
        ),
        (
            "7-Day Outlook",
            "Projects how risk and estimated fares may change over the next "
            "week based on oil momentum and news cycle.",
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

    st.markdown("""
<div style='font-size:11px;color:#444;padding:8px 4px;line-height:1.6'>
    Risk scores are model-derived estimates based on news volume, oil price momentum,
    and historical event patterns. Not financial or travel advice. Always check official
    airline and government travel advisories before booking.
</div>
""", unsafe_allow_html=True)

    google_flights_url = (
        f"https://www.google.com/travel/flights"
        f"?q=flights+from+{origin}+to+{destination}"
    )
    st.link_button("Search flights on Google Flights →", google_flights_url)

    if d.get("fare_note"):
        st.warning(d["fare_note"])
    if not d["current_fare"]:
        st.info(
            "Live fare data unavailable for this route. "
            "Risk analysis is still accurate — fare projections require Serpapi data."
        )

    # Four metric columns
    mc1, mc2, mc3, mc4 = st.columns(4)
    best_fare_str = f"${best_day['projected_fare']:.0f}" if best_day["projected_fare"] else "—"

    if d["current_fare"] and best_day["projected_fare"]:
        diff       = best_day["projected_fare"] - d["current_fare"]
        impact_str = f"+${diff:.0f}" if diff >= 0 else f"-${abs(diff):.0f}"
    else:
        impact_str = "Unavailable"

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
    tab_globe, tab_forecast, tab_oil, tab_cascade, tab_ai, tab_backtest = st.tabs([
        "  Globe  ", "  7-Day Forecast  ", "  Oil Market  ",
        "  Cascade Risk  ", "  AI Analysis  ", "  Backtest  ",
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
        if d["agent_response"] is None:
            # Fetch events for top-2 waypoints only (fast, cached)
            if d["events"] is None:
                waypoints = get_waypoints(origin, destination)
                ai_events: list = []
                for region in waypoints[:2]:
                    try:
                        ai_events.extend(get_cached_events(region))
                        time.sleep(1)
                    except Exception:
                        pass
                d["events"] = ai_events
                st.session_state["data"] = d

            agent_response = None
            with st.spinner("Agent analyzing route..."):
                try:
                    query = (
                        f"Analyse the geopolitical risk for flying {origin} to {destination}. "
                        "Include 7-day forecast and cascade risks."
                    )
                    agent_response = run_agent(query)
                except Exception:
                    st.warning(
                        "AI Analysis temporarily unavailable. "
                        "All other tabs are fully functional."
                    )

            d["agent_response"] = agent_response or ""
            st.session_state["data"] = d
            if agent_response:
                st.rerun()

        elif d["agent_response"]:
            _render_agent_response(d["agent_response"], score)

        events = d.get("events") or []
        if events:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#6b7280;font-size:11px;margin:0 0 8px 0;'
                'text-transform:uppercase;letter-spacing:0.07em">News Feed</p>',
                unsafe_allow_html=True,
            )
            for event in sorted(events, key=lambda e: e.get("relevance_score", 0), reverse=True):
                title    = html.escape(event.get("title", ""))
                url      = event.get("url", "")
                date     = str(event.get("date", ""))
                ev_score = event.get("relevance_score", 0)
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc or url
                except Exception:
                    domain = url

                dot_color = (
                    "#E8593C" if ev_score >= 3
                    else "#F5A623" if ev_score >= 2
                    else "#555555"
                )

                link_html = (
                    f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                    f'style="color:#ffffff;text-decoration:none;font-size:14px;'
                    f'font-weight:500;line-height:1.4">{title}</a>'
                    if url else
                    f'<span style="color:#ffffff;font-size:14px">{title}</span>'
                )

                st.markdown(
                    f'<div style="padding:10px 0;border-bottom:1px solid #1e1e2e">'
                    f'<span style="color:{dot_color};font-size:9px;'
                    f'vertical-align:middle;margin-right:8px">&#9679;</span>'
                    f'{link_html}<br>'
                    f'<span style="color:#6b7280;font-size:12px;margin-left:17px">'
                    f'{domain}&nbsp;&middot;&nbsp;{date}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tab_backtest:
        st.markdown("""
<div style='font-size:13px;color:#888;margin-bottom:16px;line-height:1.7'>
    AeroSignal's risk model is validated against 3 known historical geopolitical
    events. For each event, we test whether the model correctly flags elevated risk
    using only data available BEFORE the event peaked — simulating real-world
    early warning capability.
</div>
""", unsafe_allow_html=True)

        with st.spinner("Running backtest..."):
            from rag.signals import validate_model
            backtest_results = validate_model()

        correct  = sum(1 for r in backtest_results if r["correctly_flagged"])
        total    = len(backtest_results)
        accuracy = round(correct / total * 100) if total else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Events correctly flagged", f"{correct}/{total}")
        with col2:
            st.metric("Directional accuracy", f"{accuracy}%")
        with col3:
            st.metric("Events tested", total)

        st.divider()

        for r in backtest_results:
            flagged_color = "#00C897" if r["correctly_flagged"] else "#E8593C"
            flagged_text  = "Correctly flagged" if r["correctly_flagged"] else "Missed"
            oil_text      = (
                f"{r['actual_oil_change_pct']:+.1f}%"
                if r["actual_oil_change_pct"] is not None else "N/A"
            )
            st.markdown(f"""
<div style='background:#0f0f1a;border:1px solid #1e1e2e;border-radius:8px;
    padding:14px 18px;margin-bottom:10px'>
    <div style='display:flex;justify-content:space-between;align-items:center;
        flex-wrap:wrap;gap:8px'>
        <div>
            <div style='font-size:14px;font-weight:500;color:#fff'>{r["event"]}</div>
            <div style='font-size:11px;color:#555;margin-top:2px'>
                {r["date"]} &middot; {r["route"]} &middot; {r["region"]}
            </div>
        </div>
        <div style='text-align:right'>
            <div style='font-size:20px;font-weight:500;color:#F5A623'>
                {r["pre_event_score"]}/100
            </div>
            <div style='font-size:11px;color:{flagged_color}'>{flagged_text}</div>
        </div>
    </div>
    <div style='margin-top:10px;font-size:12px;color:#666'>
        Oil price {oil_text} in 14 days after event &middot;
        Pre-event oil: ${r["oil_at_event"]}/bbl
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style='font-size:11px;color:#444;margin-top:12px;line-height:1.6'>
    Backtest methodology: Pre-event scores calculated using oil price data from
    30 days before each event with 3 simulated early warning news signals.
    Scores &ge;30 considered "correctly flagged." Actual oil changes measured
    14 days post-event. This is directional validation only — not a guarantee
    of future accuracy.
</div>
""", unsafe_allow_html=True)
