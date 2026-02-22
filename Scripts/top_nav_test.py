import streamlit as st

st.set_page_config(page_title="Top Nav Test", layout="wide")

PAGES = ["Weekly", "Player Profile", "Deep Dive", "Course History", "League"]

def go(page: str) -> None:
    st.session_state["page"] = page

if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]

st.markdown(
    """
    <style>
      .top-nav-wrap { padding: 6px 0 10px 0; }
      div[data-testid="column"] > div:has(button) button {
        width: 100%;
        height: 70px;
        font-size: 18px;
        font-weight: 900;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.12);
      }
      .active-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        font-weight: 800;
        margin-bottom: 10px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Top nav row ---
st.markdown('<div class="top-nav-wrap"></div>', unsafe_allow_html=True)
cols = st.columns(len(PAGES), gap="small")

for i, p in enumerate(PAGES):
    is_active = (st.session_state["page"] == p)
    label = f"✅ {p}" if is_active else p
    if cols[i].button(label, key=f"nav_{p}"):
        go(p)
        st.rerun()

st.divider()

# --- Page header / content ---
active = st.session_state["page"]
st.markdown(f'<div class="active-pill">ACTIVE: {active}</div>', unsafe_allow_html=True)

if active == "Weekly":
    st.header("Weekly")
    st.write("Put your weekly overview charts/tables here.")
    with st.expander("Example controls", expanded=True):
        st.slider("Rolling window (weeks)", 4, 52, 12)
        st.selectbox("Event", ["Genesis Invitational", "WM Phoenix Open", "The Players"])
    st.line_chart({"example_metric": [1, 4, 2, 6, 3, 7]})

elif active == "Player Profile":
    st.header("Player Profile")
    st.write("Player-centric breakdown.")
    c1, c2, c3 = st.columns([2, 1, 1], gap="large")
    with c1:
        st.subheader("Selected player")
        st.selectbox("Player", ["Xander Schauffele", "Scottie Scheffler", "Rory McIlroy"])
        st.write("Bio / KPIs / form indicators go here.")
    with c2:
        st.metric("SG Total (L12)", 1.42)
        st.metric("SG APP (L12)", 0.61)
    with c3:
        st.metric("SG OTT (L12)", 0.32)
        st.metric("SG PUTT (L12)", 0.18)

elif active == "Deep Dive":
    st.header("Deep Dive")
    st.write("This is where you put the richer, multi-section analysis layout.")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.subheader("Chart A")
        st.bar_chart({"A": [3, 1, 4, 2]})
    with c2:
        st.subheader("Chart B")
        st.area_chart({"B": [1, 2, 1, 3]})

elif active == "Course History":
    st.header("Course History")
    st.write("Course history tables and year-by-year performance.")
    st.dataframe(
        {
            "Year": [2021, 2022, 2023, 2024, 2025],
            "Finish": ["T12", "T5", "CUT", "T2", "T18"],
            "SG Total": [1.2, 3.8, -0.4, 5.1, 0.7],
        },
        use_container_width=True,
        hide_index=True,
    )

elif active == "League":
    st.header("League")
    st.write("Standings, picks log, ownership, etc.")
    st.dataframe(
        {
            "Rank": [1, 2, 3, 4, 5],
            "Player": ["A", "B", "C", "D", "E"],
            "Earnings": [6_440_000, 6_100_000, 5_850_000, 5_500_000, 5_200_000],
        },
        use_container_width=True,
        hide_index=True,
    )