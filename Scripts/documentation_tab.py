from __future__ import annotations
import streamlit as st

# ── Section registry ──────────────────────────────────────────────────────────
DOC_SECTIONS: dict[str, str] = {
    "strokes_gained":   "Strokes Gained",
    "rolling_windows":  "Rolling Windows (L12 / L24 / L40)",
    "contender_model":  "Contender Model",
    "sg_profile":       "SG Profile Chart",
    "form_trend":       "Form Trend",
    "form_context":     "Form Context (L12 vs L60)",
    "field_percentile": "Field Percentile",
    "course_fit":       "Course Fit & DNA",
    "approach_skill":   "Approach & Proximity",
    "course_history":   "Course History",
}

# ── Section renderers ─────────────────────────────────────────────────────────

def _strokes_gained():
    st.markdown("## Strokes Gained")
    st.markdown(
        "Strokes Gained (SG) is the golf analytics standard for measuring performance. "
        "Every shot is compared to the tour average from the same situation — lie, distance, surface. "
        "A positive SG means the player did better than average on that shot; negative means worse. "
        "Totals accumulate across an entire round or tournament."
    )
    st.markdown("### Components")
    rows = [
        ("SG: Off the Tee (OTT)",      "Tee shots on par-4s and par-5s. Measures driving distance and accuracy vs tour average."),
        ("SG: Approach (APP)",          "Full shots into the green. The single biggest differentiator between elite and average players."),
        ("SG: Around the Green (ARG)",  "Chips, pitches, and bunker shots within ~30 yards of the green. Often called the 'scoring zone'."),
        ("SG: Putting (PUTT)",          "All putts. Measured relative to the average make probability from each distance."),
        ("SG: Tee-to-Green (T2G)",      "OTT + APP + ARG combined. The 'ball-striking' composite — excludes putting."),
        ("SG: Total",                   "Sum of all four components. The best single-number summary of performance."),
    ]
    for name, desc in rows:
        st.markdown(f"**{name}** — {desc}")

    st.markdown("### Interpreting values")
    st.markdown(
        "Values are per round relative to the field average (0.0). "
        "+1.0 SG Total per round means roughly one shot better than an average tour player per round. "
        "The best players in the world average around +2.5 to +3.0 per round over a season."
    )


def _rolling_windows():
    st.markdown("## Rolling Windows — L12 / L24 / L40 / L60")
    st.markdown(
        "Rolling windows count the most recent N rounds a player has played, regardless of how long "
        "ago they happened. A player who took three weeks off still has the same L36 — it doesn't "
        "go stale the way a fixed calendar window would."
    )
    rows = [
        ("L12", "~3 tournaments",  "Hot-streak signal. High noise but catches genuine recent form shifts. Good for spotting a player peaking right now."),
        ("L24", "~6 tournaments",  "Short-term baseline. Balances recency and sample size. Good for current-form context."),
        ("L36", "~9 tournaments",  "The model's primary window. Long enough to filter noise, short enough to stay current. Validated as optimal for the Contender Model."),
        ("L40", "~10 tournaments", "Near-full-season baseline. Used for the main field analysis table as a stable reference."),
        ("L60", "~15 tournaments", "Long-term baseline. Used as the 'career average' proxy in form context comparisons."),
    ]
    st.markdown("| Window | Approx. span | Best for |")
    st.markdown("|---|---|---|")
    for window, span, use in rows:
        st.markdown(f"| **{window}** | {span} | {use} |")

    st.markdown("### When windows disagree")
    st.markdown(
        "If L12 is much higher than L40, the player is heating up. "
        "If L12 is much lower, they're cooling off. "
        "The Form Context chart (Player Deep Dive) visualizes this gap directly."
    )


def _contender_model():
    st.markdown("## Contender Model")
    st.markdown(
        "**Formula:** `Contender Score = mean(sg_total, L36) − 0.25 × σ(sg_total, L36)`"
    )
    st.markdown(
        "The Contender Score answers: *which players produce strong SG reliably, not just occasionally?* "
        "The mean captures sustained output over the last 36 rounds (~9–12 months). "
        "The volatility penalty discounts boom-or-bust players — a player averaging +1.5 with wild swings "
        "is less predictable than one averaging +1.2 consistently."
    )

    st.markdown("### Why L36?")
    st.markdown(
        "Five formula variants were validated against every 2025/2026 PGA Tour tournament:"
    )
    st.markdown(
        "| Formula | Top-25% | Top-10% |\n"
        "|---|---|---|\n"
        "| **mean(L36) − 0.25σ L36 (production)** | **66.5%** | **35.9%** |\n"
        "| 0.6×mean(L60) + 0.4×expDecay(L12) − 0.3σ L36 | 63.6% | 34.6% |\n"
        "| mean(L60) − 0.3σ L60 | 63.2% | 34.6% |\n"
        "| exp_decay(L36) − 0.3σ L36 | 63.1% | 33.8% |\n"
        "| mean(L60) − 0.3σ L36 | 62.9% | 34.6% |"
    )

    st.markdown("### Score interpretation")
    rows = [
        ("≥ 1.0",       "Elite",      "Consistently among the best in the field. High floor, top-25 very likely."),
        ("0.5 – 1.0",   "Contender",  "Strong player, competitive most weeks."),
        ("0.0 – 0.5",   "Fringe",     "Capable but inconsistent. Needs a peak week."),
        ("< 0.0",       "Below",      "Negative expected value relative to field average."),
    ]
    for score, label, desc in rows:
        st.markdown(f"**{score} ({label})** — {desc}")

    st.markdown("### Why not a more complex model?")
    st.markdown(
        "After testing 288 configurations — linear regression, random forest, multi-window ensembles — "
        "every complex model underperformed the simple L36 mean/volatility formula out-of-sample. "
        "Complex models learn noise as signal and overfit to training data. Simplicity wins in golf."
    )

    st.markdown("### Scatter plot")
    st.markdown(
        "Each dot is a player. Further right = higher mean SG (better output). "
        "Lower = lower std dev (more consistent). The dotted lines are iso-score contours. "
        "Elite picks live in the bottom-right: high mean, low variance."
    )


def _sg_profile():
    st.markdown("## SG Profile Chart (Ridge / Density)")
    st.markdown(
        "Each row shows the **full distribution of round-by-round performance** for one skill — "
        "not just an average, but every round played across the last 60. "
        "Think of it as a histogram smoothed into a curve."
    )
    st.markdown("### Reading the chart")
    st.markdown(
        "- **Tall narrow peak** → consistent player. Their rounds cluster tightly around the mean.\n"
        "- **Wide flat curve** → volatile player. Big swings round-to-round.\n"
        "- **Dotted vertical line + labeled dot** → the player's mean for that skill.\n"
        "- **Grey shaded area** → field distribution for context.\n"
        "- **Green band** → the SG range this course historically rewards. "
        "A player whose peak sits inside the band fits this track."
    )
    st.markdown("### Z-score axis")
    st.markdown(
        "All skills are converted to tour z-scores so rows are directly comparable on the same axis. "
        "0 = tour average. +1 = one standard deviation above average. "
        "This lets you compare a driver (high raw numbers) directly against a putter (low raw numbers)."
    )


def _form_trend():
    st.markdown("## Form Trend Chart")
    st.markdown(
        "SG Total per round (grey dots) with a smoothed moving average (orange line) and "
        "±1 standard deviation band over the last 40 rounds."
    )
    st.markdown(
        "- **Rising line** → improving form over the recent sample.\n"
        "- **Wide band** → inconsistent round-to-round. Hard to predict which version shows up.\n"
        "- **Narrow band** → consistent. The average line is a reliable predictor.\n"
        "- **Smoothing slider** → controls how many rounds the moving average spans. "
        "Lower = more reactive to recent rounds, higher = smoother long-term trend."
    )


def _form_context():
    st.markdown("## Form Context — L12 vs L60 Baseline")
    st.markdown(
        "Shows how this player's last 12 rounds compare to their own long-term baseline "
        "(last 60 rounds) in each SG component."
    )
    st.markdown(
        "- **Green bars** → the recent stretch is above the player's own 60-round baseline.\n"
        "- **Red bars** → below their own baseline.\n"
        "- **Number shown** → z-score: how many standard deviations above/below their own average."
    )
    st.markdown("### Noise vs signal")
    st.markdown(
        "Not all deviations mean the same thing:\n\n"
        "- **Putting and Around Green** → high-noise categories. Deviations tend to self-correct quickly. "
        "A player putting great over 12 rounds will usually regress toward their mean.\n"
        "- **Off-Tee and Approach** → stickier. A meaningful deviation here more often reflects "
        "a real change in ball-striking than statistical noise."
    )


def _field_percentile():
    st.markdown("## Field Percentile Bars")
    st.markdown(
        "Each bar shows where a player ranks among everyone in this week's field. "
        "0 = last place in the field, 100 = best in the field."
    )
    st.markdown(
        "- **Based on L60** → last 60 rounds of data, so it reflects sustained performance, not a recent hot streak.\n"
        "- **† (dagger) markers** → lower-is-better stats (bogeys, round score). "
        "These are inverted so right is always good.\n"
        "- **H2H version** → shows both players side by side. Colored dot indicates who wins that stat. "
        "The gap between dots shows how meaningful the edge is relative to the entire field."
    )


def _course_fit():
    st.markdown("## Course Fit & Course DNA")

    st.markdown("### Course Fit Score")
    st.markdown(
        "Course Fit measures how well a player's skill profile aligns with what a specific course "
        "historically rewards."
    )
    st.markdown(
        "**Formula:** `Course Fit = Σ (course beta × player z-score)` for each skill\n\n"
        "- **Course beta** — regression coefficient measuring how strongly each skill predicted "
        "actual scoring at that venue across thousands of rounds. Higher beta = more important skill.\n"
        "- **Player z-score** — how many standard deviations a player's stat sits above or below tour average.\n"
        "- **Positive fit score** → player's strengths align with what this course rewards.\n"
        "- **Bar chart** shows the per-skill contribution, so you can see *why* a player fits or doesn't."
    )

    st.markdown("### Course DNA")
    st.markdown(
        "The Course DNA dumbbell chart shows which skills this course rewards *relative to other courses* "
        "— not just raw weight."
    )
    st.markdown(
        "- **Grey hollow dot** → tour-average course weight for that skill.\n"
        "- **Colored filled dot** → this course's weight.\n"
        "- **+X.X% deviation** → how much this course over- or under-indexes on that skill vs. tour average.\n\n"
        "Distance tends to have the highest raw weight everywhere (longer hitters have shorter approaches), "
        "so raw ranking is misleading. Deviation-based ranking shows what actually *distinguishes* this venue."
    )


def _approach_skill():
    st.markdown("## Approach Skill & Proximity")

    st.markdown("### Distance buckets")
    st.markdown(
        "Approach shots are split into four distance ranges from fairway and two from rough:\n\n"
        "- **50–100 yds** → wedge game close to the green\n"
        "- **100–150 yds** → mid-short irons\n"
        "- **150–200 yds** → mid irons\n"
        "- **200+ yds** → long irons / hybrids\n"
        "- **Fairway vs Rough** → tracked separately because lie dramatically changes expected proximity."
    )

    st.markdown("### Metrics")
    st.markdown(
        "- **SG/Shot** → strokes gained per approach shot from that bucket vs tour average.\n"
        "- **Proximity** → average distance to hole after the shot, in feet. Lower is better.\n"
        "- **GIR %** → greens in regulation percentage from that bucket.\n"
        "- **Good shot %** → percentage of shots classified as above-average by DataGolf's shot model.\n"
        "- **Poor avoid %** → percentage of shots avoiding the 'poor' outcome category."
    )

    st.markdown("### Proximity chart")
    st.markdown(
        "Top-down view of a green. The center dot is the hole. "
        "**Smaller ring = closer to the hole = better.** "
        "Left green = from fairway. Right green = from rough. "
        "White dashed ring = tour average proximity from that bucket."
    )


def _course_history():
    st.markdown("## Course History")
    st.markdown(
        "Past results at a specific venue going back to 2017. "
        "Shows how each player has historically performed at this exact course — "
        "not just finishes, but strokes gained averages per round."
    )
    st.markdown(
        "- **SG columns** → per-round averages across all rounds the player played at this venue.\n"
        "- **Green/red coloring** → above or below the field average for that event.\n"
        "- **Win count badge** → total wins at this venue in the dataset.\n"
        "- **Years shown** → each column is one calendar year. Blank = player didn't play."
    )
    st.markdown(
        "**Note:** Pre-2022 data has no `round_date` timestamps so some older rows show without date context. "
        "The SG values themselves are still accurate."
    )


# ── Section map ───────────────────────────────────────────────────────────────
_RENDERERS = {
    "strokes_gained":   _strokes_gained,
    "rolling_windows":  _rolling_windows,
    "contender_model":  _contender_model,
    "sg_profile":       _sg_profile,
    "form_trend":       _form_trend,
    "form_context":     _form_context,
    "field_percentile": _field_percentile,
    "course_fit":       _course_fit,
    "approach_skill":   _approach_skill,
    "course_history":   _course_history,
}


# ── Main entry point ──────────────────────────────────────────────────────────

def render_documentation_tab():
    section_key = st.session_state.pop("doc_section", "strokes_gained")
    if section_key not in _RENDERERS:
        section_key = "strokes_gained"

    keys   = list(DOC_SECTIONS.keys())
    labels = list(DOC_SECTIONS.values())
    idx    = keys.index(section_key)

    selected_label = st.selectbox(
        "Section",
        labels,
        index=idx,
        key="doc_section_select",
        label_visibility="collapsed",
    )
    selected_key = keys[labels.index(selected_label)]

    st.divider()
    _RENDERERS[selected_key]()
