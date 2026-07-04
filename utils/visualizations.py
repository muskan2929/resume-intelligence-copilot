"""
visualizations.py

Builds all Plotly charts used in the dashboard:
1. Gauge chart      -> overall match score
2. Radar chart       -> resume strength across 7 dimensions
3. Heatmap           -> missing vs matched skills
4. Progress bars     -> simple bar chart for section scores

All functions return Plotly Figure objects, ready to be passed
straight into gr.Plot() in the Gradio UI (Phase 6).
"""

from typing import Dict, List

import plotly.graph_objects as go

# A consistent dark, professional color palette used across all charts
COLORS = {
    "background": "#1e1e2f",
    "text": "#f5f5f5",
    "primary": "#6c5ce7",
    "success": "#00b894",
    "warning": "#fdcb6e",
    "danger": "#e17055",
    "grid": "#3a3a4d",
}


# ------------------------------------------------------------------
# 1. GAUGE CHART — Overall Match Score
# ------------------------------------------------------------------

def build_gauge_chart(overall_score: float) -> go.Figure:
    """
    Builds a gauge chart showing the overall resume-job match score.

    Args:
        overall_score: Score from 0-100.

    Returns:
        Plotly Figure object.
    """
    # Pick a color band based on score range
    if overall_score >= 75:
        bar_color = COLORS["success"]
    elif overall_score >= 50:
        bar_color = COLORS["warning"]
    else:
        bar_color = COLORS["danger"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        number={"suffix": "%", "font": {"color": COLORS["text"], "size": 40}},
        title={"text": "Resume Match Score", "font": {"color": COLORS["text"], "size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text"]},
            "bar": {"color": bar_color},
            "bgcolor": COLORS["background"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#3a2f3f"},
                {"range": [50, 75], "color": "#3a3a2f"},
                {"range": [75, 100], "color": "#2f3a35"},
            ],
        },
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font={"color": COLORS["text"]},
        margin=dict(l=30, r=30, t=60, b=20),
        height=300,
    )
    return fig


# ------------------------------------------------------------------
# 2. RADAR CHART — Resume Strength Across 7 Dimensions
# ------------------------------------------------------------------

def build_radar_chart(strength_scores: Dict[str, float]) -> go.Figure:
    """
    Builds a radar chart showing resume strength across 7 dimensions:
    Technical Skills, Projects, Experience, Leadership, Communication,
    Achievements, Problem Solving.

    Args:
        strength_scores: Dict mapping dimension name -> score (0-100).
            Example: {"Technical Skills": 80, "Projects": 60, ...}

    Returns:
        Plotly Figure object.
    """
    categories = list(strength_scores.keys())
    values = list(strength_scores.values())

    # Close the radar loop by repeating the first point at the end
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(108, 92, 231, 0.35)",
        line=dict(color=COLORS["primary"], width=2),
        name="Resume Strength",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["background"],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color=COLORS["text"],
                gridcolor=COLORS["grid"],
            ),
            angularaxis=dict(color=COLORS["text"], gridcolor=COLORS["grid"]),
        ),
        showlegend=False,
        paper_bgcolor=COLORS["background"],
        font={"color": COLORS["text"]},
        margin=dict(l=40, r=40, t=40, b=40),
        height=420,
        title={"text": "Resume Strength Radar", "font": {"color": COLORS["text"], "size": 16}},
    )
    return fig


def compute_strength_scores(
    section_scores: Dict[str, float], risk_report: Dict, resume_skills: List[str]
) -> Dict[str, float]:
    """
    Derives the 7 radar dimensions from data we already computed in
    Phases 3-4. This keeps the radar chart explainable and consistent
    with the rest of the analysis rather than inventing new numbers.

    Args:
        section_scores: Output of embedding_model section similarity scores (0-1).
        risk_report: Output of analyzer.run_risk_detector().
        resume_skills: List of extracted resume skills.

    Returns:
        Dict of 7 dimension scores (0-100), ready for build_radar_chart().
    """
    def pct(x: float) -> float:
        return round(x * 100, 1)

    # Technical Skills: based on skills section similarity + skill count
    technical = pct(section_scores.get("skills", 0.0))

    # Projects & Experience: directly from section similarity
    projects = pct(section_scores.get("projects", 0.0))
    experience = pct(section_scores.get("experience", 0.0))

    # Leadership & Communication: derived from leadership/summary sections
    leadership = pct(section_scores.get("leadership", 0.0))
    communication = pct(section_scores.get("summary", 0.0))

    # Achievements: from achievements section, penalized if risk detector
    # found "no measurable achievements"
    achievements = pct(section_scores.get("achievements", 0.0))
    if any("measurable achievements" in issue for issue in risk_report["issues"]):
        achievements = max(0, achievements - 25)

    # Problem Solving: approximate using average of projects + experience
    # (no dedicated section usually exists for this)
    problem_solving = round((projects + experience) / 2, 1)

    return {
        "Technical Skills": technical,
        "Projects": projects,
        "Experience": experience,
        "Leadership": leadership,
        "Communication": communication,
        "Achievements": achievements,
        "Problem Solving": problem_solving,
    }


# ------------------------------------------------------------------
# 3. HEATMAP — Missing vs Matched Skills
# ------------------------------------------------------------------

def build_skills_heatmap(heatmap_data: List[Dict]) -> go.Figure:
    """
    Builds an interactive heatmap showing every required skill,
    whether it's matched or missing, and its importance level.

    Args:
        heatmap_data: Output of analyzer.build_missing_skills_heatmap().

    Returns:
        Plotly Figure object.
    """
    if not heatmap_data:
        # Return an empty placeholder figure so the UI doesn't crash
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["background"],
            font={"color": COLORS["text"]},
            title="No skills data available",
            height=300,
        )
        return fig

    importance_order = {"Critical": 0, "Important": 1, "Optional": 2}
    sorted_data = sorted(heatmap_data, key=lambda x: (importance_order[x["importance"]], -x["match_score"]))

    skills = [d["skill"] for d in sorted_data]
    importance = [d["importance"] for d in sorted_data]
    match_scores = [d["match_score"] for d in sorted_data]
    statuses = [d["status"] for d in sorted_data]

    # Build a single-column heatmap where color = match_score,
    # and hover text shows status + importance for explainability.
    hover_text = [
        f"Skill: {s}<br>Status: {st}<br>Importance: {imp}<br>Match Score: {sc:.2f}"
        for s, st, imp, sc in zip(skills, statuses, importance, match_scores)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=[[score] for score in match_scores],
        y=skills,
        x=["Match Strength"],
        colorscale=[
            [0.0, COLORS["danger"]],
            [0.5, COLORS["warning"]],
            [1.0, COLORS["success"]],
        ],
        zmin=0, zmax=1,
        text=[[t] for t in hover_text],
        hoverinfo="text",
        colorbar=dict(title="Match", tickfont={"color": COLORS["text"]}),
    ))

    fig.update_layout(
        title={"text": "Missing Skills Heatmap", "font": {"color": COLORS["text"], "size": 16}},
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font={"color": COLORS["text"]},
        yaxis=dict(autorange="reversed"),  # Critical skills on top
        height=max(300, len(skills) * 35),
        margin=dict(l=120, r=40, t=60, b=40),
    )
    return fig


# ------------------------------------------------------------------
# 4. SECTION SCORE PROGRESS BARS
# ------------------------------------------------------------------

def build_section_scores_bar(section_scores: Dict[str, float]) -> go.Figure:
    """
    Builds a horizontal bar chart showing similarity score per resume
    section — acts as a simple "progress bar" style visualization.

    Args:
        section_scores: Dict of section_name -> similarity score (0-1).

    Returns:
        Plotly Figure object.
    """
    sections = list(section_scores.keys())
    scores_pct = [round(v * 100, 1) for v in section_scores.values()]

    colors = [
        COLORS["success"] if s >= 70 else COLORS["warning"] if s >= 45 else COLORS["danger"]
        for s in scores_pct
    ]

    fig = go.Figure(go.Bar(
        x=scores_pct,
        y=sections,
        orientation="h",
        marker_color=colors,
        text=[f"{s}%" for s in scores_pct],
        textposition="outside",
    ))

    fig.update_layout(
        title={"text": "Section-Level Match Scores", "font": {"color": COLORS["text"], "size": 16}},
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font={"color": COLORS["text"]},
        xaxis=dict(range=[0, 100], gridcolor=COLORS["grid"]),
        height=max(250, len(sections) * 45),
        margin=dict(l=100, r=40, t=50, b=40),
    )
    return fig
