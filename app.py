"""
app.py

Main Gradio application for Resume Intelligence Copilot.

Ties together:
- resume_parser.py   (extract + clean + section-split resume)
- embedding_model.py (semantic similarity scoring)
- analyzer.py         (heatmap, transferability, risk, summary, roadmap, questions)
- visualizations.py   (gauge, radar, heatmap, bar charts)

Layout: dark theme, tabbed dashboard.
Tabs: Overview | Analysis | Skills | Risks | Interview Questions | Roadmap
"""

import tempfile
from typing import Tuple

import gradio as gr
import subprocess
import importlib.util

# Ensure spaCy's English model is available (HF Spaces doesn't always
# install it from requirements.txt reliably, so we check at runtime).
if importlib.util.find_spec("en_core_web_sm") is None:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)


from utils.resume_parser import parse_resume
from model.analyzer import analyze_resume
from utils.visualizations import (
    build_gauge_chart,
    build_radar_chart,
    compute_strength_scores,
    build_skills_heatmap,
    build_section_scores_bar,
)


# ------------------------------------------------------------------
# CORE PIPELINE FUNCTION — called when the user clicks "Analyze"
# ------------------------------------------------------------------

def run_pipeline(resume_file, job_description: str):
    """
    Full end-to-end pipeline: takes an uploaded resume file + JD text,
    runs parsing -> analysis -> visualization, and returns everything
    the UI needs to populate all tabs.

    Args:
        resume_file: Gradio File object (uploaded resume, .pdf or .txt).
        job_description: Job description text entered by the user.

    Returns:
        A tuple of all outputs needed to fill every UI component.
    """
    if resume_file is None:
        raise gr.Error("Please upload a resume file (.pdf or .txt).")
    if not job_description or not job_description.strip():
        raise gr.Error("Please paste a job description.")

    # --- Step 1: Parse resume ---
    parsed = parse_resume(resume_file.name)

    # --- Step 2: Run full analysis ---
    result = analyze_resume(parsed, job_description)

    match_result = result["match_result"]
    heatmap_data = result["heatmap_data"]
    transferability_report = result["transferability_report"]
    risk_report = result["risk_report"]
    recruiter_summary = result["recruiter_summary"]
    roadmap = result["roadmap"]
    interview_questions = result["interview_questions"]

    # --- Step 3: Build charts ---
    gauge_fig = build_gauge_chart(match_result["overall_score"])

    strength_scores = compute_strength_scores(
        match_result["section_scores"], risk_report, parsed["skills_list"]
    )
    radar_fig = build_radar_chart(strength_scores)

    heatmap_fig = build_skills_heatmap(heatmap_data)
    bar_fig = build_section_scores_bar(match_result["section_scores"])

    # --- Step 4: Format text/table outputs for the UI ---

    confidence_text = f"**Confidence:** {match_result['confidence']}% (based on resume completeness)"

    # Missing skills table (list of lists for gr.Dataframe)
    skills_table = [
        [d["skill"], d["status"], d["importance"], round(d["match_score"], 2)]
        for d in heatmap_data
    ]

    # Transferability table
    transfer_table = [
        [t["skill"], t["transferability"], t["learning_difficulty"],
         ", ".join(t["related_skills_found"]) or "-", t["reason"]]
        for t in transferability_report
    ]

    # Risk issues as a bullet list string
    if risk_report["issues"]:
        risk_text = f"### Risk Level: {risk_report['risk_level']}\n\n"
        risk_text += "\n".join(f"- {issue}" for issue in risk_report["issues"])
    else:
        risk_text = f"### Risk Level: {risk_report['risk_level']}\n\n✅ No significant issues detected."

    # Roadmap grouped by priority
    roadmap_text = ""
    for priority in [1, 2, 3]:
        items = [r["action"] for r in roadmap if r["priority"] == priority]
        if items:
            roadmap_text += f"### Priority {priority}\n"
            roadmap_text += "\n".join(f"- {item}" for item in items)
            roadmap_text += "\n\n"
    if not roadmap_text:
        roadmap_text = "✅ No major improvements needed — resume is well aligned."

    # Interview questions table
    questions_table = [
        [q["skill"], q["difficulty"], q["question"]] for q in interview_questions
    ]

    return (
        gauge_fig,
        confidence_text,
        recruiter_summary,
        radar_fig,
        bar_fig,
        heatmap_fig,
        skills_table,
        transfer_table,
        risk_text,
        questions_table,
        roadmap_text,
    )


# ------------------------------------------------------------------
# CUSTOM DARK THEME
# ------------------------------------------------------------------

custom_theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
).set(
    body_background_fill="#14141f",
    background_fill_primary="#1e1e2f",
    background_fill_secondary="#26263a",
    border_color_primary="#3a3a4d",
    block_background_fill="#1e1e2f",
    block_border_color="#3a3a4d",
    block_label_text_color="#f5f5f5",
    body_text_color="#f5f5f5",
    button_primary_background_fill="#6c5ce7",
    button_primary_text_color="#ffffff",
)

CUSTOM_CSS = """
.gradio-container {
    font-family: \'Inter\', sans-serif;
    background-color: #14141f !important;
}

#title_banner {
    text-align: center;
    padding: 10px 0 20px 0;
}
#title_banner h1 {
    font-size: 2rem;
    background: linear-gradient(90deg, #6c5ce7, #00cec9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.gr-button, button {
    border-radius: 12px !important;
}
.gr-box, .gr-panel, .block {
    border-radius: 16px !important;
}

/* ---- Force dark textboxes/inputs (this is the actual fix) ---- */
textarea, input[type="text"], input[type="search"] {
    background-color: #1e1e2f !important;
    color: #f5f5f5 !important;
    border: 1px solid #3a3a4d !important;
    border-radius: 10px !important;
}

textarea::placeholder, input::placeholder {
    color: #9a9ab0 !important;
    opacity: 1 !important;
}

/* Labels above inputs */
label, .gr-form label, span {
    color: #f5f5f5 !important;
}

/* File upload box */
.file-preview, .upload-box {
    background-color: #1e1e2f !important;
    color: #f5f5f5 !important;
    border: 1px dashed #3a3a4d !important;
}

/* ---- Dataframe / table styling (Gradio wraps tables in nested divs) ---- */
.gr-dataframe, .dataframe-wrap, div[data-testid="dataframe"] {
    background-color: #1e1e2f !important;
}

table, .dataframe, .table-wrap table {
    background-color: #1e1e2f !important;
    color: #f5f5f5 !important;
    border-color: #3a3a4d !important;
}

table thead, table thead tr, table th,
.dataframe thead, .dataframe thead tr, .dataframe th {
    background-color: #26263a !important;
    color: #f5f5f5 !important;
    border-color: #3a3a4d !important;
}

table tbody, table tbody tr, table td,
.dataframe tbody, .dataframe tbody tr, .dataframe td {
    background-color: #1e1e2f !important;
    color: #f5f5f5 !important;
    border-color: #3a3a4d !important;
}

/* Gradio 5/6 sometimes uses svelte-generated classes on cells directly */
[class*="dataframe"] td, [class*="dataframe"] th,
[class*="table"] td, [class*="table"] th {
    background-color: #1e1e2f !important;
    color: #f5f5f5 !important;
}

/* Row hover highlight so it stays readable */
table tbody tr:hover, .dataframe tbody tr:hover {
    background-color: #2a2a40 !important;
}

/* Scrollbar area / empty cells sometimes render as plain white blocks */
.gr-dataframe *, div[data-testid="dataframe"] * {
    background-color: transparent;
}
"""


# ------------------------------------------------------------------
# BUILD THE GRADIO APP
# ------------------------------------------------------------------

with gr.Blocks(title="Resume Intelligence Copilot") as demo:

    gr.HTML("""
    <div id="title_banner">
        <h1>🧠 Resume Intelligence Copilot</h1>
        <p>Explainable AI hiring insights — beyond the ATS score.</p>
    </div>
    """)

    # ---- Input row (always visible above tabs) ----
    with gr.Row():
        with gr.Column(scale=1):
            resume_input = gr.File(label="📄 Upload Resume (.pdf or .txt)", file_types=[".pdf", ".txt"])
        with gr.Column(scale=2):
            jd_input = gr.Textbox(
                label="💼 Job Description",
                placeholder="Paste the full job description here...",
                lines=6,
            )
    analyze_btn = gr.Button("🔍 Analyze Resume", variant="primary", size="lg")

    # ---- Tabs ----
    with gr.Tabs():

        # TAB 1: OVERVIEW
        with gr.Tab("📊 Overview"):
            with gr.Row():
                gauge_output = gr.Plot(label="Match Score")
            confidence_output = gr.Markdown()
            summary_output = gr.Markdown(label="Recruiter Summary")

        # TAB 2: ANALYSIS
        with gr.Tab("📈 Analysis"):
            with gr.Row():
                radar_output = gr.Plot(label="Resume Strength Radar")
                bar_output = gr.Plot(label="Section-Level Scores")

        # TAB 3: SKILLS
        with gr.Tab("🧩 Skills"):
            heatmap_output = gr.Plot(label="Missing Skills Heatmap")
            gr.Markdown("### Skill Match Table")
            skills_table_output = gr.Dataframe(
                headers=["Skill", "Status", "Importance", "Match Score"],
                interactive=False,
            )
            gr.Markdown("### Skill Transferability Report")
            transfer_table_output = gr.Dataframe(
                headers=["Skill", "Transferability", "Learning Difficulty", "Related Skills", "Reason"],
                interactive=False,
            )

        # TAB 4: RISKS
        with gr.Tab("⚠️ Risks"):
            risk_output = gr.Markdown()

        # TAB 5: INTERVIEW QUESTIONS
        with gr.Tab("🎤 Interview Questions"):
            questions_output = gr.Dataframe(
                headers=["Skill", "Difficulty", "Question"],
                interactive=False,
            )

        # TAB 6: ROADMAP
        with gr.Tab("🗺️ Roadmap"):
            roadmap_output = gr.Markdown()

    # ---- Wire the button to the pipeline ----
    analyze_btn.click(
        fn=run_pipeline,
        inputs=[resume_input, jd_input],
        outputs=[
            gauge_output,
            confidence_output,
            summary_output,
            radar_output,
            bar_output,
            heatmap_output,
            skills_table_output,
            transfer_table_output,
            risk_output,
            questions_output,
            roadmap_output,
        ],
    )

    gr.Markdown(
        "---\n<center>Built with ❤️ using Sentence Transformers, spaCy, and Gradio — "
        "100% free & open-source models.</center>"
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=True, theme=custom_theme, css=CUSTOM_CSS)
