---
title: Resume Intelligence Copilot
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.31.5"
app_file: app.py
pinned: false
---

# 🧠 Resume Intelligence Copilot

An explainable AI system that analyzes a candidate's resume against a job
description and produces recruiter-grade hiring insights — not just an ATS score.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🚀 Overview

Traditional ATS tools only give a match percentage based on keyword overlap.
**Resume Intelligence Copilot** goes further — using semantic embeddings
(sentence-transformers/all-MiniLM-L6-v2) to understand meaning, not just
keywords, and generates a full explainable hiring report.

---

## ✨ Features

| Feature | Description |
|---|---|
| Resume Match Score | Semantic similarity score + confidence rating |
| Resume Strength Radar | 7-dimension radar chart (skills, projects, experience, leadership, etc.) |
| Missing Skills Heatmap | Critical / Important / Optional skill gap visualization |
| Skill Transferability Score | Estimates how easily a candidate could learn a missing skill |
| Resume Risk Detector | Flags keyword stuffing, vague achievements, employment gaps, etc. |
| Recruiter Summary | Auto-generated recruiter-style writeup |
| Improvement Roadmap | Priority-ranked action plan for the candidate |
| Interview Question Generator | Auto-generated questions by difficulty |
| Explainable AI | Every score is traceable to matched/missing/transferable skills |

---

## 🏗️ Architecture

Resume (PDF/TXT) and Job Description are parsed by resume_parser.py into
cleaned sections, embedded and scored by embedding_model.py using MiniLM
cosine similarity, processed by analyzer.py into heatmap, transferability,
risk, summary, roadmap, and interview question data, visualized by
visualizations.py using Plotly, and displayed through the Gradio UI in app.py.

---

## 📂 Project Structure

resume-intelligence-copilot/
    app.py
    requirements.txt
    README.md
    model/
        embedding_model.py
        analyzer.py
    utils/
        resume_parser.py
        visualizations.py
    data/
        sample_resume.txt
    notebooks/
    screenshots/

---

## 💻 Run in Google Colab

    !git clone https://github.com/YOUR-USERNAME/resume-intelligence-copilot.git
    %cd resume-intelligence-copilot
    !pip install -r requirements.txt
    !python -m spacy download en_core_web_sm
    !python app.py

---

## 🐙 Push to GitHub (from Colab)

    git init
    git add .
    git commit -m "Initial commit: Resume Intelligence Copilot"
    git branch -M main
    git remote add origin https://github.com/YOUR-USERNAME/resume-intelligence-copilot.git
    git push -u origin main

Using a Personal Access Token (PAT): GitHub no longer accepts password
auth for git pushes. Generate a token at
GitHub -> Settings -> Developer settings -> Personal access tokens -> Fine-grained tokens,
grant it repo scope, then push using:

    git remote set-url origin https://YOUR-USERNAME:YOUR_PAT@github.com/YOUR-USERNAME/resume-intelligence-copilot.git
    git push -u origin main

---

## 🤗 Deploy on Hugging Face Spaces (Free)

1. Create a new Space at huggingface.co/new-space with SDK Gradio and Hardware CPU basic (free).
2. Push this repo to the Space:

    git remote add space https://huggingface.co/spaces/muskan2929/resume-intelligence-copilot
    git push space main

3. Hugging Face will auto-install requirements.txt and run app.py.
4. Your app will be live at https://huggingface.co/spaces/muskan2929/resume-intelligence-copilot

---

## 📸 Screenshots

Add screenshots of the Overview, Skills, and Risk tabs here.

    screenshots/
        overview.png
        skills_heatmap.png
        roadmap.png

---

## 🔮 Future Improvements

- Multi-resume batch comparison for a single job posting
- Fine-tuned domain-specific embedding model
- PDF export of the full recruiter report
- Multi-language resume support

---

## 📝 License

MIT License — free to use, modify, and deploy.
