"""
analyzer.py

The core "intelligence" layer of the app. Builds on top of
embedding_model.py and resume_parser.py to produce:

1. Job description skill extraction
2. Missing Skills Heatmap data (Critical / Important / Optional)
3. Skill Transferability Score (e.g. TensorFlow -> PyTorch)
4. Resume Risk Detector (keyword stuffing, gaps, weak sections, etc.)
5. Recruiter Summary (short human-style report)
6. Resume Improvement Roadmap (priority-ordered suggestions)
7. Interview Question Generator
8. Explainability data (matched / missing / transferable skills)

Each function returns plain Python dicts/lists so they plug directly
into the Gradio UI and Plotly charts in later phases.
"""

import re
from typing import Dict, List, Tuple

from model.embedding_model import compute_similarity, compute_similarity_matrix


# ------------------------------------------------------------------
# 1. JOB DESCRIPTION SKILL EXTRACTION
# ------------------------------------------------------------------

# A reasonably broad list of common tech/soft skills to detect in a JD.
# Kept simple and editable — a college-level project doesn't need a
# giant NLP skill-extraction model here.
COMMON_SKILLS = [
    "python", "java", "c++", "javascript", "typescript", "sql", "r",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "spacy", "nltk", "opencv", "docker", "kubernetes", "aws", "gcp",
    "azure", "git", "github", "linux", "flask", "django", "fastapi",
    "react", "node.js", "html", "css", "nlp", "computer vision",
    "machine learning", "deep learning", "data analysis", "statistics",
    "communication", "leadership", "teamwork", "problem solving",
]


def extract_required_skills(job_description: str) -> List[str]:
    """
    Scans the job description text and returns the list of known
    skills (from COMMON_SKILLS) that are mentioned in it.

    Args:
        job_description: Full job description text.

    Returns:
        List of matched required skills (lowercased).
    """
    jd_lower = job_description.lower()
    found = [skill for skill in COMMON_SKILLS if skill in jd_lower]
    return found


def categorize_skill_importance(
    required_skills: List[str], job_description: str
) -> Dict[str, str]:
    """
    Categorizes each required skill as Critical / Important / Optional
    based on simple positional + repetition heuristics:
    - Mentioned multiple times, or near words like "must", "required" -> Critical
    - Mentioned once in main body -> Important
    - Mentioned near "nice to have", "plus", "bonus" -> Optional

    Args:
        required_skills: List of skills found in the JD.
        job_description: Full job description text.

    Returns:
        Dict mapping skill -> "Critical" | "Important" | "Optional"
    """
    jd_lower = job_description.lower()
    categories = {}

    optional_markers = ["nice to have", "plus", "bonus", "preferred", "optional"]
    critical_markers = ["must have", "required", "essential", "strong experience"]

    for skill in required_skills:
        occurrences = jd_lower.count(skill)

        # Look at the text window around each mention for context clues
        idx = jd_lower.find(skill)
        window = jd_lower[max(0, idx - 40): idx + 40]

        if any(marker in window for marker in optional_markers):
            categories[skill] = "Optional"
        elif any(marker in window for marker in critical_markers) or occurrences > 1:
            categories[skill] = "Critical"
        else:
            categories[skill] = "Important"

    return categories


# ------------------------------------------------------------------
# 2. MISSING SKILLS HEATMAP DATA
# ------------------------------------------------------------------

def build_missing_skills_heatmap(
    resume_skills: List[str], job_description: str
) -> List[Dict]:
    """
    Compares candidate skills vs required job skills and produces
    heatmap-ready data: which skills are matched vs missing, and
    their importance category.

    Args:
        resume_skills: Skills extracted from the resume.
        job_description: Full job description text.

    Returns:
        List of dicts, one per required skill:
        {
            "skill": "pytorch",
            "status": "Matched" | "Missing",
            "importance": "Critical" | "Important" | "Optional",
            "match_score": float (0-1, semantic similarity to closest resume skill)
        }
    """
    required_skills = extract_required_skills(job_description)
    importance_map = categorize_skill_importance(required_skills, job_description)

    if not required_skills:
        return []

    similarity_matrix = compute_similarity_matrix(resume_skills, required_skills)

    heatmap_data = []
    for j, job_skill in enumerate(required_skills):
        # Best matching resume skill for this job skill (if any resume skills exist)
        best_score = float(similarity_matrix[:, j].max()) if resume_skills else 0.0

        # A high similarity (>0.75) counts as a direct match even if the
        # exact string differs slightly (e.g. "sklearn" vs "scikit-learn")
        status = "Matched" if best_score > 0.75 or job_skill in resume_skills else "Missing"

        heatmap_data.append({
            "skill": job_skill,
            "status": status,
            "importance": importance_map.get(job_skill, "Important"),
            "match_score": round(best_score, 3),
        })

    return heatmap_data


# ------------------------------------------------------------------
# 3. SKILL TRANSFERABILITY SCORE
# ------------------------------------------------------------------

# Simple knowledge base of related skill clusters used to explain
# *why* a skill might be transferable (keeps things explainable
# rather than a black box).
RELATED_SKILLS_MAP: Dict[str, List[str]] = {
    "pytorch": ["tensorflow", "keras", "machine learning", "deep learning"],
    "tensorflow": ["pytorch", "keras", "machine learning", "deep learning"],
    "keras": ["tensorflow", "pytorch", "deep learning"],
    "django": ["flask", "fastapi", "python"],
    "flask": ["django", "fastapi", "python"],
    "aws": ["gcp", "azure", "docker", "kubernetes"],
    "gcp": ["aws", "azure", "docker"],
    "azure": ["aws", "gcp", "docker"],
    "react": ["javascript", "typescript", "html", "css"],
    "kubernetes": ["docker", "aws", "gcp"],
}


def estimate_transferability(missing_skill: str, resume_skills: List[str]) -> Dict:
    """
    For a missing skill, checks if the candidate has related skills
    that suggest they could learn it quickly.

    Args:
        missing_skill: A skill required by the job but absent from the resume.
        resume_skills: List of skills the candidate actually has.

    Returns:
        Dict with:
            - "skill": the missing skill
            - "transferability": "High" | "Medium" | "Low" | "None"
            - "learning_difficulty": "Easy" | "Moderate" | "Hard"
            - "related_skills_found": list of matching related skills
            - "reason": human-readable explanation
    """
    related_pool = RELATED_SKILLS_MAP.get(missing_skill, [])
    related_found = [s for s in related_pool if s in resume_skills]

    if not related_pool:
        return {
            "skill": missing_skill,
            "transferability": "None",
            "learning_difficulty": "Hard",
            "related_skills_found": [],
            "reason": f"No related skills on file to support learning '{missing_skill}'.",
        }

    if len(related_found) >= 2:
        transferability, difficulty = "High", "Easy"
    elif len(related_found) == 1:
        transferability, difficulty = "Medium", "Moderate"
    else:
        transferability, difficulty = "Low", "Hard"

    if related_found:
        reason = (
            f"Candidate has experience with {', '.join(related_found)}, "
            f"which shares strong conceptual overlap with '{missing_skill}'."
        )
    else:
        reason = f"No overlapping skills found for '{missing_skill}'."

    return {
        "skill": missing_skill,
        "transferability": transferability,
        "learning_difficulty": difficulty,
        "related_skills_found": related_found,
        "reason": reason,
    }


def build_transferability_report(
    resume_skills: List[str], missing_skills: List[str]
) -> List[Dict]:
    """
    Runs estimate_transferability() for every missing skill.

    Args:
        resume_skills: Candidate's actual skills.
        missing_skills: Skills required by the job but missing from resume.

    Returns:
        List of transferability dicts (see estimate_transferability).
    """
    return [estimate_transferability(skill, resume_skills) for skill in missing_skills]


# ------------------------------------------------------------------
# 4. RESUME RISK DETECTOR
# ------------------------------------------------------------------

def detect_keyword_stuffing(resume_skills: List[str], word_count: int) -> Tuple[bool, str]:
    """
    Flags resumes that list an unusually high number of skills relative
    to overall resume length — a common keyword-stuffing pattern.
    """
    if word_count == 0:
        return False, ""
    skill_density = len(resume_skills) / word_count
    if skill_density > 0.15 and len(resume_skills) > 20:
        return True, (
            f"High skill-to-content ratio ({len(resume_skills)} skills listed "
            f"in only {word_count} words) suggests possible keyword stuffing."
        )
    return False, ""


def detect_repeated_buzzwords(cleaned_text: str) -> Tuple[bool, str]:
    """
    Flags resumes that overuse generic buzzwords instead of concrete detail.
    """
    buzzwords = ["synergy", "go-getter", "hardworking", "team player",
                 "detail-oriented", "results-driven", "dynamic", "passionate"]
    text_lower = cleaned_text.lower()
    counts = {word: text_lower.count(word) for word in buzzwords}
    total_buzz = sum(counts.values())

    if total_buzz >= 4:
        used = [w for w, c in counts.items() if c > 0]
        return True, f"Overuse of generic buzzwords detected: {', '.join(used)}."
    return False, ""


def detect_missing_achievements(experience_text: str) -> Tuple[bool, str]:
    """
    Flags experience sections with no numbers/metrics — a sign of
    vague, non-measurable achievements.
    """
    if not experience_text:
        return False, ""
    has_numbers = bool(re.search(r"\d+%|\d+\+|\$\d+|\d+x", experience_text))
    if not has_numbers:
        return True, "Experience section lacks measurable achievements (no %, numbers, or metrics found)."
    return False, ""


def detect_weak_projects(projects_text: str) -> Tuple[bool, str]:
    """
    Flags project descriptions that are too short to convey real depth.
    """
    if not projects_text:
        return False, ""
    word_count = len(projects_text.split())
    if word_count < 20:
        return True, f"Projects section is very brief ({word_count} words) — lacks depth/detail."
    return False, ""


def detect_employment_gaps(experience_text: str) -> Tuple[bool, str]:
    """
    Very simple gap detector: looks for year ranges and flags large
    numeric gaps between consecutive years mentioned.
    (Kept intentionally simple/explainable rather than using a full
    date-parsing library.)
    """
    years = sorted(set(int(y) for y in re.findall(r"(20\d{2}|19\d{2})", experience_text)))
    if len(years) < 2:
        return False, ""

    for i in range(len(years) - 1):
        gap = years[i + 1] - years[i]
        if gap >= 2:
            return True, f"Possible employment gap detected between {years[i]} and {years[i+1]}."
    return False, ""


def detect_unprofessional_wording(cleaned_text: str) -> Tuple[bool, str]:
    """
    Flags casual/unprofessional language that shouldn't appear on a resume.
    """
    casual_words = ["lol", "guys", "awesome sauce", "kinda", "gonna", "stuff like that"]
    text_lower = cleaned_text.lower()
    found = [w for w in casual_words if w in text_lower]
    if found:
        return True, f"Unprofessional wording detected: {', '.join(found)}."
    return False, ""


def detect_length_issues(word_count: int) -> Tuple[bool, str]:
    """
    Flags resumes that are unusually short or unusually long.
    """
    if word_count < 150:
        return True, f"Resume is very short ({word_count} words) — may lack sufficient detail."
    if word_count > 1200:
        return True, f"Resume is very long ({word_count} words) — consider trimming to be more concise."
    return False, ""


def run_risk_detector(parsed_resume: Dict) -> Dict:
    """
    Runs all individual risk checks and aggregates them into a single
    risk report with an overall Risk Level.

    Args:
        parsed_resume: Output of resume_parser.parse_resume().

    Returns:
        Dict with:
            - "risk_level": "Low" | "Medium" | "High"
            - "issues": list of human-readable issue explanations
            - "issue_count": number of issues detected
    """
    sections = parsed_resume["sections"]
    issues = []

    checks = [
        detect_keyword_stuffing(parsed_resume["skills_list"], parsed_resume["word_count"]),
        detect_repeated_buzzwords(parsed_resume["cleaned_text"]),
        detect_missing_achievements(sections.get("experience", "")),
        detect_weak_projects(sections.get("projects", "")),
        detect_employment_gaps(sections.get("experience", "")),
        detect_unprofessional_wording(parsed_resume["cleaned_text"]),
        detect_length_issues(parsed_resume["word_count"]),
    ]

    for flagged, message in checks:
        if flagged:
            issues.append(message)

    # Simple thresholding for overall risk level
    if len(issues) == 0:
        risk_level = "Low"
    elif len(issues) <= 2:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_level": risk_level,
        "issues": issues,
        "issue_count": len(issues),
    }


# ------------------------------------------------------------------
# 5. RECRUITER SUMMARY
# ------------------------------------------------------------------

def generate_recruiter_summary(
    match_result: Dict, heatmap_data: List[Dict], risk_report: Dict
) -> str:
    """
    Generates a short, human-readable recruiter-style summary using
    simple templated logic (no external LLM needed — keeps the
    project fully free/offline-capable).

    Args:
        match_result: Output of compute_overall_match_score().
        heatmap_data: Output of build_missing_skills_heatmap().
        risk_report: Output of run_risk_detector().

    Returns:
        A multi-line summary string.
    """
    lines = []

    score = match_result["overall_score"]
    if score >= 75:
        lines.append("Strong overall alignment with the job requirements.")
    elif score >= 50:
        lines.append("Moderate alignment with the job requirements.")
    else:
        lines.append("Limited alignment with the job requirements.")

    matched_critical = [h for h in heatmap_data if h["status"] == "Matched" and h["importance"] == "Critical"]
    missing_critical = [h for h in heatmap_data if h["status"] == "Missing" and h["importance"] == "Critical"]

    if matched_critical:
        skills_str = ", ".join(h["skill"] for h in matched_critical)
        lines.append(f"Covers critical requirements: {skills_str}.")

    if missing_critical:
        skills_str = ", ".join(h["skill"] for h in missing_critical)
        lines.append(f"Missing critical skills: {skills_str}.")

    if risk_report["risk_level"] == "Low":
        lines.append("No significant red flags detected in the resume.")
    else:
        lines.append(f"Resume risk level: {risk_report['risk_level']} — {risk_report['issue_count']} issue(s) found.")

    # Final recommendation
    if score >= 70 and risk_report["risk_level"] != "High":
        lines.append("Recommendation: Proceed to interview.")
    elif score >= 45:
        lines.append("Recommendation: Consider for interview with follow-up questions on gaps.")
    else:
        lines.append("Recommendation: Likely not a strong fit for this role.")

    return "\n".join(f"- {line}" for line in lines)


# ------------------------------------------------------------------
# 6. RESUME IMPROVEMENT ROADMAP
# ------------------------------------------------------------------

def generate_improvement_roadmap(
    heatmap_data: List[Dict], transferability_report: List[Dict], risk_report: Dict
) -> List[Dict]:
    """
    Builds a priority-ordered improvement roadmap for the candidate.

    Args:
        heatmap_data: Output of build_missing_skills_heatmap().
        transferability_report: Output of build_transferability_report().
        risk_report: Output of run_risk_detector().

    Returns:
        List of dicts: [{"priority": 1, "action": "..."}, ...]
    """
    roadmap = []

    # Priority 1: Critical missing skills with low transferability
    critical_missing = [h["skill"] for h in heatmap_data if h["status"] == "Missing" and h["importance"] == "Critical"]
    for skill in critical_missing:
        transfer_info = next((t for t in transferability_report if t["skill"] == skill), None)
        difficulty = transfer_info["learning_difficulty"] if transfer_info else "Unknown"
        roadmap.append({
            "priority": 1,
            "action": f"Learn or gain hands-on experience with '{skill}' (Estimated difficulty: {difficulty}).",
        })

    # Priority 2: Resume risk issues
    for issue in risk_report["issues"]:
        roadmap.append({"priority": 2, "action": f"Fix resume issue: {issue}"})

    # Priority 3: Important (non-critical) missing skills
    important_missing = [h["skill"] for h in heatmap_data if h["status"] == "Missing" and h["importance"] == "Important"]
    for skill in important_missing:
        roadmap.append({"priority": 3, "action": f"Consider adding exposure to '{skill}' to strengthen the profile."})

    return roadmap


# ------------------------------------------------------------------
# 7. INTERVIEW QUESTION GENERATOR
# ------------------------------------------------------------------

QUESTION_TEMPLATES = {
    "Easy": [
        "Can you explain what {skill} is and where you've used it?",
        "What was the outcome of your project involving {skill}?",
    ],
    "Medium": [
        "Describe a challenge you faced while using {skill} and how you solved it.",
        "How would you apply {skill} to a real-world business problem?",
    ],
    "Hard": [
        "How would you design a scalable system that heavily relies on {skill}?",
        "What are the trade-offs of using {skill} compared to alternative approaches?",
    ],
}


def generate_interview_questions(
    heatmap_data: List[Dict], resume_skills: List[str]
) -> List[Dict]:
    """
    Generates interview questions based on missing skills (to probe
    gaps) and matched/resume skills (to verify depth), across
    Easy/Medium/Hard difficulty levels.

    Args:
        heatmap_data: Output of build_missing_skills_heatmap().
        resume_skills: Candidate's actual skills (to ask depth questions).

    Returns:
        List of dicts: [{"skill": ..., "difficulty": ..., "question": ...}]
    """
    questions = []

    # Questions about matched skills (verify real depth)
    matched_skills = [h["skill"] for h in heatmap_data if h["status"] == "Matched"]
    for skill in matched_skills[:3]:  # limit to keep the list manageable
        questions.append({
            "skill": skill,
            "difficulty": "Medium",
            "question": QUESTION_TEMPLATES["Medium"][0].format(skill=skill),
        })

    # Questions about missing skills (probe willingness/ability to learn)
    missing_skills = [h["skill"] for h in heatmap_data if h["status"] == "Missing"]
    for skill in missing_skills[:3]:
        questions.append({
            "skill": skill,
            "difficulty": "Easy",
            "question": QUESTION_TEMPLATES["Easy"][0].format(skill=skill),
        })

    # A couple of harder, design-oriented questions from top resume skills
    for skill in resume_skills[:2]:
        questions.append({
            "skill": skill,
            "difficulty": "Hard",
            "question": QUESTION_TEMPLATES["Hard"][0].format(skill=skill),
        })

    return questions


# ------------------------------------------------------------------
# 8. MASTER ANALYSIS FUNCTION — ties everything together
# ------------------------------------------------------------------

def analyze_resume(parsed_resume: Dict, job_description: str) -> Dict:
    """
    Runs the FULL analysis pipeline and returns a single结构ured
    result dict — this is what the Gradio UI (Phase 6) will call.

    Args:
        parsed_resume: Output of resume_parser.parse_resume().
        job_description: Full job description text.

    Returns:
        Dict containing every piece of analysis:
        match_result, heatmap_data, transferability_report,
        risk_report, recruiter_summary, roadmap, interview_questions.
    """
    match_result = compute_overall_match_score(parsed_resume["sections"], job_description) \
        if False else None  # placeholder to keep import clean; replaced below

    # NOTE: compute_overall_match_score lives in embedding_model.py
    from model.embedding_model import compute_overall_match_score as _compute_score
    match_result = _compute_score(parsed_resume["sections"], job_description)

    heatmap_data = build_missing_skills_heatmap(parsed_resume["skills_list"], job_description)
    missing_skills = [h["skill"] for h in heatmap_data if h["status"] == "Missing"]
    transferability_report = build_transferability_report(parsed_resume["skills_list"], missing_skills)

    risk_report = run_risk_detector(parsed_resume)
    recruiter_summary = generate_recruiter_summary(match_result, heatmap_data, risk_report)
    roadmap = generate_improvement_roadmap(heatmap_data, transferability_report, risk_report)
    interview_questions = generate_interview_questions(heatmap_data, parsed_resume["skills_list"])

    return {
        "match_result": match_result,
        "heatmap_data": heatmap_data,
        "transferability_report": transferability_report,
        "risk_report": risk_report,
        "recruiter_summary": recruiter_summary,
        "roadmap": roadmap,
        "interview_questions": interview_questions,
    }
