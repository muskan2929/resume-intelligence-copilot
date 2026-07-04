"""
resume_parser.py

Responsible for:
1. Extracting raw text from resume files (PDF, TXT).
2. Cleaning the extracted text.
3. Splitting the resume into logical sections (Skills, Experience,
   Projects, Education, Achievements, etc.) using simple heading
   detection + spaCy for sentence segmentation.

Designed to be beginner-friendly: every function does ONE clear job.
"""

import re
from typing import Dict, List

import pdfplumber
import spacy

# Load spaCy's small English model once at import time.
# This is used for sentence splitting and basic NLP cleanup.
nlp = spacy.load("en_core_web_sm")


# ------------------------------------------------------------------
# 1. TEXT EXTRACTION
# ------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF resume using pdfplumber.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A single string containing all text from every page.
    """
    full_text = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Some pages might be empty/images-only
                full_text.append(page_text)

    return "\n".join(full_text)


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract raw text from a plain .txt resume file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        The file's text content as a string.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_resume_text(file_path: str) -> str:
    """
    Detects file type by extension and routes to the correct extractor.

    Args:
        file_path: Path to the resume file (.pdf or .txt).

    Returns:
        Extracted raw text.

    Raises:
        ValueError: If the file type is not supported.
    """
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {file_path}. Use .pdf or .txt"
        )


# ------------------------------------------------------------------
# 2. TEXT CLEANING
# ------------------------------------------------------------------

def clean_text(raw_text: str) -> str:
    """
    Cleans raw resume text by:
    - Removing extra whitespace/newlines.
    - Removing weird bullet characters.
    - Normalizing spacing around punctuation.

    Args:
        raw_text: Unprocessed text extracted from a file.

    Returns:
        Cleaned text ready for section splitting.
    """
    text = raw_text.replace("\r", "\n")

    # Remove common bullet symbols
    text = re.sub(r"[•●▪◦‣∙]", "-", text)

    # Collapse multiple blank lines into one
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ------------------------------------------------------------------
# 3. SECTION SPLITTING
# ------------------------------------------------------------------

# Common resume section headings mapped to a normalized section name.
# We match case-insensitively and allow minor variations.
SECTION_HEADINGS: Dict[str, List[str]] = {
    "summary": ["summary", "objective", "profile", "about me"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history"],
    "projects": ["projects", "personal projects", "academic projects"],
    "education": ["education", "academic background", "qualifications"],
    "achievements": ["achievements", "awards", "honors", "certifications"],
    "leadership": ["leadership", "extracurricular", "activities"],
}


def _match_section_heading(line: str) -> str:
    """
    Checks if a given line is a section heading and returns its
    normalized section name (e.g. "skills"), or an empty string if
    the line is not a recognized heading.

    Args:
        line: A single line of text from the resume.

    Returns:
        Normalized section name, or "" if not a heading.
    """
    clean_line = line.strip().lower().rstrip(":")

    # Headings are usually short (a few words), so skip long lines early
    if len(clean_line.split()) > 5:
        return ""

    for section_name, variants in SECTION_HEADINGS.items():
        if clean_line in variants:
            return section_name

    return ""


def split_into_sections(cleaned_text: str) -> Dict[str, str]:
    """
    Splits cleaned resume text into a dictionary of sections.

    Any text before the first recognized heading is stored under
    "header" (usually name/contact info).

    Args:
        cleaned_text: Text that has already been cleaned.

    Returns:
        Dict mapping section name -> section text.
        Example: {"header": "...", "skills": "...", "experience": "..."}
    """
    lines = cleaned_text.split("\n")

    sections: Dict[str, List[str]] = {"header": []}
    current_section = "header"

    for line in lines:
        heading = _match_section_heading(line)

        if heading:
            current_section = heading
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line)

    # Join each section's lines back into a single string
    return {
        name: "\n".join(content_lines).strip()
        for name, content_lines in sections.items()
        if "\n".join(content_lines).strip()  # drop empty sections
    }


# ------------------------------------------------------------------
# 4. SKILL EXTRACTION (simple, list-based)
# ------------------------------------------------------------------

def extract_skills_list(skills_section_text: str) -> List[str]:
    """
    Converts a raw "skills" section into a clean list of individual
    skill strings. Handles comma-separated, bullet, or newline-separated
    formats.

    Args:
        skills_section_text: Raw text of the skills section.

    Returns:
        List of individual skill strings (lowercased, deduplicated).
    """
    if not skills_section_text:
        return []

    # Replace bullets/newlines with commas so we can split uniformly
    normalized = re.sub(r"[\n\-]+", ",", skills_section_text)
    raw_skills = normalized.split(",")

    skills = []
    for skill in raw_skills:
        skill_clean = skill.strip().lower()
        if skill_clean and len(skill_clean) < 40:  # filter junk/long lines
            skills.append(skill_clean)

    # Deduplicate while preserving order
    seen = set()
    unique_skills = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            unique_skills.append(s)

    return unique_skills


# ------------------------------------------------------------------
# 5. MASTER FUNCTION — parses a resume file end-to-end
# ------------------------------------------------------------------

def parse_resume(file_path: str) -> Dict:
    """
    Full pipeline: file -> raw text -> cleaned text -> sections -> skills list.

    Args:
        file_path: Path to the resume file (.pdf or .txt).

    Returns:
        Dict with keys:
            - "raw_text": original extracted text
            - "cleaned_text": cleaned full text
            - "sections": dict of section_name -> text
            - "skills_list": list of extracted skills
            - "word_count": total word count (used later for risk checks)
    """
    raw_text = extract_resume_text(file_path)
    cleaned = clean_text(raw_text)
    sections = split_into_sections(cleaned)
    skills_list = extract_skills_list(sections.get("skills", ""))

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "sections": sections,
        "skills_list": skills_list,
        "word_count": len(cleaned.split()),
    }
