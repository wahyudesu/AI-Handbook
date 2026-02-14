# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an **AI/Data Science Handbook** project written in Markdown, designed as a compact practical guide for Indonesian students learning data science and machine learning for the short time, sehingga . The target audience is university students, competition participants, and practitioners.

## Project Structure

- **`Data-AI-Handbook.md`** - Main handbook file (the book content)
- **`project.md`** - Project notes, chapter structure, and TODOs
- **`marketing.md`** - Marketing notes
- **`notebook/`** - Reference Jupyter notebooks for examples
- **`Pasted image*.png`** - Diagrams and visualizations referenced in the handbook

## Writing Style & Tone

The handbook follows a specific tone defined in the frontmatter YAML of `Data-AI-Handbook.md`:
- **Approach**: Like friends sharing knowledge — confident but not condescending, leaving room for exploration
- **Style**: Straightforward, instructional, semi-formal, practical
- **Format**: Compact, dense, no fluff — critical since handbook is under 40 pages
- **Voice direction**: Slightly older to slightly younger (experienced peer to beginner)

## Content Elements per Chapter

Each chapter typically contains:
- **Paragraphs**: Explanations, theory breakdown
- **Bullet lists**: Resources, checklists
- **Tables**: Comparisons, type explanations
- **Images**: Visualizations to aid understanding
- **Code**: Examples of library/framework usage
- **Quotes**: Recommendations, questions
- **`^[...]` footnotes**: Brief chapter descriptions
- **`%%...%%` comments**: References/links
- **`==...==` highlights**: Author notes/todos

## Chapter Structure

1. **Pengantar** (Introduction) - Background
2. **Data and Digital Use case** - Getting data, checklists, next steps
3. **Data Analysis and Preprocessing**
   - Data Preprocessing (scenario-based techniques)
   - EDA (types first, then explanation with plots)
   - Statistical Analysis (theory-heavy)
   - Feature Engineering
   - Data Mining (Kaggle project examples)
4. **Model and Evaluation**
   - Core Concepts of ML
   - Types of ML
   - Model Training and Evaluation
   - Hyperparameter tuning (theory-heavy)
   - Ensemble Learning (theory + practice)
5. **Pre-Trained and Generative AI** (planned)
6. **AI Agent** (planned)
7. **Great Books** - Reading recommendations

## Editing Guidelines

### CRITICAL - Remove Work-in-Progress Markers

Before considering any section complete, **always remove**:
- `^[...]` placeholder descriptions after headers
- `==...==` inline TODO comments
- `%%...%%` reference blocks that should be formatted properly
- Draft notes like "==perlu satu halaman lagi=="
- Duplicate/copied code blocks from previous sections (e.g., the ANN section currently contains copied hyperparameter tuning code)

### Language & Terminology

- **Primary language**: Indonesian with technical terms in English
- **Technical terms**: Keep in original form (e.g., "supervised learning", "hyperparameter", "ensemble methods")
- **Spelling consistency**: Choose one variant and stick to it (algorithm/algoritma, model/model, etc.)
- **NO code copying**: Never copy-paste code from other sections without verifying it applies to the current context

### Code Examples

- Must be runnable Python code
- Include necessary imports
- Prefer scikit-learn, pandas, numpy, matplotlib, seaborn
- Test code before adding to handbook
- Comments should explain *what* and *why*, not just *what*

### Links

- Verify all links before committing
- Use descriptive link text
- Reference external resources (Kaggle, documentation, tutorials)
- Check that local image references exist

## Status

The handbook is approximately **75% complete** per the frontmatter. Chapters 1-3 are substantially written. Chapters 4-5 are planned but not yet developed.

## Development Tasks

now: 9,040 words, 70,000 characters
expected: 10,000 words, 100,000 characters
