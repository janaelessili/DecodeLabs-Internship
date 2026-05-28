# 🚀 Tech Stack Recommender

> **Week 3 AI Internship Project — AI Recommendation Logic**

A beginner-friendly recommendation system that suggests the **Top 3 tech career paths / tech stacks** based on a user's skills and interests. Built with Python using **TF-IDF vectorization** and **cosine similarity**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Sample Input / Output](#sample-input--output)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

---

## Overview

The program:

1. Asks the user to enter **at least 3** skills or interests.
2. Compares those inputs against a built-in dataset of **10 tech roles** (each tagged with related skills).
3. Uses **TF-IDF** (Term Frequency – Inverse Document Frequency) to convert text into numerical vectors.
4. Computes **cosine similarity** between the user's vector and each role's vector.
5. Returns the **Top 3** most relevant career paths along with their similarity scores.

---

## Requirements

| Dependency     | Version  |
| -------------- | -------- |
| Python         | 3.8+     |
| scikit-learn   | ≥ 1.0    |

---

## Installation & Setup

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/tech-stack-recommender.git
cd tech-stack-recommender

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install scikit-learn
```

---

## How to Run

```bash
python tech_stack_recommender.py
```

Follow the on-screen prompts to enter your skills. Type **`done`** when finished.

---

## Sample Input / Output

### Input

```
Enter your skills or interests one at a time.
Type at least 3, then type 'done' when finished.
------------------------------------------------------------
  Skill/Interest #1: python
  Skill/Interest #2: machine learning
  Skill/Interest #3: data analysis
  Skill/Interest #4: done
```

### Output

```
============================================================
       📊  TOP RECOMMENDATIONS
============================================================

  #1  Data Scientist
       Similarity Score : 0.5765
       Related Skills   : python r statistics machine learning pandas numpy matplotlib data analysis visualization jupyter scikit-learn
       ------------------------------------------------

  #2  Machine Learning Engineer
       Similarity Score : 0.4518
       Related Skills   : python tensorflow pytorch deep learning neural networks nlp computer vision model deployment mlops scikit-learn
       ------------------------------------------------

  #3  AI / NLP Engineer
       Similarity Score : 0.2842
       Related Skills   : python nlp natural language processing transformers huggingface bert gpt text classification sentiment analysis chatbot deep learning
       ------------------------------------------------

============================================================
  Thank you for using the Tech Stack Recommender!
============================================================
```

> **Note:** Exact similarity scores may vary slightly depending on the scikit-learn version.

---

## How It Works

1. **Dataset** — A list of 10 tech roles, each described by a string of related skills/tags.
2. **User Input** — The user's skills are collected and joined into a single text string.
3. **TF-IDF Vectorization** — `TfidfVectorizer` from scikit-learn converts all text (roles + user input) into numerical vectors. Words that appear in many roles are down-weighted (IDF), while words specific to a few roles are up-weighted.
4. **Cosine Similarity** — Measures the angle between the user's vector and each role's vector. A score of **1.0** means a perfect match; **0.0** means no overlap at all.
5. **Ranking** — Roles are sorted by similarity score in descending order, and the top 3 are displayed.

---

## Project Structure

```
tech-stack-recommender/
├── tech_stack_recommender.py   # Main Python script
└── README.md                   # Project documentation (this file)
```

---

## Future Improvements

- 📂 **External dataset** — Load roles and skills from a CSV or JSON file instead of hardcoding them.
- 🌐 **Web interface** — Build a simple Flask or Streamlit front-end for a more interactive experience.
- 🧠 **Smarter matching** — Use word embeddings (e.g., Word2Vec, sentence-transformers) for semantic similarity instead of keyword-based TF-IDF.
- 📊 **More roles** — Expand the dataset to cover 50+ career paths for broader recommendations.
- 🔄 **Feedback loop** — Let users rate recommendations to improve future results.
- ✅ **Unit tests** — Add `pytest` tests to verify recommendation accuracy and edge cases.

---

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---
