"""
Tech Stack Recommender — Week 3 AI Internship Project
=====================================================
A simple recommendation system that matches user skills/interests
to tech career paths using TF-IDF vectorization and cosine similarity.

Author : [Jana Elessili]
Date   : May 2026
"""

# ── Step 1: Import required libraries ────────────────────────────────────────
# sklearn provides TF-IDF and cosine similarity implementations.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Step 2: Define the dataset of tech roles ─────────────────────────────────
# Each role is a dictionary with:
#   - "role"  : the career / tech-stack name
#   - "skills": a string of related skills and tags (used for matching)

tech_roles = [
    {
        "role": "Frontend Developer",
        "skills": "html css javascript react angular vue ui ux design responsive web development typescript tailwind bootstrap"
    },
    {
        "role": "Backend Developer",
        "skills": "python java node express django flask api rest database sql server microservices authentication"
    },
    {
        "role": "Data Scientist",
        "skills": "python r statistics machine learning pandas numpy matplotlib data analysis visualization jupyter scikit-learn"
    },
    {
        "role": "DevOps Engineer",
        "skills": "docker kubernetes aws azure ci cd jenkins terraform ansible linux cloud infrastructure automation monitoring"
    },
    {
        "role": "Mobile App Developer",
        "skills": "kotlin swift flutter react native java android ios mobile ui app development firebase"
    },
    {
        "role": "Machine Learning Engineer",
        "skills": "python tensorflow pytorch deep learning neural networks nlp computer vision model deployment mlops scikit-learn"
    },
    {
        "role": "Cybersecurity Analyst",
        "skills": "networking security penetration testing ethical hacking firewall encryption linux vulnerability assessment risk analysis compliance"
    },
    {
        "role": "Full Stack Developer",
        "skills": "html css javascript python node react database sql mongodb api rest git deployment docker"
    },
    {
        "role": "Cloud Architect",
        "skills": "aws azure gcp cloud infrastructure networking serverless terraform security architecture scalability containers"
    },
    {
        "role": "AI / NLP Engineer",
        "skills": "python nlp natural language processing transformers huggingface bert gpt text classification sentiment analysis chatbot deep learning"
    },
]


def get_user_interests() -> str:
    """
    Step 3: Collect at least 3 skills / interests from the user.
    Returns them as a single space-separated string for TF-IDF processing.
    """
    print("=" * 60)
    print("       >>  TECH STACK RECOMMENDER  <<")
    print("=" * 60)
    print()
    print("Enter your skills or interests one at a time.")
    print("Type at least 3, then type 'done' when finished.")
    print("-" * 60)

    interests: list[str] = []

    while True:
        # Show how many skills have been entered so far
        prompt = f"  Skill/Interest #{len(interests) + 1}: "
        user_input = input(prompt).strip()

        # Ignore blank entries
        if not user_input:
            continue

        # Allow the user to finish only after 3+ entries
        if user_input.lower() == "done":
            if len(interests) < 3:
                print(f"  [!] Please enter at least 3 interests ({len(interests)} so far).")
                continue
            break

        interests.append(user_input.lower())

    print()
    return " ".join(interests)


def recommend(user_text: str, top_n: int = 3) -> list[dict]:
    """
    Step 4: Build TF-IDF vectors and compute cosine similarity.

    How it works:
      1. Combine all role skill-strings with the user's input into one list.
      2. TfidfVectorizer converts each string into a numerical vector based on
         word frequency (TF) weighted by inverse document frequency (IDF).
      3. Cosine similarity measures how close the user's vector is to each
         role's vector (1.0 = identical direction, 0.0 = completely unrelated).
      4. We sort by similarity and return the top N results.
    """

    # Gather all the skill descriptions from the dataset
    role_descriptions = [role["skills"] for role in tech_roles]

    # The user's input is appended as the LAST document in the corpus
    corpus = role_descriptions + [user_text]

    # ── 4a: Fit the TF-IDF vectorizer on the entire corpus ───────────────
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ── 4b: Compute cosine similarity between the user vector (last row)
    #         and every role vector (all rows except the last) ─────────────
    user_vector = tfidf_matrix[-1]          # the user's TF-IDF vector
    role_vectors = tfidf_matrix[:-1]        # all role TF-IDF vectors

    similarities = cosine_similarity(user_vector, role_vectors).flatten()
    # similarities is now a 1-D array with one score per role

    # ── 4c: Pair each role with its similarity score and sort descending ──
    scored_roles = []
    for index, score in enumerate(similarities):
        scored_roles.append({
            "role": tech_roles[index]["role"],
            "skills": tech_roles[index]["skills"],
            "score": round(score, 4),       # round for clean display
        })

    scored_roles.sort(key=lambda x: x["score"], reverse=True)

    # Return only the top N recommendations
    return scored_roles[:top_n]


def display_results(recommendations: list[dict]) -> None:
    """
    Step 5: Display the top recommendations in a readable format.
    """
    print("=" * 60)
    print("       >>  TOP RECOMMENDATIONS  <<")
    print("=" * 60)

    for rank, rec in enumerate(recommendations, start=1):
        print(f"\n  #{rank}  {rec['role']}")
        print(f"       Similarity Score : {rec['score']}")
        print(f"       Related Skills   : {rec['skills']}")
        print("       " + "-" * 48)

    print()
    print("=" * 60)
    print("  Thank you for using the Tech Stack Recommender!")
    print("=" * 60)


# ── Step 6: Main entry point ─────────────────────────────────────────────────

def main():
    """Run the full recommendation pipeline."""

    # Collect user input
    user_text = get_user_interests()

    # Generate recommendations
    recommendations = recommend(user_text, top_n=3)

    # Display results
    display_results(recommendations)


if __name__ == "__main__":
    main()
