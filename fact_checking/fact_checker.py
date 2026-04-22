import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FactChecker:
    def __init__(self, path="facts.json"):
        with open(path, "r") as f:
            self.data = json.load(f)

        self.claims = [item["claim"] for item in self.data]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.claim_vectors = self.vectorizer.fit_transform(self.claims)

    def check_fact(self, user_input):
        user_vec = self.vectorizer.transform([user_input])

        similarities = cosine_similarity(user_vec, self.claim_vectors)
        best_idx = similarities.argmax()
        score = similarities[0][best_idx]

        best_match = self.data[best_idx]

        return {
            "match": best_match["claim"],
            "label": best_match["label"],
            "source": best_match["source"],
            "similarity": float(score)
        }