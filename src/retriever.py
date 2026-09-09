import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def clean_query_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class HistoricalResolutionRetriever:
    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        golden_set_path: Optional[Path] = None,
        max_index_size: int = 5000
    ):
        self.corpus_path = corpus_path or (Path(ROOT_DIR) / "data" / "twcs_apple_support_corpus.jsonl")
        self.golden_set_path = golden_set_path or (Path(ROOT_DIR) / "data" / "golden_eval_set.json")
        self.max_index_size = max_index_size
        self.vectorizer = None
        self.corpus_matrix = None
        self.indexed_items = []
        self._is_indexed = False

    def build_index(self):
        excluded_ids = set()
        if self.golden_set_path.exists():
            with open(self.golden_set_path, "r", encoding="utf-8") as f:
                golden_data = json.load(f)
                excluded_ids = {str(item["tweet_id"]) for item in golden_data}

        documents = []
        self.indexed_items = []

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                t_id = str(item.get("customer_tweet_id", ""))
                if t_id in excluded_ids:
                    continue

                c_clean = clean_query_text(item.get("clean_customer_text", ""))
                a_clean = item.get("clean_agent_text", "").strip()

                if len(c_clean.split()) < 3 or len(a_clean) < 15:
                    continue

                documents.append(c_clean)
                self.indexed_items.append({
                    "customer_tweet_id": t_id,
                    "customer_query": item.get("clean_customer_text"),
                    "historical_reply": a_clean,
                    "created_at": item.get("created_at", "")
                })

                if len(self.indexed_items) >= self.max_index_size:
                    break

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        self.corpus_matrix = self.vectorizer.fit_transform(documents)
        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._is_indexed:
            self.build_index()

        q_clean = clean_query_text(query)
        if not q_clean:
            return []

        q_vec = self.vectorizer.transform([q_clean])
        similarities = cosine_similarity(q_vec, self.corpus_matrix)[0]

        top_indices = similarities.argsort()[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.05:
                match = self.indexed_items[idx]
                results.append({
                    "customer_tweet_id": match["customer_tweet_id"],
                    "historical_customer_query": match["customer_query"],
                    "historical_apple_reply": match["historical_reply"],
                    "similarity_score": round(score, 4)
                })

        return results


if __name__ == "__main__":
    retriever = HistoricalResolutionRetriever()
    retriever.build_index()
    test_q = "My iPhone battery is draining so fast after updating to iOS 11"
    hits = retriever.retrieve(test_q, top_k=3)
    print(f"Query: {test_q}")
    for i, h in enumerate(hits, 1):
        print(f"\nMatch {i} (Similarity: {h['similarity_score']}):")
        print(f"  Customer: {h['historical_customer_query']}")
        print(f"  Apple:    {h['historical_apple_reply']}")
