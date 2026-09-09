import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.retriever import HistoricalResolutionRetriever

TRIVIAL_STATIC_TEMPLATE = (
    "Thanks for reaching out! We'd like to help you with your device. "
    "Please send us a DM with your exact device model and iOS version: https://t.co/GDrqU22YpT"
)


class TrivialBaseline:
    def __init__(self):
        self.majority_intent = "OS_SOFTWARE_UPDATE"

    def predict(self, customer_text: str) -> Dict[str, Any]:
        return {
            "predicted_intent": self.majority_intent,
            "confidence": 0.54,
            "should_escalate": False,
            "escalation_reason": "NONE_AUTO_HANDLE",
            "stated_explanation": "Trivial baseline default auto-handle.",
            "draft_reply": TRIVIAL_STATIC_TEMPLATE,
            "grounding_exemplars": []
        }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]


class SimpleBaseline:
    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        golden_set_path: Optional[Path] = None
    ):
        self.corpus_path = corpus_path or (Path(ROOT_DIR) / "data" / "twcs_apple_support_corpus.jsonl")
        self.golden_set_path = golden_set_path or (Path(ROOT_DIR) / "data" / "golden_eval_set.json")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=500)
        self.retriever = HistoricalResolutionRetriever(self.corpus_path, self.golden_set_path)
        self._is_trained = False

    def train(self):
        from data.build_golden_set import classify_text_intent

        excluded_ids = set()
        if self.golden_set_path.exists():
            with open(self.golden_set_path, "r", encoding="utf-8") as f:
                golden_data = json.load(f)
                excluded_ids = {str(item["tweet_id"]) for item in golden_data}

        texts = []
        labels = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if str(item.get("customer_tweet_id")) in excluded_ids:
                    continue
                c_clean = item.get("clean_customer_text", "").lower().strip()
                if len(c_clean.split()) >= 3:
                    texts.append(c_clean)
                    labels.append(classify_text_intent(c_clean))

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.retriever.build_index()
        self._is_trained = True

    def predict(self, customer_text: str) -> Dict[str, Any]:
        if not self._is_trained:
            self.train()

        t_lower = customer_text.lower()
        X = self.vectorizer.transform([t_lower])
        pred_intent = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        confidence = float(max(probs))

        naive_escalation_keywords = ["broken", "refund", "locked", "sue"]
        should_escalate = any(k in t_lower for k in naive_escalation_keywords)
        reason = "NAIVE_KEYWORD_TRIGGER" if should_escalate else "NONE_AUTO_HANDLE"
        explanation = "Triggered by naive keyword match in text." if should_escalate else "No naive keyword detected."

        hits = self.retriever.retrieve(customer_text, top_k=1)
        if hits:
            draft_reply = hits[0]["historical_apple_reply"]
            exemplars = [hits[0]]
        else:
            draft_reply = TRIVIAL_STATIC_TEMPLATE
            exemplars = []

        return {
            "predicted_intent": pred_intent,
            "confidence": round(confidence, 4),
            "should_escalate": should_escalate,
            "escalation_reason": reason,
            "stated_explanation": explanation,
            "draft_reply": draft_reply,
            "grounding_exemplars": exemplars
        }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self._is_trained:
            self.train()
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    t_base = TrivialBaseline()
    s_base = SimpleBaseline()
    s_base.train()

    test_msg = "My battery drops to 20% in an hour after updating to iOS 11"
    print(t_base.predict(test_msg))
    print(s_base.predict(test_msg))
