import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

INTENTS = [
    "OS_SOFTWARE_UPDATE",
    "BATTERY_POWER_HARDWARE",
    "CONNECTIVITY_NETWORK",
    "ACCOUNT_SECURITY_ICLOUD",
    "BILLING_SUBSCRIPTIONS",
    "DEVICE_AUDIO_DISPLAY_CAMERA"
]

DEFAULT_FALLBACK_INTENT = "OS_SOFTWARE_UPDATE"


def clean_text_for_classification(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class IntentClassifier:
    def __init__(self, corpus_path: Optional[Path] = None, golden_set_path: Optional[Path] = None):
        self.corpus_path = corpus_path or (Path(__file__).resolve().parent.parent / "data" / "twcs_apple_support_corpus.jsonl")
        self.golden_set_path = golden_set_path or (Path(__file__).resolve().parent.parent / "data" / "golden_eval_set.json")
        self.classes_ = INTENTS
        self.model = None
        self.vectorizer = None
        self._is_trained = False

    def _train_from_corpus(self):
        excluded_ids = set()
        if self.golden_set_path.exists():
            with open(self.golden_set_path, "r", encoding="utf-8") as f:
                golden_data = json.load(f)
                excluded_ids = {str(item["tweet_id"]) for item in golden_data}

        from data.build_golden_set import classify_text_intent

        texts = []
        labels = []

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                t_id = str(item.get("customer_tweet_id", ""))
                if t_id in excluded_ids:
                    continue
                
                raw_c = item.get("clean_customer_text", "")
                c_clean = clean_text_for_classification(raw_c)
                if len(c_clean.split()) < 3:
                    continue

                label = classify_text_intent(c_clean)
                texts.append(c_clean)
                labels.append(label)

        self.vectorizer = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=12000)),
            ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=18000))
        ])

        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(
            C=2.5,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        self.model.fit(X, labels)
        self._is_trained = True

    def predict(self, text: str) -> Dict[str, Any]:
        if not self._is_trained:
            self._train_from_corpus()

        cleaned = clean_text_for_classification(text)
        if not cleaned:
            return {
                "predicted_intent": DEFAULT_FALLBACK_INTENT,
                "confidence": 0.0,
                "probabilities": {intent: 1.0 / len(INTENTS) for intent in INTENTS}
            }

        X = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(X)[0]
        prob_dict = {cls: float(prob) for cls, prob in zip(self.model.classes_, probs)}

        top_intent = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top_intent]

        return {
            "predicted_intent": top_intent,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)}
        }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self._is_trained:
            self._train_from_corpus()
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    clf = IntentClassifier()
    clf._train_from_corpus()
    test_queries = [
        "My iPhone 8 battery drops from 100% to 20% in 1 hour after iOS 11 update",
        "Wi-Fi disconnects constantly every 5 minutes on my iPad",
        "Someone charged my card $49.99 for an iTunes subscription I never ordered",
        "My Apple ID is disabled and 2FA code is not coming through",
        "The speaker on my iPhone 7 has constant crackling noise during phone calls"
    ]
    for q in test_queries:
        res = clf.predict(q)
        print(f"Query: {q}")
        print(f"  -> Predicted: {res['predicted_intent']} (confidence: {res['confidence']:.2f})")
