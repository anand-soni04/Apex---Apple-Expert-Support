import sys
import math
import re
from collections import Counter
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])


def compute_rouge_n(cand_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    ref_count = sum(ref_ngrams.values())
    cand_count = sum(cand_ngrams.values())

    if ref_count == 0 or cand_count == 0:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}

    overlap = sum((cand_ngrams & ref_ngrams).values())
    precision = overlap / cand_count
    recall = overlap / ref_count
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"p": precision, "r": recall, "f1": f1}


def compute_rouge_l(cand_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    m, n = len(cand_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}

    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if cand_tokens[i] == ref_tokens[j]:
                lcs[i + 1][j + 1] = lcs[i][j] + 1
            else:
                lcs[i + 1][j + 1] = max(lcs[i + 1][j], lcs[i][j + 1])

    lcs_len = lcs[m][n]
    precision = lcs_len / m
    recall = lcs_len / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"p": precision, "r": recall, "f1": f1}


def compute_bleu_4(cand_tokens: List[str], ref_tokens: List[str]) -> float:
    c_len = len(cand_tokens)
    r_len = len(ref_tokens)
    if c_len == 0 or r_len == 0:
        return 0.0

    bp = math.exp(1 - r_len / c_len) if c_len < r_len else 1.0

    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []

    for n in range(1, 5):
        cand_ng = get_ngrams(cand_tokens, n)
        ref_ng = get_ngrams(ref_tokens, n)
        c_count = sum(cand_ng.values())
        if c_count == 0:
            precisions.append(0.0)
            continue
        overlap = sum((cand_ng & ref_ng).values())
        precisions.append((overlap + 0.1) / (c_count + 0.1))

    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
    return float(bp * math.exp(log_sum))


def check_policy_compliance(draft: str, should_escalate: bool) -> bool:
    t = draft.lower()
    if any(w in t for w in ["your password", "your passcode", "your pin", "credit card"]):
        return False
    if should_escalate and not any(k in t for k in ["dm", "genius bar", "locate.apple.com", "appointment", "advisor"]):
        return False
    if len(draft) > 300:
        return False
    return True


class EvaluationHarness:
    def __init__(self, golden_set: List[Dict[str, Any]]):
        self.golden_set = golden_set

    def evaluate_system(self, system_name: str, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true_intent = [item["true_intent"] for item in self.golden_set]
        y_pred_intent = [pred["predicted_intent"] for pred in predictions]

        y_true_esc = [item["true_escalate"] for item in self.golden_set]
        y_pred_esc = [pred["should_escalate"] for pred in predictions]

        ref_texts = [item["historical_reference_reply"] for item in self.golden_set]
        draft_texts = [pred["draft_reply"] for pred in predictions]

        acc = accuracy_score(y_true_intent, y_pred_intent)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true_intent, y_pred_intent, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true_intent, y_pred_intent, average="weighted", zero_division=0)

        tp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is True and yp is True)
        fp = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is False and yp is True)
        fn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is True and yp is False)
        tn = sum(1 for yt, yp in zip(y_true_esc, y_pred_esc) if yt is False and yp is False)

        esc_acc = (tp + tn) / len(y_true_esc) if y_true_esc else 0.0
        esc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        esc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        esc_f1 = (2 * esc_prec * esc_rec / (esc_prec + esc_rec)) if (esc_prec + esc_rec) > 0 else 0.0
        
        total_true_escalations = (tp + fn)
        false_auto_handle_rate = (fn / total_true_escalations) if total_true_escalations > 0 else 0.0

        r1_list, r2_list, rl_list, bleu_list = [], [], [], []
        policy_compliant_count = 0

        for draft, ref, yt_esc in zip(draft_texts, ref_texts, y_true_esc):
            cand_tok = tokenize_text(draft)
            ref_tok = tokenize_text(ref)

            r1_list.append(compute_rouge_n(cand_tok, ref_tok, 1)["f1"])
            r2_list.append(compute_rouge_n(cand_tok, ref_tok, 2)["f1"])
            rl_list.append(compute_rouge_l(cand_tok, ref_tok)["f1"])
            bleu_list.append(compute_bleu_4(cand_tok, ref_tok))

            if check_policy_compliance(draft, yt_esc):
                policy_compliant_count += 1

        vectorizer = TfidfVectorizer().fit(draft_texts + ref_texts)
        d_vecs = vectorizer.transform(draft_texts)
        r_vecs = vectorizer.transform(ref_texts)
        cos_sims = [cosine_similarity(d_vecs[i], r_vecs[i])[0][0] for i in range(len(draft_texts))]

        return {
            "system_name": system_name,
            "sample_count": len(self.golden_set),
            "intent_accuracy": round(float(acc), 4),
            "intent_macro_f1": round(float(f1_m), 4),
            "intent_weighted_f1": round(float(f1_w), 4),
            "escalation_accuracy": round(float(esc_acc), 4),
            "escalation_precision": round(float(esc_prec), 4),
            "escalation_recall": round(float(esc_rec), 4),
            "escalation_f1": round(float(esc_f1), 4),
            "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
            "rouge_1": round(float(sum(r1_list) / len(r1_list)), 4),
            "rouge_2": round(float(sum(r2_list) / len(r2_list)), 4),
            "rouge_l": round(float(sum(rl_list) / len(rl_list)), 4),
            "bleu_4": round(float(sum(bleu_list) / len(bleu_list)), 4),
            "semantic_similarity": round(float(sum(cos_sims) / len(cos_sims)), 4),
            "policy_compliance_rate": round(float(policy_compliant_count / len(self.golden_set)), 4),
            "confusion_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        }
