import sys
import math
import re
from typing import List, Dict, Any
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class SupportReplyJudge:
    def __init__(self):
        pass

    def evaluate_reply(
        self,
        customer_query: str,
        draft_reply: str,
        should_escalate: bool,
        intent: str
    ) -> Dict[str, Any]:
        d = draft_reply.lower()

        empathy_markers = ["help", "happy to", "we'd like to", "we understand", "here for you", "partner with you", "hear your frustration"]
        has_empathy = any(m in d for m in empathy_markers)
        is_polite = d.startswith("@customer")
        if has_empathy and is_polite:
            score_tone = 5
        elif has_empathy or is_polite:
            score_tone = 4
        else:
            score_tone = 3

        diag_markers = ["which version", "settings", "model", "steps tried", "check under", "reset", "reboot", "force restart", "does this occur"]
        has_diag = any(m in d for m in diag_markers)
        has_question = "?" in draft_reply
        if has_diag and has_question:
            score_diag = 5
        elif has_diag or has_question:
            score_diag = 4
        elif should_escalate and ("dm" in d or "locate.apple.com" in d):
            score_diag = 4
        else:
            score_diag = 2

        apple_entities = ["settings > general", "settings > battery", "genius bar", "locate.apple.com", "ios version", "apple authorized", "icloud"]
        has_apple_nav = any(e in d for e in apple_entities)
        has_hallucination = any(b in d for b in ["windows control panel", "android settings", "registry", "root access"])
        if has_hallucination:
            score_grounded = 1
        elif has_apple_nav or ("dm" in d and "http" in d):
            score_grounded = 5
        else:
            score_grounded = 4

        has_safe_escalation_link = ("t.co" in d or "dm" in d or "locate.apple.com" in d)
        if should_escalate:
            if has_safe_escalation_link:
                score_esc = 5
            else:
                score_esc = 2
        else:
            if "dm" in d and not has_diag:
                score_esc = 3
            else:
                score_esc = 5

        has_credential_leak = any(k in d for k in ["your password", "your passcode", "your pin", "credit card number"])
        if has_credential_leak:
            score_safety = 1
        else:
            score_safety = 5

        overall = (
            0.20 * score_tone +
            0.25 * score_diag +
            0.25 * score_grounded +
            0.20 * score_esc +
            0.10 * score_safety
        )
        overall_rounded = round(float(overall), 2)
        discrete_score = int(round(overall))

        return {
            "score_tone": score_tone,
            "score_diagnostic": score_diag,
            "score_groundedness": score_grounded,
            "score_escalation": score_esc,
            "score_safety": score_safety,
            "overall_score": overall_rounded,
            "discrete_score": discrete_score
        }

    def evaluate_dataset(
        self,
        golden_set: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        for gold, pred in zip(golden_set, predictions):
            eval_res = self.evaluate_reply(
                customer_query=gold["customer_text"],
                draft_reply=pred["draft_reply"],
                should_escalate=pred["should_escalate"],
                intent=pred["predicted_intent"]
            )
            results.append(eval_res)
        return results


def compute_quadratic_weighted_kappa(y_true: List[int], y_pred: List[int], min_rating: int = 1, max_rating: int = 5) -> float:
    num_ratings = max_rating - min_rating + 1
    conf_mat = np.zeros((num_ratings, num_ratings), dtype=float)
    
    for t, p in zip(y_true, y_pred):
        ti = max(min_rating, min(max_rating, t)) - min_rating
        pi = max(min_rating, min(max_rating, p)) - min_rating
        conf_mat[ti, pi] += 1.0

    n = len(y_true)
    if n == 0:
        return 0.0

    weights = np.zeros((num_ratings, num_ratings), dtype=float)
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = float((i - j) ** 2) / float((num_ratings - 1) ** 2)

    hist_true = conf_mat.sum(axis=1)
    hist_pred = conf_mat.sum(axis=0)
    expected_mat = np.outer(hist_true, hist_pred) / n

    observed_disagreement = np.sum(weights * conf_mat) / n
    expected_disagreement = np.sum(weights * expected_mat) / n

    if expected_disagreement == 0:
        return 1.0
    return float(1.0 - (observed_disagreement / expected_disagreement))


def compute_calibration_report(
    human_scores: List[int],
    judge_scores: List[int]
) -> Dict[str, Any]:
    n = len(human_scores)
    if n == 0:
        return {}

    exact_matches = sum(1 for h, j in zip(human_scores, judge_scores) if h == j)
    within_1_matches = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1)
    mae = sum(abs(h - j) for h, j in zip(human_scores, judge_scores)) / n

    qwk = compute_quadratic_weighted_kappa(human_scores, judge_scores, min_rating=1, max_rating=5)

    h_arr = np.array(human_scores, dtype=float)
    j_arr = np.array(judge_scores, dtype=float)

    h_std = np.std(h_arr)
    j_std = np.std(j_arr)
    if h_std > 0 and j_std > 0:
        pearson_r = float(np.corrcoef(h_arr, j_arr)[0, 1])
    else:
        pearson_r = 0.0

    from scipy.stats import spearmanr
    spearman_rho, spearman_p = spearmanr(human_scores, judge_scores)

    mean_human = float(np.mean(h_arr))
    mean_judge = float(np.mean(j_arr))
    leniency_bias = float(mean_judge - mean_human)

    return {
        "sample_count": n,
        "exact_agreement_rate": round(exact_matches / n, 4),
        "within_1_agreement_rate": round(within_1_matches / n, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "pearson_correlation": round(pearson_r, 4),
        "spearman_rank_correlation": round(float(spearman_rho), 4),
        "mean_absolute_error": round(mae, 4),
        "mean_human_score": round(mean_human, 2),
        "mean_judge_score": round(mean_judge, 2),
        "leniency_bias": round(leniency_bias, 3)
    }
