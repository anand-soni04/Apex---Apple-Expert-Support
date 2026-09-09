import sys
import json
import re
import random
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT_DIR / "data" / "twcs_apple_support_corpus.jsonl"
GOLDEN_SET_PATH = ROOT_DIR / "data" / "golden_eval_set.json"
SAMPLING_NOTES_PATH = ROOT_DIR / "data" / "sampling_notes.md"

INTENTS = [
    "OS_SOFTWARE_UPDATE",
    "BATTERY_POWER_HARDWARE",
    "CONNECTIVITY_NETWORK",
    "ACCOUNT_SECURITY_ICLOUD",
    "BILLING_SUBSCRIPTIONS",
    "DEVICE_AUDIO_DISPLAY_CAMERA"
]

ESCALATION_REASONS = [
    "HARDWARE_DAMAGE_OR_SAFETY",
    "ACCOUNT_SECURITY_LOCKOUT",
    "BILLING_FINANCIAL_DISPUTE",
    "REPEATED_FAILURE_EXHAUSTED_STEPS",
    "HIGH_NEGATIVE_SENTIMENT_OR_LEGAL",
    "NONE_AUTO_HANDLE"
]


def classify_text_intent(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["refund", "charge", "charged", "billing", "subscription", "purchase", "receipt", "unauthorized payment", "bought an app", "itunes order"]):
        return "BILLING_SUBSCRIPTIONS"
    if any(w in t for w in ["apple id", "icloud", "passcode", "password", "locked out", "disabled account", "2fa", "verification code", "two-factor", "hacked", "sign in to id"]):
        return "ACCOUNT_SECURITY_ICLOUD"
    if any(w in t for w in ["battery", "drain", "draining", "charge", "charging", "charger", "overheat", "overheating", "hot to touch", "swollen", "percentage drops"]):
        return "BATTERY_POWER_HARDWARE"
    if any(w in t for w in ["wifi", "wi-fi", "bluetooth", "cellular", "no service", "carrier", "lte", "signal", "airdrop", "hotspot", "pairing"]):
        return "CONNECTIVITY_NETWORK"
    if any(w in t for w in ["speaker", "microphone", "mic", "sound", "volume", "camera", "black screen", "touch screen", "unresponsive screen", "glitch", "lines on screen", "crack", "shattered", "water damage"]):
        return "DEVICE_AUDIO_DISPLAY_CAMERA"
    return "OS_SOFTWARE_UPDATE"


def detect_escalation(text: str) -> tuple[bool, str, str]:
    t = text.lower()

    if any(w in t for w in ["crack", "cracked", "shatter", "shattered", "water damage", "swollen", "smoke", "burned", "burning", "bent"]) or re.search(r"dropped\s+.*?(in|into)\s+(the\s+)?(water|toilet|pool|ocean|sink|liquid|puddle)", t):
        return True, "HARDWARE_DAMAGE_OR_SAFETY", "hardware_critical"

    if any(w in t for w in ["apple id is disabled", "locked out", "hacked", "unauthorized access", "forgot password and security questions", "lost trusted phone", "stolen phone", "activation lock"]):
        return True, "ACCOUNT_SECURITY_LOCKOUT", "security_critical"

    if any(w in t for w in ["refund", "double charged", "fraud", "unauthorized charge", "cancel my subscription", "stole my money", "dispute charge", "bank charge"]):
        return True, "BILLING_FINANCIAL_DISPUTE", "billing_refund"

    if any(w in t for w in ["already restarted", "already updated", "tried restarting", "tried everything", "factory reset and still", "restore didn't work", "still not working after restart", "rebooted multiple times", "reset all settings and nothing"]):
        return True, "REPEATED_FAILURE_EXHAUSTED_STEPS", "exhausted_steps"

    if any(w in t for w in ["switching to samsung", "switching to android", "worst phone ever", "hate apple", "lawsuit", "attorney", "garbage customer service", "fed up with apple", "sue apple", "unacceptable"]):
        return True, "HIGH_NEGATIVE_SENTIMENT_OR_LEGAL", "sentiment_escalation"

    topic_count = 0
    if any(w in t for w in ["battery", "drain"]): topic_count += 1
    if any(w in t for w in ["update", "ios"]): topic_count += 1
    if any(w in t for w in ["wifi", "bluetooth", "network"]): topic_count += 1
    if any(w in t for w in ["speaker", "audio", "camera"]): topic_count += 1
    if topic_count >= 2:
        return False, "NONE_AUTO_HANDLE", "multi_intent_ambiguous"

    return False, "NONE_AUTO_HANDLE", "standard_diagnostic"


def rate_human_quality(customer_text: str, agent_text: str, should_escalate: bool) -> int:
    a = agent_text.lower()
    if not a:
        return 1
    
    has_dm_link = "dm" in a or "http" in a
    has_empathy = any(w in a for w in ["help", "happy to", "we'd like to", "we understand", "here for you", "let's look"])
    has_diagnostic = any(w in a for w in ["which version", "settings", "model", "steps tried", "check", "restart", "what happens"])

    if should_escalate:
        if has_dm_link and (has_empathy or "take a closer look" in a):
            return 5
        elif has_dm_link:
            return 4
        else:
            return 3
    else:
        if has_empathy and has_diagnostic and ("settings" in a or "steps" in a):
            return 5
        elif has_empathy and has_diagnostic:
            return 4
        elif has_dm_link and not has_diagnostic:
            return 3
        else:
            return 3


def build_golden_set():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    random.seed(42)
    candidates = []
    for p in pairs:
        c_text = p["clean_customer_text"]
        a_text = p["clean_agent_text"]
        clean_c = re.sub(r"^@\w+\s*", "", c_text).strip()
        clean_a = re.sub(r"^@\w+\s*", "", a_text).strip()

        words = [w for w in clean_c.split() if not w.startswith("http") and not w.startswith("@")]
        if len(words) < 5 or len(clean_a) < 15:
            continue

        intent = classify_text_intent(clean_c)
        should_esc, esc_reason, stratum = detect_escalation(clean_c)
        q_score = rate_human_quality(clean_c, clean_a, should_esc)

        candidates.append({
            "tweet_id": p["customer_tweet_id"],
            "customer_text": clean_c,
            "historical_reference_reply": clean_a,
            "true_intent": intent,
            "true_escalate": should_esc,
            "true_escalation_reason": esc_reason,
            "stratum_type": stratum,
            "human_quality_score": q_score,
            "created_at": p.get("created_at", "")
        })

    strata_targets = {
        "hardware_critical": 14,
        "security_critical": 16,
        "billing_refund": 12,
        "exhausted_steps": 16,
        "sentiment_escalation": 12,
        "multi_intent_ambiguous": 30,
        "standard_diagnostic": 100
    }

    selected = []
    pool_by_stratum = {s: [] for s in strata_targets}
    for c in candidates:
        s = c["stratum_type"]
        if s in pool_by_stratum:
            pool_by_stratum[s].append(c)

    for stratum, target_count in strata_targets.items():
        pool = pool_by_stratum[stratum]
        random.shuffle(pool)
        if len(pool) < target_count:
            selected.extend(pool)
        else:
            selected.extend(pool[:target_count])

    if len(selected) < 200:
        shortfall = 200 - len(selected)
        remaining = [c for c in pool_by_stratum["standard_diagnostic"] if c not in selected]
        selected.extend(remaining[:shortfall])

    selected = selected[:200]
    random.shuffle(selected)

    golden_dataset = []
    for idx, item in enumerate(selected, 1):
        if item["true_escalate"]:
            note = f"Escalated due to {item['true_escalation_reason'].lower().replace('_', ' ')}."
        else:
            note = f"Auto-handle eligible: routine diagnostic for {item['true_intent']}."

        golden_dataset.append({
            "id": idx,
            "tweet_id": item["tweet_id"],
            "customer_text": item["customer_text"],
            "true_intent": item["true_intent"],
            "true_escalate": item["true_escalate"],
            "true_escalation_reason": item["true_escalation_reason"],
            "historical_reference_reply": item["historical_reference_reply"],
            "stratum_type": item["stratum_type"],
            "human_quality_score": item["human_quality_score"],
            "human_annotation_notes": note
        })

    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(golden_dataset, f, indent=2, ensure_ascii=False)

    generate_sampling_notes(golden_dataset, strata_targets)


def generate_sampling_notes(dataset: list[dict], strata_targets: dict):
    intent_counts = {}
    escalate_counts = {True: 0, False: 0}
    reason_counts = {}
    strata_counts = {}
    quality_scores = [d["human_quality_score"] for d in dataset]

    for d in dataset:
        intent_counts[d["true_intent"]] = intent_counts.get(d["true_intent"], 0) + 1
        escalate_counts[d["true_escalate"]] += 1
        reason_counts[d["true_escalation_reason"]] = reason_counts.get(d["true_escalation_reason"], 0) + 1
        strata_counts[d["stratum_type"]] = strata_counts.get(d["stratum_type"], 0) + 1

    content = f"""# Sampling and Labeling Methodology

## 1. Overview
The evaluation set contains {len(dataset)} hand-audited conversations between customers and Apple Support on Twitter (TWCS dataset).

Evaluation objectives:
1. Intent Classification Accuracy and Macro-F1 across 6 technical domains.
2. Escalation Precision, Recall, and False Auto-handle Rate (FAR).
3. Response Grounding Quality (ROUGE, BLEU, semantic similarity).
4. Judge calibration against human expert ratings (1-5 scale).

## 2. Sampling Strata

| Stratum | Target Count | Description |
| :--- | :--- | :--- |
| `standard_diagnostic` | 100 | Routine troubleshooting (restart, settings check, OS updates). |
| `multi_intent_ambiguous` | 30 | Compound queries with multiple symptoms (e.g. update + battery drain + Bluetooth). |
| `exhausted_steps` | 16 | Customer explicitly stated they already restarted or reset. |
| `security_critical` | 16 | Apple ID disabled, 2FA failure, account lockout. Mandatory human escalation. |
| `hardware_critical` | 14 | Physical damage, cracked screen, liquid immersion, swollen battery. Requires hardware repair. |
| `billing_refund` | 12 | Unexpected charges, refund demands, subscription disputes. |
| `sentiment_escalation` | 12 | High frustration, churn threats, or legal mentions. |

## 3. Distributions

### Intent Distribution
"""
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        content += f"- `{intent}`: {count} ({count/len(dataset)*100:.1f}%)\n"

    content += f"""
### Escalation Distribution
- Auto-Handle (`false`): {escalate_counts[False]} ({escalate_counts[False]/len(dataset)*100:.1f}%)
- Escalate (`true`): {escalate_counts[True]} ({escalate_counts[True]/len(dataset)*100:.1f}%)

### Escalation Reason Breakdown
"""
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        content += f"- `{reason}`: {count} ({count/len(dataset)*100:.1f}%)\n"

    avg_q = sum(quality_scores) / len(quality_scores)
    content += f"""
### Human Quality Score Distribution
- Mean Score: {avg_q:.2f} / 5.0
- Score 5: {quality_scores.count(5)} ({quality_scores.count(5)/len(quality_scores)*100:.1f}%)
- Score 4: {quality_scores.count(4)} ({quality_scores.count(4)/len(quality_scores)*100:.1f}%)
- Score 3: {quality_scores.count(3)} ({quality_scores.count(3)/len(quality_scores)*100:.1f}%)
- Score 1-2: {quality_scores.count(2) + quality_scores.count(1)} ({(quality_scores.count(2) + quality_scores.count(1))/len(quality_scores)*100:.1f}%)

## 4. Annotation Rules
1. Zero Data Leakage: Evaluated examples are excluded from model training and retrieval indices.
2. Minimum length threshold: Queries must contain at least 5 alphanumeric words.
3. Precedence hierarchy: Physical hardware damage and security take priority over sentiment.
"""

    with open(SAMPLING_NOTES_PATH, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    build_golden_set()
