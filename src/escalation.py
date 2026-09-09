import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ESCALATION_REASONS = {
    "HARDWARE_DAMAGE_OR_SAFETY": "Physical hardware defect, battery swelling, or liquid damage requires physical Genius Bar/mail-in repair.",
    "ACCOUNT_SECURITY_LOCKOUT": "Apple ID lockout, security credential dispute, or 2FA verification failure requires secure human identity validation.",
    "BILLING_FINANCIAL_DISPUTE": "Monetary dispute, unauthorized charge, or refund request requires human agent billing authority.",
    "REPEATED_FAILURE_EXHAUSTED_STEPS": "Customer already attempted standard diagnostic steps (restart, update, reset); repeating basic steps degrades CSAT.",
    "HIGH_NEGATIVE_SENTIMENT_OR_LEGAL": "Severe customer distress, legal mentions, or high churn risk requires specialized human de-escalation.",
    "AMBIGUOUS_QUERY_LOW_CONFIDENCE": "Inquiry is ambiguous or classifier confidence is below safety threshold; safe escalation required to prevent hallucinated advice.",
    "NONE_AUTO_HANDLE": "Query is eligible for automated troubleshooting, initial diagnostics, and standard knowledge base grounding."
}


class EscalationPolicyEngine:
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def evaluate(self, customer_text: str, intent: Optional[str] = None, confidence: float = 1.0) -> Dict[str, Any]:
        t = customer_text.lower()

        hardware_pats = [
            r"\bcrack\b", r"\bcracked\b", r"\bshatter\b", r"\bshattered\b",
            r"dropped\s+.*?(in|into)\s+(the\s+)?(water|toilet|pool|ocean|sink|liquid|puddle)",
            r"fell\s+.*?(in|into)\s+(the\s+)?(water|toilet|pool|ocean|sink|liquid|puddle)",
            r"(water|liquid)\s+damag", r"\bswollen\b", r"expanding\s+battery",
            r"\bsmoke\b", r"\bsmoking\b", r"\bburned\b", r"\bburning\b",
            r"bent\s+(phone|iphone|ipad)", r"broken\s+(screen|glass|display|button|port|hardware)"
        ]
        if any(re.search(pat, t) for pat in hardware_pats):
            return {
                "should_escalate": True,
                "escalation_reason": "HARDWARE_DAMAGE_OR_SAFETY",
                "stated_explanation": ESCALATION_REASONS["HARDWARE_DAMAGE_OR_SAFETY"]
            }

        security_pats = [
            r"apple id (is )?(disabled|locked)", r"locked out of (my )?(account|iphone|ipad|apple id)",
            r"\bhacked\b", r"unauthorized access", r"forgot (my )?password and (security questions|email)",
            r"lost (my )?trusted (phone|device)", r"stolen (iphone|device|phone)", r"activation lock"
        ]
        if any(re.search(pat, t) for pat in security_pats):
            return {
                "should_escalate": True,
                "escalation_reason": "ACCOUNT_SECURITY_LOCKOUT",
                "stated_explanation": ESCALATION_REASONS["ACCOUNT_SECURITY_LOCKOUT"]
            }

        billing_pats = [
            r"\brefund\b", r"double charged", r"fraud", r"unauthorized charge",
            r"cancel (my )?subscription", r"stole (my )?money", r"dispute charge",
            r"bank charge", r"charged (me )?without permission"
        ]
        if any(re.search(pat, t) for pat in billing_pats):
            return {
                "should_escalate": True,
                "escalation_reason": "BILLING_FINANCIAL_DISPUTE",
                "stated_explanation": ESCALATION_REASONS["BILLING_FINANCIAL_DISPUTE"]
            }

        exhausted_pats = [
            r"already (restarted|rebooted|updated|restored)",
            r"tried (restarting|updating|rebooting|restoring|everything)",
            r"factory reset and still", r"still not working after",
            r"rebooted multiple times", r"reset all settings and nothing",
            r"tried all (the )?(steps|troubleshooting)"
        ]
        if any(re.search(pat, t) for pat in exhausted_pats):
            return {
                "should_escalate": True,
                "escalation_reason": "REPEATED_FAILURE_EXHAUSTED_STEPS",
                "stated_explanation": ESCALATION_REASONS["REPEATED_FAILURE_EXHAUSTED_STEPS"]
            }

        sentiment_pats = [
            r"(switching|switch|buying|buy|getting|get)\s+to\s+(samsung|android|pixel)",
            r"buying\s+a\s+(samsung|android|pixel|galaxy)",
            r"worst\s+(phone|device|company|customer service|product)\s+(i('ve| have)?\s+)?ever",
            r"hate\s+apple", r"\blawsuit\b", r"\battorney\b", r"\bsue\b\s+apple",
            r"garbage\s+customer\s+service", r"fed\s+up\s+with\s+apple", r"unacceptable\s+service"
        ]
        if any(re.search(pat, t) for pat in sentiment_pats):
            return {
                "should_escalate": True,
                "escalation_reason": "HIGH_NEGATIVE_SENTIMENT_OR_LEGAL",
                "stated_explanation": ESCALATION_REASONS["HIGH_NEGATIVE_SENTIMENT_OR_LEGAL"]
            }

        if confidence < self.confidence_threshold:
            return {
                "should_escalate": True,
                "escalation_reason": "AMBIGUOUS_QUERY_LOW_CONFIDENCE",
                "stated_explanation": ESCALATION_REASONS["AMBIGUOUS_QUERY_LOW_CONFIDENCE"]
            }

        return {
            "should_escalate": False,
            "escalation_reason": "NONE_AUTO_HANDLE",
            "stated_explanation": ESCALATION_REASONS["NONE_AUTO_HANDLE"]
        }


if __name__ == "__main__":
    engine = EscalationPolicyEngine()
    test_cases = [
        ("My screen is completely shattered after I dropped it", 0.95),
        ("My Apple ID is disabled and I can't receive 2FA codes", 0.98),
        ("I was double charged $19.99 for Apple Music and want a refund", 0.99),
        ("I already restarted my phone and reset network settings, still no WiFi", 0.90),
        ("This is ridiculous, worst support ever, I'm switching to Samsung!", 0.85),
        ("How do I check my battery health in settings?", 0.92),
        ("Some random text with obscure symbols #$*@", 0.20)
    ]
    for text, conf in test_cases:
        res = engine.evaluate(text, confidence=conf)
        print(f"Text: '{text}'")
        print(f"  -> Escalate: {res['should_escalate']} | Reason: {res['escalation_reason']}")
