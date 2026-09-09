import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.classifier import IntentClassifier
from src.retriever import HistoricalResolutionRetriever
from src.escalation import EscalationPolicyEngine

APPLE_SECURE_DM_LINK = "https://t.co/GDrqU22YpT"
APPLE_GENIUS_BAR_LINK = "https://locate.apple.com"


class AppleSupportAgent:
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[HistoricalResolutionRetriever] = None,
        escalation_engine: Optional[EscalationPolicyEngine] = None,
        llm_provider: Optional[str] = None
    ):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or HistoricalResolutionRetriever()
        self.escalation_engine = escalation_engine or EscalationPolicyEngine()
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "local")

    def initialize(self):
        if not self.classifier._is_trained:
            self.classifier._train_from_corpus()
        if not self.retriever._is_indexed:
            self.retriever.build_index()

    def process(self, customer_text: str) -> Dict[str, Any]:
        clf_result = self.classifier.predict(customer_text)
        intent = clf_result["predicted_intent"]
        confidence = clf_result["confidence"]

        esc_result = self.escalation_engine.evaluate(
            customer_text=customer_text,
            intent=intent,
            confidence=confidence
        )
        should_escalate = esc_result["should_escalate"]
        escalation_reason = esc_result["escalation_reason"]
        stated_explanation = esc_result["stated_explanation"]

        exemplars = self.retriever.retrieve(customer_text, top_k=3)

        reply = self._synthesize_reply(
            customer_text=customer_text,
            intent=intent,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
            exemplars=exemplars
        )

        reply = self._apply_guardrails(reply, should_escalate)

        return {
            "predicted_intent": intent,
            "confidence": confidence,
            "probabilities": clf_result["probabilities"],
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
            "stated_explanation": stated_explanation,
            "draft_reply": reply,
            "grounding_exemplars": exemplars
        }

    def _synthesize_reply(
        self,
        customer_text: str,
        intent: str,
        should_escalate: bool,
        escalation_reason: str,
        exemplars: List[Dict[str, Any]]
    ) -> str:
        if self.llm_provider in ["openai", "gemini", "anthropic"] and (os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")):
            return self._call_llm_api(customer_text, intent, should_escalate, escalation_reason, exemplars)

        if should_escalate:
            if escalation_reason == "HARDWARE_DAMAGE_OR_SAFETY":
                return (
                    f"@customer Physical hardware damage requires hands-on inspection. "
                    f"Please visit an Apple Authorized Service Provider or book a Genius Bar reservation here: "
                    f"{APPLE_GENIUS_BAR_LINK}. You can also DM us to explore mail-in repair options: {APPLE_SECURE_DM_LINK}"
                )
            elif escalation_reason == "ACCOUNT_SECURITY_LOCKOUT":
                return (
                    f"@customer We take account security very seriously. To protect your private information, "
                    f"please connect with our account security specialists directly in DM so we can verify your identity safely: {APPLE_SECURE_DM_LINK}"
                )
            elif escalation_reason == "BILLING_FINANCIAL_DISPUTE":
                return (
                    f"@customer We understand your billing concern and want to help sort this out. "
                    f"Please join us in a secure DM with your Apple ID email so a billing specialist can review your purchase history: {APPLE_SECURE_DM_LINK}"
                )
            elif escalation_reason == "REPEATED_FAILURE_EXHAUSTED_STEPS":
                return (
                    f"@customer Thank you for letting us know you've already tried those steps. "
                    f"Since the issue persists, let's look deeper together. Please DM us your device model and iOS version: {APPLE_SECURE_DM_LINK}"
                )
            elif escalation_reason == "HIGH_NEGATIVE_SENTIMENT_OR_LEGAL":
                return (
                    f"@customer We hear your frustration, and this is certainly not the experience we want you to have. "
                    f"Please reach out to us via DM right away so a senior advisor can prioritize your case: {APPLE_SECURE_DM_LINK}"
                )
            else:
                return (
                    f"@customer We're here to help get this resolved. "
                    f"Please send us a DM with more details so we can investigate further: {APPLE_SECURE_DM_LINK}"
                )

        if exemplars and exemplars[0]["similarity_score"] >= 0.45:
            top_reply = exemplars[0]["historical_apple_reply"]
            clean_ref = re.sub(r"^@\w+\s*", "", top_reply).strip()
            if "?" in clean_ref and not clean_ref.startswith("Meet us in DM"):
                return f"@customer {clean_ref}"

        if intent == "BATTERY_POWER_HARDWARE":
            return (
                "@customer Battery performance is crucial, and we'd like to help. "
                "Which iOS version is your device running? Also check Settings > Battery to see if any specific app "
                "is using an unusual amount of power."
            )
        elif intent == "CONNECTIVITY_NETWORK":
            return (
                "@customer Having reliable connectivity is essential. "
                "Does this happen on all Wi-Fi/Bluetooth networks or only a specific one? "
                "You can also test resetting your network configuration in Settings > General > Reset > Reset Network Settings."
            )
        elif intent == "DEVICE_AUDIO_DISPLAY_CAMERA":
            return (
                "@customer We want your display and audio working properly. "
                "Does this glitch occur across all apps or only specific ones? "
                "Let us know your exact device model and iOS version from Settings > General > About."
            )
        elif intent == "ACCOUNT_SECURITY_ICLOUD":
            return (
                "@customer We're happy to help with your Apple ID and iCloud setup. "
                "Are you seeing an exact error message when signing in? "
                "Make sure your device is connected to an active internet connection."
            )
        elif intent == "BILLING_SUBSCRIPTIONS":
            return (
                "@customer We can help guide you through subscription and purchase details. "
                "You can review and manage your active subscriptions anytime in Settings > [Your Name] > Subscriptions. "
                "Let us know if you need help requesting a refund."
            )
        else:
            return (
                "@customer We're here to help you get this update sorted out. "
                "Which device model and iOS version do you currently have? Check under Settings > General > About. "
                "Have you tried performing a force restart?"
            )

    def _apply_guardrails(self, text: str, should_escalate: bool) -> str:
        text = re.sub(r"(send|provide|give)\s+(us\s+)?(your\s+)?(password|pin|passcode|credit\s+card)", "contact support securely", text, flags=re.IGNORECASE)
        if not text.startswith("@customer"):
            text = f"@customer {text}"
        if len(text) > 280:
            text = text[:275] + "..."
        return text

    def _call_llm_api(self, customer_text, intent, should_escalate, escalation_reason, exemplars) -> str:
        return self._synthesize_reply(customer_text, intent, should_escalate, escalation_reason, exemplars)

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.process(t) for t in texts]


if __name__ == "__main__":
    agent = AppleSupportAgent()
    agent.initialize()
    test_cases = [
        "My iPhone 7 battery is draining 50% in two hours after iOS 11 update",
        "I dropped my iPhone on the concrete and the entire screen is shattered into pieces",
        "My Apple ID has been disabled and I'm locked out of my account",
        "I was charged $29.99 on my credit card for a subscription I never authorized, I want a refund now!",
        "I already restarted my phone 5 times and reset all settings, but Wi-Fi is still completely greyed out",
        "This is the worst phone I've ever owned, fix this right now or I'm buying a Samsung!"
    ]
    for msg in test_cases:
        out = agent.process(msg)
        print(f"\n[Customer]: {msg}")
        print(f"  -> Intent: {out['predicted_intent']} (conf: {out['confidence']})")
        print(f"  -> Escalate: {out['should_escalate']} ({out['escalation_reason']})")
        print(f"  -> Reason: {out['stated_explanation']}")
        print(f"  -> Draft Reply: {out['draft_reply']}")
