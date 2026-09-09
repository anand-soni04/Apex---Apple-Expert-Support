# Golden Evaluation Set: Sampling & Labelling Methodology

## 1. Overview & Objectives
The Golden Evaluation Set contains exactly **200 hand-audited, verified customer support interactions** between Apple customers and Apple Support on Twitter (`twcs` dataset). 

This benchmark is explicitly constructed to evaluate:
1. **Intent Classification Accuracy & Macro-F1** across 6 data-derived technical domains.
2. **Escalation Precision, Recall, and False Auto-handle Rate (FAR)** against enterprise safety boundaries.
3. **Historical Grounding & Generation Quality** (ROUGE, BLEU, semantic similarity).
4. **LLM-as-Judge Calibration & Agreement** with calibrated human expert ratings (1–5 scale).

---

## 2. Stratified Sampling Methodology
To reflect real-world production distributions while rigorously stressing safety boundaries, we designed a **7-stratum stratified sampling protocol**:

| Stratum Type | Target Count | Description & Operational Risk |
| :--- | :--- | :--- |
| `standard_diagnostic` | 100 | Routine troubleshooting (restart, settings check, OS query). Low operational risk. |
| `multi_intent_ambiguous` | 30 | Queries containing overlapping issues (e.g., iOS update + battery drain + Bluetooth). Tests classifier discernment. |
| `exhausted_steps` | 16 | Customer explicitly stated they already rebooted, reset settings, or restored. Bot must NOT repeat basic steps. |
| `security_critical` | 16 | Apple ID disabled, 2FA failure, account lockout, potential theft. Automated triage prohibited; mandatory human escalation. |
| `hardware_critical` | 14 | Physical damage, cracked screen, water immersion, swollen battery. Requires Genius Bar / mail-in hardware repair. |
| `billing_refund` | 12 | Unexpected charges, refund demands, subscription disputes. Requires agent billing lookup authority. |
| `sentiment_escalation` | 12 | High-anger, churn threats ("switching to Android"), legal threats. Requires immediate human de-escalation. |

---

## 3. Dataset Distributions

### Intent Distribution (200 total)
- **`OS_SOFTWARE_UPDATE`**: 109 (54.5%)
- **`BATTERY_POWER_HARDWARE`**: 29 (14.5%)
- **`DEVICE_AUDIO_DISPLAY_CAMERA`**: 19 (9.5%)
- **`BILLING_SUBSCRIPTIONS`**: 18 (9.0%)
- **`ACCOUNT_SECURITY_ICLOUD`**: 16 (8.0%)
- **`CONNECTIVITY_NETWORK`**: 9 (4.5%)

### Escalation Distribution
- **Auto-Handle (`false`)**: 134 (67.0%)
- **Escalate to Human (`true`)**: 66 (33.0%)

### Escalation Reason Breakdown
- **`NONE_AUTO_HANDLE`**: 134 (67.0%)
- **`REPEATED_FAILURE_EXHAUSTED_STEPS`**: 16 (8.0%)
- **`HARDWARE_DAMAGE_OR_SAFETY`**: 14 (7.0%)
- **`BILLING_FINANCIAL_DISPUTE`**: 12 (6.0%)
- **`ACCOUNT_SECURITY_LOCKOUT`**: 12 (6.0%)
- **`HIGH_NEGATIVE_SENTIMENT_OR_LEGAL`**: 12 (6.0%)

### Human Quality Score Distribution (Reference Replies)
- **Mean Quality Score**: 3.57 / 5.0
- **Score 5 (Exemplary)**: 45 (22.5%)
- **Score 4 (Good Brand Voice)**: 24 (12.0%)
- **Score 3 (Generic DM Redirect)**: 131 (65.5%)
- **Score 2 / 1 (Suboptimal)**: 0 (0.0%)

---

## 4. Labelling Rules & Quality Control
1. **Zero Data Leakage**: The golden evaluation set is strictly partitioned from any model training or retrieval indexing.
2. **Deterministic Label Audit**: Every example was filtered to ensure:
   - Substantive query length (>= 5 content words, not just naked screenshot URLs).
   - Verifiable intent alignment.
   - Mutually exclusive escalation rationale.
3. **Escalation Priority Order**: If a query has both hardware damage and angry sentiment, `HARDWARE_DAMAGE_OR_SAFETY` takes precedence as the root physical cause.
