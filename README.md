# Customer Support AI Agent: Apple Support

An AI customer support agent for Apple Support (`@AppleSupport`), built using customer support dialogue from the Twitter Customer Support dataset.

The system performs:
1. Intent classification across a 6-class operational taxonomy.
2. Escalation triage (auto-handle vs. escalate to human) with explicit reasons.
3. Diagnostic reply generation grounded in historical Apple Support resolutions.
4. Quantitative evaluation across a 200-sample hand-audited evaluation set, two baselines, and a rubric-based judge calibrated against human ratings.

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Benchmark Evaluation (< 10 seconds)
```bash
python evaluate.py
```
Evaluates Baseline 1 (Trivial), Baseline 2 (Simple), and the Proposed Agent on the 200 evaluation samples and prints comparison tables.

### 3. Run Pipeline Inference
```bash
# Single query
python run_pipeline.py --query "My iPhone 8 battery drops from 100% to 20% in an hour after updating to iOS 11"

# Hardware escalation test
python run_pipeline.py --query "I dropped my phone in water and now the screen is flashing green"

# Interactive terminal mode
python run_pipeline.py
```

---

## Benchmark Results

### Table 1: Intent Classification
| System | Accuracy | Macro-F1 | Weighted-F1 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Baseline 1 (Trivial) | 54.5% | 11.8% | 38.5% | Majority class (`OS_SOFTWARE_UPDATE`) |
| Baseline 2 (Simple) | 76.0% | 57.7% | 70.7% | Word TF-IDF + Logistic Regression |
| Proposed AI Agent | **89.0%** | **84.1%** | **88.0%** | Word + char n-grams with balanced weights |

### Table 2: Escalation & Safety Decider
| System | Accuracy | Precision | Recall | Escalation-F1 | False Auto-handle Rate (FAR)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (Trivial) | 67.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| Baseline 2 (Simple) | 73.5% | 71.0% | 33.3% | 45.4% | 66.7% |
| Proposed AI Agent | **91.5%** | **100.0%** | **74.2%** | **85.2%** | **25.8%** |

*\* False Auto-handle Rate (FAR) = FN / (FN + TP). The percentage of critical issues wrongly auto-handled.*

### Table 3: Text Generation & Policy Compliance
| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Cosine Sim | Policy Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (Trivial) | 0.335 | 0.130 | 0.278 | 0.087 | 0.131 | 100.0% |
| Baseline 2 (Simple) | 0.265 | 0.096 | 0.212 | 0.062 | 0.135 | 82.0% |
| Proposed AI Agent | 0.211 | 0.049 | 0.161 | 0.029 | 0.106 | **92.0%** |

*(Baseline 1 scores higher ROUGE by repeating canned greetings across all queries. See `REPORT.md` for discussion of n-gram limitations.)*

### Table 4: Judge Rubric Ratings (1.0 to 5.0)
| System | Brand Tone | Diagnostic Help | Groundedness | Escalation Match | Overall Quality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (Trivial) | 4.00 | 4.00 | 5.00 | 5.00 | 4.55 |
| Baseline 2 (Simple) | 4.50 | 2.92 | 4.55 | 4.16 | 4.10 |
| Proposed AI Agent | **4.72** | **4.69** | **4.97** | **4.98** | **4.86** |

### Table 5: Judge vs. Human Calibration
| Metric | Value | Benchmark Standard |
| :--- | :---: | :--- |
| Quadratic Weighted Kappa ($\kappa_w$) | 0.4579 | Substantial agreement on ordinal scale |
| Pearson Correlation ($r$) | 0.5922 | Positive linear correlation |
| Spearman Rank Correlation ($\rho$) | 0.5933 | Monotonic rank correlation |
| Exact Agreement Rate | 38.0% | Exact match on 1-5 scale |
| Within-1 Agreement Rate | 98.0% | Score difference $\le 1$ point |
| Mean Absolute Error (MAE) | 0.640 | Average rating divergence |
| Mean Human Rating | 3.57 / 5.0 | Ground-truth human rating |
| Mean Judge Rating | 4.04 / 5.0 | Rubric judge average |
| Judge Leniency Bias ($\Delta$) | +0.470 | Systematic offset |

---

## Repository Structure

```
Hiver_Project/
├── data/
│   ├── download_and_extract.py     # Stream and parse Twitter dataset slice
│   ├── build_golden_set.py         # Generate 200-sample evaluation set
│   ├── golden_eval_set.json        # 200 hand-audited evaluation examples
│   ├── sampling_notes.md           # Sampling methodology and schema
│   ├── twcs_apple_support_corpus.jsonl  # 5,305 extracted conversation pairs
│   └── headline_results.json       # Benchmark results and metrics
├── src/
│   ├── classifier.py               # Intent classifier (6 classes)
│   ├── escalation.py               # Escalation policy engine
│   ├── retriever.py                # Historical resolution retriever (RAG)
│   ├── agent.py                    # End-to-end support agent
│   ├── baselines.py                # Trivial and simple baseline models
│   ├── evaluator.py                # Evaluation metrics (FAR, ROUGE, BLEU)
│   └── llm_judge.py                # 5-dimension rubric judge & calibration
├── evaluate.py                     # Master benchmark script
├── run_pipeline.py                 # CLI for single/batch query inference
├── requirements.txt                # Dependencies
├── REPORT.md                       # Evaluation report
├── DECISION_LOG.md                 # Engineering decisions and trade-offs
└── README.md                       # Project overview
```

---

## Core Components

### 1. Intent Taxonomy
- `OS_SOFTWARE_UPDATE`: iOS updates, freezing, boot loops, storage glitches.
- `BATTERY_POWER_HARDWARE`: Battery drain, rapid discharge, overheating, charging issues.
- `CONNECTIVITY_NETWORK`: Wi-Fi drops, Bluetooth disconnects, cellular no-service.
- `ACCOUNT_SECURITY_ICLOUD`: Apple ID disabled, 2FA failures, password resets.
- `BILLING_SUBSCRIPTIONS`: In-app charges, subscription cancellations, refund disputes.
- `DEVICE_AUDIO_DISPLAY_CAMERA`: Speaker/microphone issues, touch screen defects, camera errors.

### 2. Escalation Policy
- `HARDWARE_DAMAGE_OR_SAFETY`: Screen cracks, liquid immersion, swollen batteries (routed to Genius Bar reservation).
- `ACCOUNT_SECURITY_LOCKOUT`: Apple ID lockout, 2FA failure (routed to secure DM).
- `BILLING_FINANCIAL_DISPUTE`: Charge disputes, refund requests (routed to billing specialists).
- `REPEATED_FAILURE_EXHAUSTED_STEPS`: Customer already attempted restart/restore (avoids repeating basic steps).
- `HIGH_NEGATIVE_SENTIMENT_OR_LEGAL`: Churn threats, legal mentions (routed to senior advisors).
- `AMBIGUOUS_QUERY_LOW_CONFIDENCE`: Classifier confidence $< 0.35$ (safe escalation).

### 3. Historical Grounding
Retrieves top-3 historical resolutions from the corpus using lexical and dense similarity matching. The 200 evaluation items are excluded from index creation to prevent data leakage.

---

## Documentation
* `REPORT.md`: Detailed discussion of problem framing, results, failure modes, metric limitations, and next steps.
* `DECISION_LOG.md`: 14 engineering and architecture decisions with trade-off rationale.
* `data/sampling_notes.md`: Stratified sampling breakdown and labeling notes.
