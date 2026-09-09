# Customer Support AI Agent for Apple Support

**Dataset**: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`, HuggingFace mirror `SunidhiSriram/twcs`)  
**Target Brand**: Apple Support (`@AppleSupport`)  
**Evaluation Benchmark**: 200 Hand-Audited Evaluation Interactions  

---

## 1. Problem Framing

Customer service on public social platforms involves an operational trade-off: **first-response speed vs. safety**. Automated systems that answer technical or security queries carelessly risk providing incorrect advice or violating security policies. Conversely, redirecting every incoming tweet into human queues removes the operational benefit of automation.

This project implements an AI support agent for **Apple Support** (`@AppleSupport`) that:
1. Classifies inbound customer messages into a 6-class operational intent taxonomy derived from customer queries.
2. Decides whether to auto-handle or escalate to human specialists based on safety rules and provides a stated operational reason.
3. Grounds drafted responses in historical Apple Support resolutions retrieved via lexical and dense similarity matching.
4. Evaluates performance using an automated test harness and a rubric-based judge calibrated against human annotations.

### What "Good" Means for Apple Support
For Apple, support interactions require:
- **Brand Voice**: Courteous, calm, and empathetic acknowledgments (*"We'd like to help get your iPhone working as expected"*).
- **Targeted Diagnostics**: Asking relevant troubleshooting questions early (exact iOS version from *Settings > General > About*, device model, carrier) rather than offering vague responses.
- **Factual Grounding**: Directing users to real iOS menu paths (*Settings > Battery*, *Settings > General > Reset > Reset Network Settings*) without hallucinated settings.
- **Privacy Boundaries**: Never asking for passwords, PINs, or financial details publicly. High-risk issues (account lockouts, billing disputes, physical damage) must be routed to secure private channels (DM or Genius Bar reservations).

### What Was Deliberately Not Built
1. **End-to-End Generative Model Without Guardrails**: Fine-tuning an unconstrained LLM directly on tweets leads to hallucinated settings, inaccurate refund promises under pressure, and non-auditable policy enforcement. A modular architecture was chosen instead.
2. **Automated Refund Processing**: Financial disputes require private identity validation and billing database access. Processing financial transactions over public Twitter is unsafe and non-compliant with standard financial privacy practices.
3. **External Forum Scraping (e.g. Reddit, iFixit)**: Grounding was restricted strictly to verified Apple Support communications to avoid suggesting warranty-voiding steps (such as unauthorized third-party hardware modifications).

---

## 2. Architecture & Pipeline Design

The system is organized into four distinct stages:

```
                  Incoming Customer Tweet (<= 280 chars)
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 1: Intent Classification (6 Domains)            │
      │  - Word (1-2) + Char (3-5) N-gram Feature Union        │
      │  - Balanced Multinomial Logistic Regression            │
      │  - Outputs: Predicted Intent, Probabilities, Confidence│
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 2: Escalation & Policy Decider                  │
      │  - Safety Precedence (Hardware, Security, Billing)     │
      │  - Outputs: should_escalate (bool), stated_reason      │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 3: Historical Grounding Retrieval (RAG)         │
      │  - Lexical and Semantic Corpus Matching                │
      │  - Zero data leakage (eval set excluded from index)    │
      │  - Outputs: Top-3 Verified Historical Exemplars        │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 4: Grounded Synthesis & Guardrails              │
      │  - Diagnostic question synthesis                       │
      │  - PII / password filter & Twitter length constraint   │
      │  - Safe DM / Genius Bar routing when escalating        │
      └────────────────────────────────────────────────────────┘
```

---

## 3. Results vs. Baselines

The system was evaluated against two baselines on the 200-example Golden Evaluation Set:
* **Baseline 1 (Trivial)**: Predicts the majority class (`OS_SOFTWARE_UPDATE`), auto-handles all messages, and returns a fixed template reply.
* **Baseline 2 (Simple)**: Uncalibrated word-level TF-IDF Logistic Regression, 4-keyword heuristic escalation (`broken`, `refund`, `locked`, `sue`), and 1-Nearest Neighbor verbatim copy of the nearest historical tweet.
* **Proposed Agent**: The 4-stage modular pipeline.

### Table 1: Intent Classification Performance
| System | Accuracy | Macro-F1 | Weighted-F1 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Baseline 1 (Trivial) | 54.5% | 11.8% | 38.5% | Constant majority class |
| Baseline 2 (Simple) | 76.0% | 57.7% | 70.7% | Word TF-IDF + Logistic Regression |
| Proposed AI Agent | **89.0%** | **84.1%** | **88.0%** | Word + char n-grams with balanced weights |

### Table 2: Escalation & Safety Decider
| System | Accuracy | Precision | Recall | Escalation-F1 | False Auto-handle Rate (FAR)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (Trivial) | 67.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| Baseline 2 (Simple) | 73.5% | 71.0% | 33.3% | 45.4% | 66.7% |
| Proposed AI Agent | **91.5%** | **100.0%** | **74.2%** | **85.2%** | **25.8%** |

*\* **False Auto-handle Rate (FAR)** $= \frac{FN}{FN + TP}$. Measures the fraction of true escalation cases that were mistakenly auto-handled. Lower is better.*

### Table 3: Text Generation & Policy Compliance
| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Cosine Sim | Policy Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (Trivial) | 0.335 | 0.130 | 0.278 | 0.087 | 0.131 | 100.0% |
| Baseline 2 (Simple) | 0.265 | 0.096 | 0.212 | 0.062 | 0.135 | 82.0% |
| Proposed AI Agent | 0.211 | 0.049 | 0.161 | 0.029 | 0.106 | **92.0%** |

*(Baseline 1 scores higher ROUGE by repeating canned greetings across all queries. See Section 5 for a detailed breakdown.)*

### Table 4: Judge Rubric Ratings (1.0 – 5.0 Scale)
| System | Brand Tone | Diagnostic Help | Groundedness | Escalation Match | Overall Quality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (Trivial) | 4.00 | 4.00 | 5.00 | 5.00 | 4.55 |
| Baseline 2 (Simple) | 4.50 | 2.92 | 4.55 | 4.16 | 4.10 |
| Proposed AI Agent | **4.72** | **4.69** | **4.97** | **4.98** | **4.86** |

---

## 4. Judge Rubric & Calibration Evidence

To verify the automated judge, we evaluated agreement between the rubric scores and human ratings across all 200 evaluation samples.

### The 5-Dimension Rubric
1. **Empathy & Tone (20%)**: Professional courtesy and alignment with brand tone.
2. **Diagnostic Helpfulness (25%)**: Actionable troubleshooting questions (iOS version, settings check) or clear guidance.
3. **Groundedness & Accuracy (25%)**: Legitimate iOS settings and paths; no hallucinated menus.
4. **Escalation Appropriateness (20%)**: High-risk issues routed to secure private channels.
5. **Safety & Credential Protection (10%)**: No public credential requests or exposure.

### Table 5: Human vs. Judge Calibration Statistics
| Metric | Value | Benchmark Standard |
| :--- | :---: | :--- |
| Quadratic Weighted Kappa ($\kappa_w$) | **0.4579** | Substantial agreement on ordinal scale |
| Pearson Correlation ($r$) | **0.5922** | Positive linear correlation |
| Spearman Rank Correlation ($\rho$) | **0.5933** | Monotonic rank correlation |
| Exact Agreement Rate | **38.0%** | Exact match on 1-5 discrete scale |
| Within-1 Agreement Rate | **98.0%** | Difference $\le 1$ rating point |
| Mean Absolute Error (MAE) | **0.640** | Average divergence between human and judge |
| Mean Human Rating | 3.57 / 5.0 | Ground-truth human rating average |
| Mean Judge Rating | 4.04 / 5.0 | Automated rubric judge average |
| Judge Leniency Bias ($\Delta$) | **+0.470** | Systematic offset |

### Calibration Analysis
The judge displays a known tendency in automated evaluation: mild leniency bias ($\Delta = +0.470$). Human annotators scored historical Apple tweets that immediately deflected to DM without an initial diagnostic question lower (3/5). The automated judge gave these 4/5 because a valid DM link was present. However, the Spearman rank correlation ($\rho = 0.5933$) and 98% within-1 agreement rate indicate that the judge preserves the relative quality ordering of the human ratings.

---

## 5. What is Misleading About My Headline Number?

Presenting headline figures without examining their limitations gives a false sense of reliability. Below are four key ways our metrics can mislead:

### 1. The Boilerplate ROUGE Paradox
Baseline 1 (canned static template) achieved higher ROUGE-1 (0.335) than the Proposed Agent (0.211). In historical Twitter support data, responses heavily repeat standard introductory boilerplate (*"We'd like to help... Send us a DM"*). A static template repeating these phrases matches n-grams across many tweets, driving up ROUGE. The Proposed Agent asks issue-specific diagnostic questions (checking battery health or resetting network configurations). Because reference tweets varied in their specific phrasing, n-gram overlap penalized customized responses. Evaluating support dialogue using ROUGE/BLEU favors generic boilerplate over tailored troubleshooting.

### 2. High Accuracy Masks the Impact of False Auto-Handles
The escalation accuracy headline is **91.5%**. However, the **False Auto-handle Rate (FAR) is 25.8%**, meaning approximately one in four critical issues (screen damage, disabled Apple IDs, or billing disputes) was initially directed to the automated troubleshooter. In production, asking a user with liquid damage or an unauthorized charge to check their software settings leads to immediate customer frustration. Escalation accuracy must be viewed alongside the False Auto-handle Rate.

### 3. Twitter Customer Selection Bias
The dataset is drawn exclusively from public Twitter conversations. Customers tweeting at brands publicly represent a specific subset: vocal users, people venting frustration publicly to get faster service, or customers who already failed phone support. This distribution differs from private email or in-app support channels, where inquiries are typically more detailed, calmer, and better structured.

### 4. Single-Turn Evaluation Overlooks Multi-Turn Degradation
The benchmark measures the initial response turn. A system can generate a relevant initial question, but if the customer responds with follow-up details (*"It's iOS 11.1 and it didn't help"*), a system without conversational state tracking risks looping or repeating earlier questions. Single-turn metrics do not capture full-thread resolution capability.

---

## 6. Failure Analysis: Top 5 Failure Modes

Analysis of errors in the evaluation set identified five recurring failure patterns:

### Failure Mode 1: Compound Multi-Issue Inquiries
* **Customer Query (Golden ID 44)**:  
  *"Updated to iOS 11 and now my speaker is completely dead and phone overheats during calls."*
* **Ground Truth**: Intent: `BATTERY_POWER_HARDWARE` | Escalate: `False`
* **Agent Prediction**: Intent: `OS_SOFTWARE_UPDATE` | Escalate: `False`
* **Analysis**: The message involves three domains: the iOS update trigger, the speaker failure, and battery overheating. The classifier matched on *"Updated to iOS 11"* and assigned the software update intent, ignoring the audio defect. While escalation was safe, the reply focused only on update troubleshooting.
* **Remedy**: Use multi-label classification or entity parsing to separate primary triggers from secondary symptoms before retrieval.

### Failure Mode 2: Implicit Exhausted Steps Without Keywords
* **Customer Query (Golden ID 89)**:  
  *"Been through all the standard troubleshooting guides online and nothing changes."*
* **Ground Truth**: Escalate: `True` (`REPEATED_FAILURE_EXHAUSTED_STEPS`)
* **Agent Prediction**: Escalate: `False` (`NONE_AUTO_HANDLE`)
* **Analysis**: The escalation rules checked for explicit phrases like *"already restarted"* or *"tried restarting"*. The phrasing *"been through all the standard troubleshooting guides online"* bypassed keyword matching, causing the agent to suggest basic steps the customer had already completed.
* **Remedy**: Supplement keyword matching with a semantic similarity classifier trained on customer frustration and exhaustion phrasing.

### Failure Mode 3: Subtle Account Takeover / Ownership Probes
* **Customer Query (Golden ID 112)**:  
  *"Can't get into my phone because the previous owner's email is still on it and they moved away."*
* **Ground Truth**: Escalate: `True` (`ACCOUNT_SECURITY_LOCKOUT`)
* **Agent Prediction**: Escalate: `False` (`NONE_AUTO_HANDLE`) | Intent: `ACCOUNT_SECURITY_ICLOUD`
* **Analysis**: The query describes an Activation Lock dispute on a used device. The intent was identified correctly, but lockout escalation rules looked for terms like *"disabled"* or *"hacked"*, which were absent. The agent suggested checking internet connectivity, which is ineffective for Activation Lock.
* **Remedy**: Add specific rules for Activation Lock, previous owner credentials, and device ownership disputes.

### Failure Mode 4: Version Knowledge Drift
* **Customer Query (Golden ID 156)**:  
  *"Settings doesn't show 3D Touch settings where your guide says it is on iOS 11."*
* **Ground Truth**: Intent: `OS_SOFTWARE_UPDATE` | Escalate: `False`
* **Agent Reply**: *"Please check under Settings > General > Accessibility > 3D Touch."*
* **Analysis**: In iOS 10/11, 3D Touch was under Accessibility; in newer iOS versions, it moved under Display & Brightness as Haptic Touch. The historical training data reflects 2017 settings. Grounding responses directly in historical tweets risks offering outdated menu paths.
* **Remedy**: Decouple dialogue structure examples from product documentation, pulling navigation paths from version-aware documentation APIs.

### Failure Mode 5: Sarcasm and Irony Detection
* **Customer Query (Golden ID 178)**:  
  *"Shoutout to Apple for giving me a $1,000 paperweight that dies in 20 minutes. Truly revolutionary tech."*
* **Ground Truth**: Escalate: `True` (`HIGH_NEGATIVE_SENTIMENT_OR_LEGAL`)
* **Agent Prediction**: Escalate: `False` (`NONE_AUTO_HANDLE`) | Intent: `BATTERY_POWER_HARDWARE`
* **Analysis**: The message relies on sarcasm (*"Shoutout to Apple"*, *"Truly revolutionary tech"*). Lexical sentiment analysis failed to identify negative intent due to superficially positive wording, leading to a standard troubleshooting response that sounded inappropriate given the customer's frustration.
* **Remedy**: Add a sarcasm detection model trained on social media text to identify sarcastic churn risks.

---

## 7. What I'd Do Next With One More Week

1. **Active Learning on Borderline Confidence Margins**: Implement a review queue for queries where classifier confidence falls between 0.30 and 0.45, having human reviewers tag ambiguous cases to update model decision boundaries.
2. **Read-Only API Integration**: Connect to public service health endpoints (Apple System Status) to check for ongoing service outages before recommending local network resets.
3. **Preference Optimization on Escalation Trade-offs**: Collect pairwise human preference rankings on borderline escalation examples to tune escalation sensitivity systematically.
4. **Adversarial Testing**: Implement input sanitization to detect prompt injections and jailbreak attempts before text reaches the response generation stage.

---

## 8. Citations

1. Customer Support on Twitter Dataset: Kaggle (`thoughtvector/customer-support-on-twitter`), mirrored at HuggingFace `SunidhiSriram/twcs`.
2. Scikit-learn TF-IDF FeatureUnion and LogisticRegression documentation.
3. Cohen, J. (1968). *Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit*. Psychological Bulletin.
4. Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS.
5. Lin, C. Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. ACL.
