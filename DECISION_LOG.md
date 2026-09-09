# Engineering Decision Log

14 technical and architectural decisions made while building the Apple Support agent, including trade-offs and rationale.

---

### 1. Brand Selection: Apple Support (`@AppleSupport`)
* **Decision**: Selected `AppleSupport` from the TWCS dataset over `AmazonHelp` and `Uber_Support`.
* **Rationale**: Apple Support queries combine technical troubleshooting (OS updates, battery health, Wi-Fi/Bluetooth, hardware glitches) with clear safety boundaries (cracked glass, account lockouts, billing disputes). Delivery and ride-sharing brands rely heavily on live package/driver tracking queries, which without internal private tracking APIs reduce to generic "check your tracking link" deflections.

---

### 2. Intent Granularity: 6 Operational Domains vs. Fine-Grained Taxonomies
* **Decision**: Designed a 6-class operational intent taxonomy (`OS_SOFTWARE_UPDATE`, `BATTERY_POWER_HARDWARE`, `CONNECTIVITY_NETWORK`, `ACCOUNT_SECURITY_ICLOUD`, `BILLING_SUBSCRIPTIONS`, `DEVICE_AUDIO_DISPLAY_CAMERA`) rather than an ultra-fine 70+ class taxonomy.
* **Rationale**: Twitter queries are terse (<= 280 chars) and often mention multiple issues. Fine-grained taxonomies in short-text domains suffer from class imbalance and label overlap. These 6 categories align with common support escalation queues.

---

### 3. Partitioning the Golden Evaluation Set
* **Decision**: Filtered out all 200 evaluation set tweet IDs from both the training corpus and the retrieval grounding index at startup.
* **Rationale**: In retrieval-augmented generation (RAG), if test queries or their reference replies remain in the index, nearest-neighbor matching trivially fetches the target answer, inflating ROUGE and similarity metrics. Explicit ID exclusion prevents test leakage.

---

### 4. Safety Metric: False Auto-Handle Rate (FAR) Over Standard F1
* **Decision**: Prioritized False Auto-handle Rate (FAR = FN / (FN + TP)) as the primary escalation safety metric.
* **Rationale**: Operational cost is asymmetric. Routing a simple restart question to a human agent costs staff time. In contrast, auto-handling a compromised Apple ID, swelling battery, or unauthorized charge causes security breaches and customer churn. FAR measures the fraction of critical issues the model failed to escalate.

---

### 5. Precedence Order in Escalation Triggers
* **Decision**: Enforced an explicit evaluation order when parsing customer messages:
  Hardware Damage > Account Security > Billing Disputes > Exhausted Steps > Frustration/Sentiment
* **Rationale**: Customer messages are frequently compound (e.g. *"I dropped my phone, the screen cracked, and I am furious"*). If classified purely by sentiment, the message routes to a de-escalation agent rather than a repair workflow. The root physical cause takes precedence over emotional expressions.

---

### 6. Substantive Query Threshold (>= 5 Content Words)
* **Decision**: Required at least 5 content words for benchmark inclusion, filtering out image-only links and single-word pings.
* **Rationale**: A large portion of raw customer support tweets consist solely of screenshot links or greetings like *"help?"*. Evaluating an NLP agent on image URLs tests OCR rather than dialogue reasoning.

---

### 7. Normalizing Customer Handles
* **Decision**: Replaced anonymized customer handles (`@115854`) with `@customer` prior to vectorization.
* **Rationale**: Customer handles in the dataset are anonymized numeric IDs. Standard tokenizers treat them as vocabulary features, causing models to memorize correlations between specific user numbers and problem types.

---

### 8. Evaluating Dialogue Quality Beyond ROUGE and BLEU
* **Decision**: Relied on diagnostic actionability rubrics rather than n-gram overlap scores as the primary quality metric.
* **Rationale**: A trivial canned response achieved higher ROUGE-1 (0.335) than the proposed agent (0.211) because historical Twitter replies heavily reuse boilerplate greetings (*"We'd like to help... Send us a DM"*). N-gram metrics reward generic boilerplate and penalize issue-specific diagnostic phrasing.

---

### 9. Quadratic Weighted Kappa ($\kappa_w$) for Judge Calibration
* **Decision**: Used Cohen's Quadratic Weighted Kappa ($\kappa_w$) to measure agreement between human ratings and rubric judge scores.
* **Rationale**: The quality rubric uses an ordinal 1-to-5 scale. Standard unweighted Kappa treats a 1-point difference (rating 4 vs 5) the same as a 4-point disagreement (rating 1 vs 5). Quadratic weighting penalizes larger rating discrepancies more heavily.

---

### 10. Self-Contained Local Execution
* **Decision**: Built a local pipeline that executes in seconds without requiring external paid API keys, with optional provider hooks for live LLMs.
* **Rationale**: Eliminates dependencies on external API keys, network latency, and billing limits during review, ensuring immediate reproducibility.

---

### 11. Post-Generation Credential Guardrail
* **Decision**: Implemented a regex filter that intercepts any output requesting passwords, PINs, or card numbers, replacing them with secure DM routing.
* **Rationale**: Twitter support threads are public. An automated agent asking for login credentials or payment numbers publicly creates an immediate compliance and security risk.

---

### 12. Penalizing Premature DM Deflection
* **Decision**: Penalized immediate DM routing in the rubric for routine technical queries that can be resolved via public troubleshooting steps.
* **Rationale**: Human agents historically deflected many tweets to DM by default. However, forcing users into private queues for routine questions (*"How do I check battery health?"*) reduces first-contact resolution and creates unnecessary queue volume.

---

### 13. Sublinear Word and Character N-Grams
* **Decision**: Combined word-level n-grams (1, 2) with character-level boundary n-grams (3, 5) using sublinear term frequency scaling for intent classification.
* **Rationale**: Social media text contains typos, shorthand, and concatenated version strings (`ios11.1`). Character n-grams provide sub-word robustness, improving classification macro-F1 on minority classes.

---

### 14. Modular Pipeline vs. End-to-End Black Box
* **Decision**: Implemented a modular pipeline (Classifier -> Escalation Policy -> Retrieval -> Guarded Synthesis) rather than a single end-to-end generative model.
* **Rationale**: Modular pipelines allow independent auditing, deterministic policy enforcement, and targeted debugging of classification versus safety logic, which is critical in regulated support environments.
