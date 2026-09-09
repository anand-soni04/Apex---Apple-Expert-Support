import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = str(Path(__file__).resolve().parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines import TrivialBaseline, SimpleBaseline
from src.agent import AppleSupportAgent
from src.evaluator import EvaluationHarness
from src.llm_judge import SupportReplyJudge, compute_calibration_report

GOLDEN_SET_PATH = Path(ROOT_DIR) / "data" / "golden_eval_set.json"
RESULTS_OUTPUT_PATH = Path(ROOT_DIR) / "data" / "headline_results.json"


def print_table(headers: List[str], rows: List[List[Any]]):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row)))


def run_benchmark():
    start_time = time.time()
    print("=" * 80)
    print("  EVALUATION BENCHMARK: APPLE SUPPORT AI AGENT")
    print("=" * 80)

    if not GOLDEN_SET_PATH.exists():
        print(f"Error: Golden evaluation set not found at {GOLDEN_SET_PATH}")
        sys.exit(1)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"Loaded Golden Evaluation Set: {len(golden_set)} samples.")

    customer_texts = [item["customer_text"] for item in golden_set]
    human_scores = [item["human_quality_score"] for item in golden_set]

    trivial_sys = TrivialBaseline()
    simple_sys = SimpleBaseline()
    simple_sys.train()
    
    proposed_agent = AppleSupportAgent()
    proposed_agent.initialize()

    trivial_preds = trivial_sys.batch_predict(customer_texts)
    simple_preds = simple_sys.batch_predict(customer_texts)
    proposed_preds = proposed_agent.batch_process(customer_texts)

    harness = EvaluationHarness(golden_set)
    t_metrics = harness.evaluate_system("Baseline 1 (Trivial)", trivial_preds)
    s_metrics = harness.evaluate_system("Baseline 2 (Simple)", simple_preds)
    p_metrics = harness.evaluate_system("Proposed AI Agent", proposed_preds)

    judge = SupportReplyJudge()
    t_judge = judge.evaluate_dataset(golden_set, trivial_preds)
    s_judge = judge.evaluate_dataset(golden_set, simple_preds)
    p_judge = judge.evaluate_dataset(golden_set, proposed_preds)

    ref_predictions = [{
        "draft_reply": item["historical_reference_reply"],
        "should_escalate": item["true_escalate"],
        "predicted_intent": item["true_intent"]
    } for item in golden_set]
    ref_judge = judge.evaluate_dataset(golden_set, ref_predictions)
    judge_scores_on_ref = [rj["discrete_score"] for rj in ref_judge]

    calibration_report = compute_calibration_report(human_scores, judge_scores_on_ref)

    def summarize_judge(scores_list):
        return {
            "tone": round(sum(s["score_tone"] for s in scores_list) / len(scores_list), 2),
            "diagnostic": round(sum(s["score_diagnostic"] for s in scores_list) / len(scores_list), 2),
            "groundedness": round(sum(s["score_groundedness"] for s in scores_list) / len(scores_list), 2),
            "escalation": round(sum(s["score_escalation"] for s in scores_list) / len(scores_list), 2),
            "overall": round(sum(s["overall_score"] for s in scores_list) / len(scores_list), 2)
        }

    t_judge_summary = summarize_judge(t_judge)
    s_judge_summary = summarize_judge(s_judge)
    p_judge_summary = summarize_judge(p_judge)

    print("\nTABLE 1: INTENT CLASSIFICATION")
    headers_t1 = ["System", "Accuracy", "Macro-F1", "Weighted-F1"]
    rows_t1 = [
        ["Baseline 1 (Trivial)", f"{t_metrics['intent_accuracy']*100:.1f}%", f"{t_metrics['intent_macro_f1']*100:.1f}%", f"{t_metrics['intent_weighted_f1']*100:.1f}%"],
        ["Baseline 2 (Simple)", f"{s_metrics['intent_accuracy']*100:.1f}%", f"{s_metrics['intent_macro_f1']*100:.1f}%", f"{s_metrics['intent_weighted_f1']*100:.1f}%"],
        ["Proposed AI Agent", f"{p_metrics['intent_accuracy']*100:.1f}%", f"{p_metrics['intent_macro_f1']*100:.1f}%", f"{p_metrics['intent_weighted_f1']*100:.1f}%"]
    ]
    print_table(headers_t1, rows_t1)

    print("\nTABLE 2: ESCALATION & SAFETY DECIDER")
    headers_t2 = ["System", "Accuracy", "Precision", "Recall", "Escalation-F1", "False Auto-handle Rate (FAR)"]
    rows_t2 = [
        ["Baseline 1 (Trivial)", f"{t_metrics['escalation_accuracy']*100:.1f}%", f"{t_metrics['escalation_precision']*100:.1f}%", f"{t_metrics['escalation_recall']*100:.1f}%", f"{t_metrics['escalation_f1']*100:.1f}%", f"{t_metrics['false_auto_handle_rate']*100:.1f}%"],
        ["Baseline 2 (Simple)", f"{s_metrics['escalation_accuracy']*100:.1f}%", f"{s_metrics['escalation_precision']*100:.1f}%", f"{s_metrics['escalation_recall']*100:.1f}%", f"{s_metrics['escalation_f1']*100:.1f}%", f"{s_metrics['false_auto_handle_rate']*100:.1f}%"],
        ["Proposed AI Agent", f"{p_metrics['escalation_accuracy']*100:.1f}%", f"{p_metrics['escalation_precision']*100:.1f}%", f"{p_metrics['escalation_recall']*100:.1f}%", f"{p_metrics['escalation_f1']*100:.1f}%", f"{p_metrics['false_auto_handle_rate']*100:.1f}%"]
    ]
    print_table(headers_t2, rows_t2)

    print("\nTABLE 3: TEXT GENERATION & POLICY COMPLIANCE")
    headers_t3 = ["System", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-4", "Cosine Sim", "Policy Compliance"]
    rows_t3 = [
        ["Baseline 1 (Trivial)", f"{t_metrics['rouge_1']:.3f}", f"{t_metrics['rouge_2']:.3f}", f"{t_metrics['rouge_l']:.3f}", f"{t_metrics['bleu_4']:.3f}", f"{t_metrics['semantic_similarity']:.3f}", f"{t_metrics['policy_compliance_rate']*100:.1f}%"],
        ["Baseline 2 (Simple)", f"{s_metrics['rouge_1']:.3f}", f"{s_metrics['rouge_2']:.3f}", f"{s_metrics['rouge_l']:.3f}", f"{s_metrics['bleu_4']:.3f}", f"{s_metrics['semantic_similarity']:.3f}", f"{s_metrics['policy_compliance_rate']*100:.1f}%"],
        ["Proposed AI Agent", f"{p_metrics['rouge_1']:.3f}", f"{p_metrics['rouge_2']:.3f}", f"{p_metrics['rouge_l']:.3f}", f"{p_metrics['bleu_4']:.3f}", f"{p_metrics['semantic_similarity']:.3f}", f"{p_metrics['policy_compliance_rate']*100:.1f}%"]
    ]
    print_table(headers_t3, rows_t3)

    print("\nTABLE 4: JUDGE RUBRIC RATINGS (1.0 - 5.0)")
    headers_t4 = ["System", "Brand Tone", "Diagnostic Help", "Groundedness", "Escalation Match", "Overall Quality"]
    rows_t4 = [
        ["Baseline 1 (Trivial)", f"{t_judge_summary['tone']:.2f}", f"{t_judge_summary['diagnostic']:.2f}", f"{t_judge_summary['groundedness']:.2f}", f"{t_judge_summary['escalation']:.2f}", f"{t_judge_summary['overall']:.2f}"],
        ["Baseline 2 (Simple)", f"{s_judge_summary['tone']:.2f}", f"{s_judge_summary['diagnostic']:.2f}", f"{s_judge_summary['groundedness']:.2f}", f"{s_judge_summary['escalation']:.2f}", f"{s_judge_summary['overall']:.2f}"],
        ["Proposed AI Agent", f"{p_judge_summary['tone']:.2f}", f"{p_judge_summary['diagnostic']:.2f}", f"{p_judge_summary['groundedness']:.2f}", f"{p_judge_summary['escalation']:.2f}", f"{p_judge_summary['overall']:.2f}"]
    ]
    print_table(headers_t4, rows_t4)

    print("\nTABLE 5: JUDGE VS. HUMAN CALIBRATION EVIDENCE")
    headers_t5 = ["Metric", "Value", "Benchmark Standard"]
    rows_t5 = [
        ["Quadratic Weighted Kappa (κ_w)", f"{calibration_report['quadratic_weighted_kappa']:.4f}", "Substantial agreement > 0.40"],
        ["Pearson Correlation (r)", f"{calibration_report['pearson_correlation']:.4f}", "Linear correlation with human scores"],
        ["Spearman Rank Correlation (ρ)", f"{calibration_report['spearman_rank_correlation']:.4f}", "Rank correlation with human scores"],
        ["Exact Agreement Rate", f"{calibration_report['exact_agreement_rate']*100:.1f}%", "Exact match on 1-5 scale"],
        ["Within-1 Agreement Rate", f"{calibration_report['within_1_agreement_rate']*100:.1f}%", "Score difference <= 1 point"],
        ["Mean Absolute Error (MAE)", f"{calibration_report['mean_absolute_error']:.3f}", "Average rating divergence"],
        ["Mean Human Rating", f"{calibration_report['mean_human_score']:.2f} / 5.0", "Ground-truth human expert rating"],
        ["Mean Judge Rating", f"{calibration_report['mean_judge_score']:.2f} / 5.0", "Rubric judge average rating"],
        ["Judge Leniency Bias (Δ)", f"{calibration_report['leniency_bias']:+.3f}", "Offset between judge and human"]
    ]
    print_table(headers_t5, rows_t5)

    total_time = time.time() - start_time
    print(f"\nEvaluation completed in {total_time:.2f} seconds.")

    full_results = {
        "execution_time_seconds": round(total_time, 2),
        "dataset_size": len(golden_set),
        "systems": {
            "baseline_1_trivial": {
                "automated_metrics": t_metrics,
                "judge_summary": t_judge_summary
            },
            "baseline_2_simple": {
                "automated_metrics": s_metrics,
                "judge_summary": s_judge_summary
            },
            "proposed_agent": {
                "automated_metrics": p_metrics,
                "judge_summary": p_judge_summary
            }
        },
        "human_judge_calibration": calibration_report
    }

    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_benchmark()
