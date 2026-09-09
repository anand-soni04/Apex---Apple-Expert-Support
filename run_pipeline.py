import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = str(Path(__file__).resolve().parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.agent import AppleSupportAgent


def display_result(res: dict, query: str):
    print("\n" + "-" * 70)
    print(f"Customer Inquiry: \"{query}\"")
    print("-" * 70)
    
    print("[1] Intent Classification:")
    print(f"  - Intent: {res['predicted_intent']}")
    print(f"  - Confidence: {res['confidence']*100:.1f}%")
    top_probs = list(res["probabilities"].items())[:3]
    prob_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in top_probs)
    print(f"  - Probabilities: {prob_str}")

    print("[2] Escalation Decision:")
    status = "ESCALATE TO HUMAN" if res["should_escalate"] else "AUTO-HANDLE"
    print(f"  - Action: {status}")
    print(f"  - Trigger: {res['escalation_reason']}")
    print(f"  - Reason: {res['stated_explanation']}")

    print("[3] Grounding Exemplars:")
    if res["grounding_exemplars"]:
        for idx, ex in enumerate(res["grounding_exemplars"][:2], 1):
            print(f"  - Match {idx} (Sim {ex['similarity_score']:.3f}): {ex['historical_customer_query'][:75]}...")
    else:
        print("  - No direct match above threshold.")

    print("[4] Draft Reply:")
    print(f"  \"{res['draft_reply']}\"")
    print("-" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Apple Support AI Agent Pipeline")
    parser.add_argument("--query", type=str, help="Single customer query to process")
    parser.add_argument("--file", type=str, help="Path to text file containing queries (one per line)")
    parser.add_argument("--provider", type=str, default="local", choices=["local", "openai", "gemini"], help="Backend provider")
    args = parser.parse_args()

    agent = AppleSupportAgent(llm_provider=args.provider)
    agent.initialize()

    if args.query:
        res = agent.process(args.query)
        display_result(res, args.query)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        for q in queries:
            res = agent.process(q)
            display_result(res, q)
    else:
        sample_queries = [
            "My iPhone battery goes from 100% to 15% in two hours after updating to iOS 11",
            "I dropped my phone in water and now the screen is flashing green",
            "My Apple ID has been locked and I'm not receiving my verification codes",
            "I was charged $49.99 on my credit card for a subscription I never purchased, give me a refund",
            "I already restarted my phone and reset network settings 3 times, WiFi is still dropping",
            "Worst support ever, fix this right now or I am buying a Samsung today"
        ]
        print("Enter a customer inquiry (or 1-6 for sample query, 'exit' to quit):")
        for idx, sq in enumerate(sample_queries, 1):
            print(f"  [{idx}] {sq}")

        while True:
            try:
                user_input = input("\nQuery: ").strip()
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    break
                if user_input.isdigit() and 1 <= int(user_input) <= len(sample_queries):
                    query_text = sample_queries[int(user_input) - 1]
                else:
                    query_text = user_input
                
                res = agent.process(query_text)
                display_result(res, query_text)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
