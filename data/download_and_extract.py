import sys
import io
import csv
import json
import re
import html
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
OUTPUT_CORPUS = Path(__file__).resolve().parent / "twcs_apple_support_corpus.jsonl"
DEFAULT_BYTE_RANGE = 35 * 1024 * 1024


def clean_tweet_text(text: str) -> str:
    if not text:
        return ""
    cleaned = html.unescape(text)
    cleaned = cleaned.replace("\ufffd", "'").replace("\ufe0f", "")
    cleaned = re.sub(r"@\d{4,7}\b", "@customer", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_apple_conversations(byte_limit: int = DEFAULT_BYTE_RANGE) -> list[dict]:
    print(f"Streaming {byte_limit // (1024 * 1024)} MB from {DATASET_URL}...")
    headers = {"Range": f"bytes=0-{byte_limit}", "User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(DATASET_URL, headers=headers)
    
    with urllib.request.urlopen(req) as res:
        raw_bytes = res.read()

    text_data = raw_bytes.decode("utf-8", errors="ignore")
    last_newline = text_data.rfind("\n")
    if last_newline > 0:
        text_data = text_data[:last_newline]

    reader = csv.DictReader(io.StringIO(text_data))
    tweets = {}
    apple_replies = []

    for row in reader:
        t_id = row.get("tweet_id", "").strip()
        if not t_id:
            continue
        tweets[t_id] = row
        if row.get("author_id") == "AppleSupport" and row.get("in_response_to_tweet_id"):
            apple_replies.append(row)

    pairs = []
    seen_customer_texts = set()

    for agent_tweet in apple_replies:
        parent_id = agent_tweet.get("in_response_to_tweet_id", "").strip()
        if parent_id in tweets:
            customer_tweet = tweets[parent_id]
            if customer_tweet.get("inbound") == "True":
                c_text = customer_tweet.get("text", "")
                a_text = agent_tweet.get("text", "")
                
                clean_c = clean_tweet_text(c_text)
                clean_a = clean_tweet_text(a_text)

                if len(clean_c) < 10 or clean_c in seen_customer_texts:
                    continue
                seen_customer_texts.add(clean_c)

                pairs.append({
                    "customer_tweet_id": customer_tweet.get("tweet_id"),
                    "customer_text": c_text,
                    "clean_customer_text": clean_c,
                    "agent_tweet_id": agent_tweet.get("tweet_id"),
                    "agent_text": a_text,
                    "clean_agent_text": clean_a,
                    "created_at": customer_tweet.get("created_at"),
                })

    print(f"Constructed {len(pairs)} conversation pairs.")
    return pairs


def save_corpus(pairs: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    pairs = extract_apple_conversations()
    save_corpus(pairs, OUTPUT_CORPUS)


if __name__ == "__main__":
    main()
