#!/usr/bin/env python3
"""
collect_comments.py - pull comments from public Reddit threads into a CSV ready for labeling.

Usage:
    pip install requests
    python collect_comments.py URL_OR_FILE [URL_OR_FILE ...] --out raw_comments.csv --max 400

If Reddit blocks the script (403), open the thread's .json URL in your browser, save the page
as a .json file, and pass the file path instead of the URL, e.g.:
    python collect_comments.py game_thread.json waiver.json --out raw_comments.csv

Pass several threads (morning game thread, afternoon thread, post-game / overreaction threads)
to get a better mix of discourse types. Output columns:
    id, thread, score, text, label, notes, prelabeled, permalink
Fill in `label` (and `notes` for hard cases). If you use an LLM to pre-label, mark `prelabeled`
as "yes" on those rows so you can disclose it. The notebook only needs `text` and `label`.
"""
import argparse
import csv
import json
import os
import random
import re
import time

import requests

# Reddit requires a descriptive User-Agent; generic ones get 429s fast.
UA = "takemeter-collector/0.1 (AI201 class project; by u/YOUR_REDDIT_USERNAME)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
URL_RE = re.compile(r"https?://\S+")
BOTS = {"AutoModerator"}


def thread_json_url(url):
    url = url.split("?")[0].rstrip("/")
    return url if url.endswith(".json") else url + ".json"


def get(url, params=None, tries=4):
    for i in range(tries):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:  # rate limited: back off and retry
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Still rate limited after {tries} tries: {url}")


def walk(children, out, more_ids):
    """Recursively flatten a comment tree; collect ids of collapsed 'load more' stubs."""
    for c in children:
        kind, d = c.get("kind"), c.get("data", {})
        if kind == "t1":
            out.append(d)
            replies = d.get("replies")
            if isinstance(replies, dict):
                walk(replies["data"]["children"], out, more_ids)
        elif kind == "more":
            more_ids.extend(d.get("children", []))


def fetch_more(link_id, ids, out, cap):
    """Expand collapsed comments (up to `cap` ids, 100 per request)."""
    ids = ids[:cap]
    for i in range(0, len(ids), 100):
        data = get(
            "https://www.reddit.com/api/morechildren.json",
            params={"link_id": link_id, "children": ",".join(ids[i:i + 100]),
                    "api_type": "json", "raw_json": 1},
        )
        for t in data.get("json", {}).get("data", {}).get("things", []):
            if t.get("kind") == "t1":
                out.append(t["data"])
        time.sleep(3)


def clean(body):
    body = URL_RE.sub("[link]", body)
    return re.sub(r"\s+", " ", body).strip()


def keep(d, min_words):
    body = d.get("body", "")
    if body in ("[deleted]", "[removed]") or d.get("author") in BOTS or d.get("stickied"):
        return False
    return len(clean(body).split()) >= min_words


def collect(url, expand_cap):
    if os.path.exists(url):  # a .json file saved from the browser
        with open(url, encoding="utf-8") as f:
            listing = json.load(f)
        expand_cap = 0  # collapsed comments can't be expanded offline
    else:
        listing = get(thread_json_url(url), params={"limit": 500, "sort": "top", "raw_json": 1})
    post = listing[0]["data"]["children"][0]["data"]
    comments, more_ids = [], []
    walk(listing[1]["data"]["children"], comments, more_ids)
    if expand_cap and more_ids:
        fetch_more(post["name"], more_ids, comments, expand_cap)
    return post["title"], comments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="Reddit thread URLs or saved .json files")
    ap.add_argument("--out", default="raw_comments.csv")
    ap.add_argument("--max", type=int, default=400, help="max rows to write (random sample)")
    ap.add_argument("--min-words", type=int, default=3,
                    help="drop comments shorter than this (note: filtering short posts removes "
                         "many reactions; document whatever you choose)")
    ap.add_argument("--expand", type=int, default=500,
                    help="how many collapsed comments to expand per thread (0 = none)")
    ap.add_argument("--replies-only", action="store_true",
                    help="keep only replies to other comments (skips top-level questions in "
                         "waiver/advice threads, where the takes are in the answers)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, seen = [], set()
    for url in args.urls:
        title, comments = collect(url, args.expand)
        print(f"{title}: {len(comments)} raw comments")
        for d in comments:
            if not keep(d, args.min_words):
                continue
            if args.replies_only and not str(d.get("parent_id", "")).startswith("t1_"):
                continue
            text = clean(d["body"])
            key = text.lower()
            if key in seen:  # drop exact duplicates (copypasta, repeated "LETS GO")
                continue
            seen.add(key)
            rows.append({
                "id": d["id"], "thread": title, "score": d.get("score", 0),
                "is_reply": str(d.get("parent_id", "")).startswith("t1_"), "text": text,
                "label": "", "notes": "", "prelabeled": "",
                "permalink": "https://www.reddit.com" + d.get("permalink", ""),
            })
        time.sleep(3)

    # Shuffle so you don't label in thread order (order effects bias annotation).
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["text", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
