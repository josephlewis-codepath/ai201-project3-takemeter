#!/usr/bin/env python3
"""
TakeMeter — r/fantasyfootball corpus collector.  v2

Reddit now returns 403 for unauthenticated .json requests, so there are two
supported routes. Both produce the same cleaned, unlabeled CSV.

ROUTE A — OAuth (preferred; one-time 2-minute setup, then it just works)
-----------------------------------------------------------------------
1. Go to https://www.reddit.com/prefs/apps  ->  "create another app..."
2. Pick type "script". Name it anything. redirect uri: http://localhost:8080
3. Create it. The client id is the string under the app name; the secret is
   the field labeled "secret".
4. Export them in your shell (do NOT paste them into a chat or commit them):

       export REDDIT_CLIENT_ID='...'
       export REDDIT_CLIENT_SECRET='...'

5. python3 scripts/collect_reddit.py --out data/raw_unlabeled.csv --target 400

ROUTE B — saved JSON from your browser (zero setup, no credentials)
-------------------------------------------------------------------
Your logged-in browser is not blocked. Open each of these, then File > Save Page As
into a folder named `saved/` (keep the .json extension):

    https://www.reddit.com/r/fantasyfootball/comments.json?limit=100
    https://www.reddit.com/r/fantasyfootball/top.json?t=week&limit=100
    https://www.reddit.com/r/fantasyfootball/top.json?t=month&limit=100
    https://www.reddit.com/r/fantasyfootball/hot.json?limit=100

plus 6-10 individual threads (append .json to any thread URL), mixing game
threads, post-game threads, rant threads, and long top-of-month posts.

    python3 scripts/collect_reddit.py --files "saved/*.json" --out data/raw_unlabeled.csv

The file parser accepts any shape Reddit serves — subreddit listings, the sub-wide
comment stream, or a thread page — so it does not matter which you save.

No third-party dependencies. Python 3.8+.
"""

import argparse
import csv
import glob
import html
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "python:takemeter-ai201:0.3 (AI201 coursework)"
OAUTH_BASE = "https://oauth.reddit.com"

# Sampling frame from planning.md section 4. Deliberately mixed so the corpus is
# not 80% game-thread venting.
LISTINGS = [
    ("/r/fantasyfootball/top", {"t": "week", "limit": 100}, 14, "top_week"),
    ("/r/fantasyfootball/top", {"t": "month", "limit": 100}, 12, "top_month"),
    ("/r/fantasyfootball/hot", {"limit": 100}, 12, "hot"),
    ("/r/fantasyfootball/new", {"limit": 100}, 8, "new"),
]
COMMENT_STREAM_PAGES = 4

BOT_AUTHORS = {
    "automoderator", "remindmebot", "sneakpeekbot", "wikitextbot",
    "b0trank", "converter-bot", "ff-bot", "fantasyfootball-modteam",
}

# Bare advice requests are out of scope per planning.md section 3.
ADVICE_PATTERNS = [
    re.compile(r"^\s*\w[\w\s.'-]{0,40}\s+(or|vs\.?)\s+\w[\w\s.'-]{0,40}\s*\??\s*$", re.I),
    re.compile(r"\b(start|sit|flex)\s*\??\s*$", re.I),
    re.compile(r"^\s*(who|which)\s+(do i|should i|would you)\b", re.I),
    re.compile(r"\b(thoughts|worth it|fair|yay or nay)\s*\?\s*$", re.I),
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"/?u/[A-Za-z0-9_-]+")
SUB_RE = re.compile(r"/?r/[A-Za-z0-9_]+")
WS_RE = re.compile(r"\s+")
QUOTE_RE = re.compile(r"^\s*(?:&gt;|>).*$", re.M)


# --------------------------------------------------------------------------
# cleaning (planning.md section 4)
# --------------------------------------------------------------------------

def clean(text):
    if not text:
        return ""
    t = html.unescape(text)
    t = QUOTE_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = USER_RE.sub(" ", t)
    t = SUB_RE.sub(" ", t)
    t = t.replace("&amp;", "&").replace("​", " ")
    return WS_RE.sub(" ", t).strip()


def usable(text, author):
    if not text or text in ("[deleted]", "[removed]", "[deleted by user]"):
        return False
    if author and str(author).lower() in BOT_AUTHORS:
        return False
    n = len(text.split())
    if n < 4 or n > 120:
        return False
    return not any(p.search(text) for p in ADVICE_PATTERNS)


def truncate(text, limit=120):
    return " ".join(text.split()[:limit])


# --------------------------------------------------------------------------
# universal extractor: walks ANY Reddit JSON shape, pulling t1 bodies + t3 selftexts
# --------------------------------------------------------------------------

def extract(node, out, tag, seen_ids=None, depth=0):
    if seen_ids is None:
        seen_ids = set()
    if depth > 40:
        return
    if isinstance(node, list):
        for item in node:
            extract(item, out, tag, seen_ids, depth + 1)
        return
    if not isinstance(node, dict):
        return

    kind = node.get("kind")
    data = node.get("data")

    if kind in ("t1", "t3") and isinstance(data, dict):
        rid = data.get("id")
        if not rid or rid not in seen_ids:
            if rid:
                seen_ids.add(rid)
            body = data.get("body") if kind == "t1" else data.get("selftext")
            text = clean(body or "")
            if usable(text, data.get("author", "")):
                out.append({
                    "text": truncate(text),
                    "score": data.get("score", 0),
                    "source_thread": tag if kind == "t1" else f"{tag}_selftext",
                    "permalink": data.get("permalink", ""),
                })
        for key in ("replies", "children"):
            if key in data:
                extract(data[key], out, tag, seen_ids, depth + 1)
        return

    if isinstance(data, dict):
        extract(data.get("children", []), out, tag, seen_ids, depth + 1)
        return

    for v in node.values():
        if isinstance(v, (dict, list)):
            extract(v, out, tag, seen_ids, depth + 1)


# --------------------------------------------------------------------------
# Route A: OAuth
# --------------------------------------------------------------------------

def get_token():
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        return None
    import base64
    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token", data=body,
        headers={"Authorization": f"Basic {auth}", "User-Agent": UA,
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            tok = json.loads(r.read().decode())["access_token"]
        print("OAuth token acquired.")
        return tok
    except urllib.error.HTTPError as e:
        print(f"[oauth {e.code}] could not get a token. Check the id/secret and that "
              f"the app type is 'script'.", file=sys.stderr)
        return None
    except Exception as e:  # noqa: BLE001
        print(f"[oauth err] {type(e).__name__}: {e}", file=sys.stderr)
        return None


def api(token, path, params=None, sleep=1.0, tries=4):
    qs = "?" + urllib.parse.urlencode(params) if params else ""
    url = OAUTH_BASE + path + qs
    for attempt in range(tries):
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            time.sleep(sleep)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = sleep * (2 ** attempt) + random.random()
                print(f"  [{e.code}] backing off {wait:.1f}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  [http {e.code}] {url}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(sleep * (attempt + 1))
    return None


def collect_oauth(token, target, sleep, add):
    print("Pulling sub-wide recent comments ...")
    after = None
    for page in range(COMMENT_STREAM_PAGES):
        params = {"limit": 100}
        if after:
            params["after"] = after
        d = api(token, "/r/fantasyfootball/comments", params, sleep)
        if not d:
            break
        got = []
        extract(d, got, "recent_comment_stream")
        print(f"  page {page + 1}: +{add(got)}")
        after = d.get("data", {}).get("after")
        if not after:
            break

    for path, params, n_posts, tag in LISTINGS:
        if add(None) >= target:
            break
        print(f"Pulling {tag} ...")
        d = api(token, path, params, sleep)
        if not d:
            continue
        got = []
        extract(d, got, tag)
        print(f"  selftexts: +{add(got)}")

        posts = [c.get("data", {}) for c in d.get("data", {}).get("children", [])]
        posts = [p for p in posts if not p.get("stickied") and p.get("permalink")]
        for p in posts[:n_posts]:
            if add(None) >= target:
                break
            d2 = api(token, p["permalink"].rstrip("/"),
                     {"limit": 200, "sort": "top"}, sleep)
            if not d2:
                continue
            got = []
            extract(d2, got, tag)
            print(f"  {p['permalink'][:58]}: +{add(got)}")


# --------------------------------------------------------------------------
# Route B: saved files
# --------------------------------------------------------------------------

def collect_files(paths, add):
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        tag = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:28]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        got = []
        extract(data, got, tag or "saved")
        print(f"  {os.path.basename(path)}: +{add(got)}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_unlabeled.csv")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--files", nargs="*", default=None,
                    help="saved .json files from your browser (Route B)")
    args = ap.parse_args()

    rows, seen = [], set()

    def add(items):
        """Add deduped items, return how many were added.
        Called with None it reports the running total instead."""
        if items is None:
            return len(rows)
        before = len(rows)
        for it in items:
            key = re.sub(r"[^a-z0-9]", "", it["text"].lower())[:180]
            if key and key not in seen:
                seen.add(key)
                rows.append(it)
        return len(rows) - before

    if args.files:
        paths = []
        for pat in args.files:
            paths.extend(sorted(glob.glob(pat)) or [pat])
        print(f"Reading {len(paths)} saved file(s) ...")
        collect_files(paths, add)
    else:
        token = get_token()
        if not token:
            print("\nNo OAuth credentials found (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET),\n"
                  "and Reddit blocks unauthenticated requests. See the header of this file\n"
                  "for Route A (OAuth setup) or Route B (--files with saved JSON).",
                  file=sys.stderr)
            sys.exit(1)
        collect_oauth(token, args.target, args.sleep, add)

    if not rows:
        print("\nNothing collected.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    random.seed(17)
    random.shuffle(rows)
    rows = rows[: args.target]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "notes", "prelabel",
                                          "source_thread", "score", "permalink"])
        w.writeheader()
        for r in rows:
            w.writerow({"text": r["text"], "label": "", "notes": "", "prelabel": "",
                        "source_thread": r["source_thread"], "score": r["score"],
                        "permalink": r["permalink"]})

    print(f"\nWrote {len(rows)} unlabeled examples to {args.out}")
    from collections import Counter
    for src, n in Counter(r["source_thread"] for r in rows).most_common():
        print(f"  {src:30s} {n}")
    print("\nNext: fill the `label` column with analysis | hot_take | reaction.")


if __name__ == "__main__":
    main()
