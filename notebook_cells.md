# Colab cells for TakeMeter

Paste these into the corresponding sections of your copy of the starter notebook
(https://colab.research.google.com/drive/17rhO2vDojhJTnsaM5riGDD9gCviQv4vL).
Runtime → Change runtime type → **T4 GPU** before running anything.

---

## Section 1 — label map

```python
LABEL_MAP = {
    "analysis":  0,
    "hot_take":  1,
    "reaction":  2,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = [ID2LABEL[i] for i in range(len(ID2LABEL))]
```

Your CSV's `label` column must contain exactly these three strings. Quick guard to run
right after upload — it catches typos and stray whitespace before they become a silent
class-imbalance bug:

```python
import pandas as pd
df = pd.read_csv(CSV_PATH)                      # whatever the notebook names it
df["label"] = df["label"].astype(str).str.strip().str.lower()
bad = sorted(set(df["label"]) - set(LABEL_MAP))
assert not bad, f"unexpected labels: {bad}"
print(df["label"].value_counts())
print("total:", len(df))
top_share = df["label"].value_counts(normalize=True).max()
print(f"largest class share: {top_share:.1%}  (rubric cap: 70%)")
assert top_share <= 0.70, "rebalance before training"
```

---

## Section 3 — training arguments

The starter defaults are 3 epochs / lr 2e-5 / batch 16. Start there, then use this
block, which is the version I'd defend in the README:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./takemeter",
    num_train_epochs=6,                 # see note below
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",              # older transformers: evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    logging_steps=10,
    seed=42,
    report_to="none",
)
```

**The hyperparameter decision to write up: epochs 3 → 6 with best-checkpoint selection.**

Reasoning to verify against your own run, not to copy blindly: with ~180 training
examples and batch size 16, 3 epochs is only ~33 optimizer steps. That is not enough
for DistilBERT's classification head to converge on a task where the signal is argument
structure rather than vocabulary — you will typically see training loss still falling
when it stops. Running 6 epochs with `load_best_model_at_end` and macro-F1 as the
selection metric gets the extra steps without eating the overfit: if validation macro-F1
peaks at epoch 4 and degrades after, the checkpoint you keep is epoch 4's.

**Record the actual per-epoch numbers from your run** and quote them in the README —
the rubric wants an observation, and "val macro-F1 went 0.41 → 0.58 → 0.66 → 0.71 →
0.70 → 0.68, so I kept epoch 4" is worth far more than the reasoning above.

Learning rate 3e-5 over 2e-5 is a smaller call, justified by the same step-count
argument. If you see the loss spiking or validation F1 thrashing between epochs, go
back to 2e-5 and say so.

Make sure the trainer computes macro-F1, since `metric_for_best_model` depends on it:

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
```

---

## Section 5 — Groq zero-shot baseline

### The prompt

Kept strictly zero-shot: label definitions and the decision procedure, **no labeled
examples**. Few-shot examples would be task-specific supervision drawn from the same
distribution as the training set, which would make "fine-tuning vs. baseline" a
comparison between two trained systems rather than trained vs. untrained. Worth one
sentence in the README — it's a fairness decision, not an oversight.

```python
SYSTEM_PROMPT = """You classify comments from the subreddit r/fantasyfootball by how the post relates a claim to its support. You output exactly one label and nothing else."""

USER_PROMPT_TEMPLATE = """Classify the r/fantasyfootball post below into exactly one of three labels.

analysis
The post argues toward an evaluative or predictive conclusion using at least one specific, checkable piece of support - a statistic, a usage figure (snaps, targets, routes, touches), a named matchup or scheme detail, or a historical comparison with particulars - and that support does real work: the conclusion would lose its footing if you deleted it.

hot_take
The post states a confident evaluative claim about a player, team, or strategy without genuinely supporting it. Support is absent, vague ("he's been trash", "he doesn't look right"), or decorative - a figure picked for rhetorical effect that does not actually establish the claim being made.

reaction
The post is an immediate emotional response to something that just happened - a touchdown, an injury, a blown lead, a lineup decision that went wrong. It expresses feeling (celebration, anguish, rage, disbelief, humor) rather than making or defending a claim about player value that could still be evaluated next week.

Apply this decision procedure in order:
1. Is there a durable evaluative claim - something that could be judged right or wrong next week? If no, the label is reaction.
2. If yes, is there specific checkable support that the claim actually rests on? If yes, the label is analysis. If the support is absent, vague, or decorative, the label is hot_take.
3. If a post is emotionally charged AND carries real supporting evidence, evidence wins: label it analysis.

Two rules for hard cases:
- A number only counts as support if it is used toward a claim, not as an intensifier for a complaint about an outcome. "Three targets. THREE." in a post about losing a matchup is venting, not evidence.
- An explicit generalization ("every week", "all season", "he always") makes a post a durable claim even if it cites only one game, which puts it in hot_take rather than reaction.

POST:
\"\"\"{post}\"\"\"

Respond with exactly one word, lowercase, no punctuation, no explanation: analysis, hot_take, or reaction."""
```

### Calling it

```python
import os, re, time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])   # from Colab Secrets
BASELINE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VALID = set(LABEL_MAP)

def baseline_classify(post, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=BASELINE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(post=post)},
                ],
                temperature=0,          # deterministic: this is a measurement, not generation
                max_tokens=8,           # the answer is one word; cap it so it can't ramble
            )
            raw = resp.choices[0].message.content.strip().lower()
            raw = re.sub(r"[^a-z_]", "", raw)
            if raw in VALID:
                return raw, True
            for lab in VALID:                      # salvage "label: hot_take" style replies
                if lab in raw:
                    return lab, True
            return raw, False                      # unparseable; counted, not silently dropped
        except Exception as e:
            if attempt == retries - 1:
                print("giving up on one example:", type(e).__name__, e)
                return "", False
            time.sleep(2 ** attempt)
```

`temperature=0` matters: at the default temperature the same test example can get
different labels on different runs, and your baseline number stops being reproducible.

Track unparseable responses explicitly rather than dropping them — the milestone asks
you to flag them, and >10% unparseable means the prompt needs a clearer output
instruction, not a better parser.

---

## Section 4 / 6 — sample classifications with confidence

You need this for the README's Sample Classifications table **and** for the demo video,
where label *and confidence* have to be visible on screen.

```python
import torch, torch.nn.functional as F

def predict(texts, model, tokenizer, device="cuda"):
    if isinstance(texts, str):
        texts = [texts]
    model.eval().to(device)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1).cpu()
    out = []
    for t, p in zip(texts, probs):
        idx = int(p.argmax())
        out.append({
            "text": t,
            "predicted": ID2LABEL[idx],
            "confidence": float(p[idx]),
            "all_probs": {ID2LABEL[i]: round(float(v), 3) for i, v in enumerate(p)},
        })
    return out

for r in predict(demo_posts, model, tokenizer):     # demo_posts = list of 3-5 strings
    print(f"{r['predicted']:<10} {r['confidence']:.1%}   {r['text'][:80]}")
    print(f"           {r['all_probs']}\n")
```

Pick your demo posts so one is a clear correct prediction you can narrate and one is a
failure you can explain — grab the failure from your actual test-set errors rather than
hunting for one live on camera.

### Markdown confusion matrix (the graded version)

The committed PNG is supplementary; the README needs the text table.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
hdr = "| true \\ pred | " + " | ".join(LABEL_NAMES) + " | total |"
print(hdr)
print("|" + "---|" * (len(LABEL_NAMES) + 2))
for i, name in enumerate(LABEL_NAMES):
    print(f"| **{name}** | " + " | ".join(str(x) for x in cm[i]) + f" | {cm[i].sum()} |")
```

---

## Stretch: confidence calibration

```python
import numpy as np, pandas as pd

conf = probs.max(axis=1)                 # max softmax prob per test example
correct = (y_pred == y_true).astype(int)
bins = pd.cut(conf, [0, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
cal = pd.DataFrame({"conf": conf, "correct": correct, "bin": bins})
print(cal.groupby("bin", observed=True)
         .agg(n=("correct", "size"), accuracy=("correct", "mean"),
              mean_conf=("conf", "mean")))
```

Read it honestly: if the 0.9–1.0 bin is not more accurate than the 0.6–0.7 bin,
your confidences are not meaningful and you should say so. DistilBERT fine-tuned on a
small set is usually **overconfident** — mean confidence well above bin accuracy — and
reporting that is worth more than pretending otherwise.

---

## Stretch: deployed interface (Gradio, runs inside Colab)

```python
!pip -q install gradio
import gradio as gr

def classify_ui(post):
    if not post.strip():
        return {}
    r = predict(post, model, tokenizer)[0]
    return r["all_probs"]

gr.Interface(
    fn=classify_ui,
    inputs=gr.Textbox(lines=5, label="Paste an r/fantasyfootball post"),
    outputs=gr.Label(num_top_classes=3, label="TakeMeter"),
    title="TakeMeter",
    description="analysis / hot_take / reaction — fine-tuned DistilBERT on r/fantasyfootball.",
    examples=[
        "Bijan ran 71% of routes last week vs 44% in weeks 1-3 and Allgeier's snap share fell off a cliff after the bye. This is a real usage change.",
        "Started him over the obvious play and lost by 2. Three targets. THREE.",
        "He's the overall RB1 rest of season, take it to the bank.",
    ],
).launch(share=True)
```

`share=True` gives a public link that works in the demo video. Commit this cell as
`app.py` in the repo and document `python app.py` in the README to claim the stretch
point.
