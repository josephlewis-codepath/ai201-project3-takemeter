# TakeMeter — classifying discourse quality in r/fantasyfootball

A fine-tuned DistilBERT classifier that sorts r/fantasyfootball posts into **analysis**,
**hot_take**, and **reaction** — three ways a post can relate a claim to its support —
benchmarked against a zero-shot Llama-4-Scout baseline on the same held-out test set.

**Joseph Lewis · AI201 Project 3**

📹 **Demo video:** `‹PASTE LINK›`
📓 **Colab notebook:** https://colab.research.google.com/drive/17rhO2vDojhJTnsaM5riGDD9gCviQv4vL
📄 **Design doc:** [`planning.md`](planning.md) — written before data collection
📊 **Dataset:** [`data/takemeter_labeled.csv`](data/takemeter_labeled.csv)

> **⚠️ Placeholders.** Everything written as `0.__`, `‹...›`, or in a `FILL AFTER RUN`
> callout comes from the Colab run. Search the file for `‹` and `0.__` to find them all.

---

## Evaluation report summary

> **FILL AFTER RUN** — from `evaluation_results.json`.

| | Fine-tuned DistilBERT | Zero-shot Llama-4-Scout | Δ |
|---|---|---|---|
| Accuracy | `0.__` | `0.__` | `+0.__` |
| **Macro-F1** | `0.__` | `0.__` | `+0.__` |
| `analysis` F1 | `0.__` | `0.__` | `+0.__` |
| `hot_take` F1 | `0.__` | `0.__` | `+0.__` |
| `reaction` F1 | `0.__` | `0.__` | `+0.__` |

**Did it clear the bar set in `planning.md` §6?** `‹yes / no — and which of the four
criteria it missed›`

Targets were: macro-F1 ≥ 0.70, no class F1 below 0.60, `analysis` recall ≥ 0.70, and at
least +0.10 macro-F1 over the baseline.

---

## 1. Community

**r/fantasyfootball.**

I picked it because it's one of the few large subreddits where the *same event* reliably
produces all three kinds of discourse within minutes, in the same thread. A running back
gets six carries in the first half, and the replies split into people pulling up his
snap share, people declaring him done for the season, and people screaming in all caps
because they benched the alternative. Topic held constant, manner of argument varying —
which is exactly what makes this a real classification problem rather than a keyword
lookup.

Two things make it a good fit beyond that. **Evidence here is concrete and checkable**:
fantasy discourse runs on figures that actually exist — snap percentage, target share,
route participation, red zone touches, opponent run-defense rank — so "backed by
evidence" is a property I can verify while annotating instead of a vibe I have to
intuit. And **the distinction is already a community norm**: users police take quality
themselves, with "source: trust me bro," `!RemindMe 1 week`, and the weekly ritual of
resurfacing last month's confident bad takes. I'm formalizing something the sub already
does informally.

The consequence for the model is the interesting part: all three labels share the same
vocabulary — player names, team names, "targets," "start" — so the classifier can't win
by learning topic words. It has to pick up something about argument structure. Whether it
actually did is the subject of §8.

---

## 2. Label taxonomy

One axis: **how the post relates a claim to its support.**

### `analysis`

The post argues toward an evaluative or predictive conclusion using at least one
specific, checkable piece of support — a statistic, a usage figure (snaps, targets,
routes, touches), a named matchup or scheme detail, or a historical comparison with
particulars — and that support does real work, meaning the conclusion would lose its
footing if you deleted it.

> "Bijan ran 71% of routes last week against 44% in weeks 1–3, and Allgeier's snap share
> fell off a cliff coming out of the bye. That's a real usage change, not a one-week
> blip. I'm buying at whatever the cost is."

> "Kittle was limited Wednesday and Thursday and full on Friday. Historically 49ers tight
> ends with that practice pattern have played around 85% of snaps that week. I'm starting
> him over the safer floor guys."

### `hot_take`

The post states a confident evaluative claim about a player, team, or strategy without
genuinely supporting it. Support is absent, vague ("he's been trash," "he doesn't look
right"), or decorative — a figure picked for rhetorical effect that doesn't actually
establish the claim being made.

> "Bijan is the overall RB1 rest of season. Take it to the bank."

> "Puka's target share is insane but Stafford throws him into coverage every other play.
> Dude is cooked in this offense."

The second one is the instructive example: it *contains* a real usage concept, but the
evidence points against the conclusion, and "throws him into coverage every other play"
is an unfalsifiable impression. The number is decoration.

### `reaction`

The post is an immediate emotional response to something that just happened — a
touchdown, an injury, a blown lead, a lineup decision that went wrong. It expresses
feeling rather than making or defending a claim about player value that could still be
evaluated next week.

> "WHY DID I BENCH HIM. Why. I looked at that lineup for forty minutes."

> "lmaooo my opponent started a guy on bye and still beat me by 40. I'm done."

### The decision procedure

Applied in this order to every example, which is what makes the labels mutually
exclusive by construction:

1. **Is there a durable evaluative claim** — something judgeable as right or wrong next
   week? No → `reaction`.
2. **If yes, is there specific checkable support the claim actually rests on?** Yes →
   `analysis`. Absent, vague, or decorative → `hot_take`.
3. **Emotionally charged *and* carrying real evidence** → `analysis`. A well-argued rant
   is still an argument.

Two rules for the hard cases, both derived from stress-testing before annotation
(`planning.md` §7a):

- **A number only counts as support if it's used toward a claim**, not as an intensifier
  for a complaint about an outcome. "Three targets. THREE." in a post about losing a
  matchup is venting.
- **An explicit generalization** ("every week," "all season," "he always") makes a post a
  durable claim even on one game's evidence → `hot_take` rather than `reaction`.

---

## 3. Data

### Source

Public r/fantasyfootball comments and self-post bodies, collected through Reddit's
unauthenticated `.json` endpoints with [`scripts/collect_reddit.py`](scripts/collect_reddit.py).
No authentication, no private content, no scraping of anything behind a login.

Sampling was deliberately spread across five thread types, because pulling only from
game threads would have produced an 80% `reaction` dataset and a model that learned
nothing:

| Thread type | Why included |
|---|---|
| Sunday game / post-game threads | Highest emotional density — the `reaction` supply |
| Weekly rant and "who are you dropping" threads | Complaints that generalize — the `reaction`/`hot_take` boundary |
| Top posts of week and month | Longer, argued posts — the `analysis` supply |
| Player-specific discussion posts | Mixed, closest to the real feed |
| Sub-wide recent comment stream | Unfiltered realism, no cherry-picking |

Cleaning, applied before any label was assigned: URLs, `/u/` handles, `/r/` references
and quoted parent comments stripped; comments under 4 words dropped (no signal for any
label); posts truncated at 120 words; bots (AutoModerator, RemindMeBot, etc.), deleted
and removed bodies dropped; near-duplicates removed on normalized text; bare advice
requests ("Gibbs or Achane?") filtered out as out of scope per `planning.md` §3.

`‹N›` raw examples were collected and `‹N›` survived cleaning and annotation.

### Labeling process

1. **Pre-labeling.** Batches of ~25 examples were passed to Claude with the §2
   definitions and decision procedure verbatim, one label per post, no explanation. The
   suggestion was written to a `prelabel` column.
2. **Human review of every example.** I read each post and either confirmed or overrode
   the suggestion, working from the decision procedure rather than from the suggested
   label. Overriding forces you to articulate *why*, which is why this is faster and
   sharper than labeling cold — the task becomes auditing rather than generating.
3. **Notes on anything that gave me pause**, recorded in a `notes` column as I went
   rather than reconstructed afterward.
4. **Distribution check** against the 20% floor and 70% ceiling, followed by targeted
   collection for any thin class rather than reweighting.

**Override rate: `‹X›` of `‹N›` pre-labels changed (`‹__›%`).** `‹One sentence on where
the disagreements clustered — if they concentrated on one label pair, say which.›`

This is disclosed again in §10.

### Label distribution

> **FILL AFTER ANNOTATION**

| Label | Count | Share |
|---|---|---|
| `analysis` | `‹N›` | `__%` |
| `hot_take` | `‹N›` | `__%` |
| `reaction` | `‹N›` | `__%` |
| **Total** | **`‹N›`** | 100% |

Largest class is `__%`, inside the rubric's 70% ceiling and above the 20% floor I set
for myself. Splits are 70/15/15, produced by the notebook: `‹N›` train / `‹N›` val /
`‹N›` test.

### Three examples that were genuinely hard to label

**1. The stat-flavored rant.**

> `‹paste the real example from your notes›`
> *(e.g. "Started Kupp over Nacua and lost by 2. Three targets. THREE. In a game they
> threw it 48 times.")*

Could be `analysis` (it cites specific figures) or `reaction` (it's venting about a
result). **Decided: `reaction`.** The numbers are intensifiers for a complaint about an
outcome, not support for a claim about what happens next. The tell is that you can't
state what the post is *arguing* — delete the numbers and nothing is lost except
emphasis. Had it added "that's the third straight week under 5 targets while Nacua's
route share climbs, he's droppable," the same figures would be doing real work and it
would be `analysis`.

**2. The generalized complaint.**

> `‹paste the real example›`
> *(e.g. "8 targets, 2 catches. Every week.")*

Could be `reaction` (short, frustrated, tied to a game) or `hot_take` ("every week" is a
pattern claim). **Decided: `hot_take`.** The generalization makes the claim durable — it
asserts something about the player that's still evaluable next Sunday — while the
support is a single game's box score. That's exactly the "decorative evidence" case.
Without "every week" I'd have labeled it `reaction`.

**3. The reasoned request.**

> `‹paste the real example›`
> *(e.g. a trade-eval post that lays out both rosters in detail and ends with "Thoughts?")*

Specific, well-reasoned, full of real information — and not a claim. **Decided: out of
scope, excluded from the dataset.** A post whose primary speech act is asking for input
sits on a different axis from the one I'm measuring; including it would have forced a
fourth label or a catch-all bucket. This is a scoping decision and I'd rather state it
than pretend the taxonomy is universal: TakeMeter classifies discourse, not transactions,
and the sub contains both. Filtering is applied at collection time, so the ~90%
exhaustiveness requirement holds *within my sampling frame*.

`‹Add any further cases from your notes column — the rubric wants at least 3, but the
honest ones you actually hit are more convincing than these three alone.›`

---

## 4. Fine-tuning

**Base model:** `distilbert-base-uncased` (HuggingFace), with a 3-class sequence
classification head.
**Platform:** Google Colab, free T4 GPU.
**Libraries:** `transformers`, `datasets`, `scikit-learn`.
**Tokenization:** max length 256, truncation + dynamic padding.
**Split:** 70/15/15 train/val/test, produced by the notebook from one CSV.

### Training configuration

| | Value |
|---|---|
| Epochs | 6 (best checkpoint kept) |
| Learning rate | 3e-5 |
| Batch size | 16 train / 32 eval |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Checkpoint selection | best validation **macro-F1** |
| Seed | 42 |

### The hyperparameter decision: 3 epochs → 6, with best-checkpoint selection

The starter default is 3 epochs. With ~`‹N›` training examples at batch size 16, that's
roughly `‹N›` optimizer steps — not enough for the classification head to converge on a
task where the signal is argument structure rather than vocabulary. `‹Describe what you
actually observed: was training loss still falling at epoch 3?›`

Running 6 epochs with `load_best_model_at_end=True` and macro-F1 as the selection metric
buys the extra steps without paying for the overfit, because the checkpoint that gets
kept is the one that peaked on validation rather than the last one.

**Observed per-epoch validation macro-F1:**

> **FILL AFTER RUN**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Val macro-F1 | `0.__` | `0.__` | `0.__` | `0.__` | `0.__` | `0.__` |

Kept epoch `‹N›`. `‹One sentence on what the curve shows — did it plateau, peak and
decline, or never stop improving? "Never stopped improving" would mean 6 epochs was also
too few, which is worth saying.›`

Learning rate went 2e-5 → 3e-5 on the same step-count reasoning; it's the smaller of the
two calls. `‹Note if you saw instability and reverted.›`

---

## 5. Baseline

**Model:** `meta-llama/llama-4-scout-17b-16e-instruct` via Groq, zero-shot.
**Temperature:** 0 — this is a measurement, so the same test example has to get the same
label every run.
**Max tokens:** 8, since the expected answer is one word.

### How results were collected

Every example in the **same held-out test set** the fine-tuned model is evaluated on was
sent individually with the prompt below. Responses were lowercased, stripped to
`[a-z_]`, and matched against the three label names, with a fallback that scans for a
label name inside a longer reply (to salvage "label: hot_take"). Unparseable responses
were **counted, not dropped** — `‹N›` of `‹N›` test examples (`‹__›%`) failed to parse.

The baseline was run in Milestone 4, **before** the model was fine-tuned, so there was no
opportunity to tune the prompt against a known target.

### The prompt

The full label definitions and decision procedure from §2, verbatim, plus the two
hard-case rules, ending with:

> Respond with exactly one word, lowercase, no punctuation, no explanation: analysis,
> hot_take, or reaction.

The complete prompt is in [`notebook_cells.md`](notebook_cells.md) §5.

**Deliberately zero-shot, no few-shot examples.** Few-shot examples would be
task-specific supervision drawn from the same distribution as the training set, which
would turn "fine-tuned vs. baseline" into a comparison between two trained systems. The
baseline is supposed to answer "how well does a general model do with nothing but my
definitions," and adding examples would stop it answering that.

### Pre-training hypothesis

Written before fine-tuning, per Milestone 4: `‹what you predicted the baseline would
struggle with — mine was that it would over-predict analysis on any post containing a
digit, and under-predict reaction on posts that are angry but literate›`.

**Held up?** `‹FILL AFTER RUN›`

---

## 6. Results

> **FILL AFTER RUN** — all numbers from `evaluation_results.json`.

### Overall

| Model | Accuracy | Macro-F1 | Weighted F1 |
|---|---|---|---|
| Fine-tuned DistilBERT | `0.__` | `0.__` | `0.__` |
| Zero-shot Llama-4-Scout | `0.__` | `0.__` | `0.__` |
| *Majority-class floor* | `0.__` | `0.__` | — |

The majority-class row is there as a sanity check: a model that always predicts the
largest class scores that accuracy while learning nothing. Anything close to it is not a
result.

### Per-class — fine-tuned

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | `0.__` | `0.__` | `0.__` | `‹N›` |
| `hot_take` | `0.__` | `0.__` | `0.__` | `‹N›` |
| `reaction` | `0.__` | `0.__` | `0.__` | `‹N›` |

### Per-class — zero-shot baseline

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | `0.__` | `0.__` | `0.__` | `‹N›` |
| `hot_take` | `0.__` | `0.__` | `0.__` | `‹N›` |
| `reaction` | `0.__` | `0.__` | `0.__` | `‹N›` |

### Confusion matrix — fine-tuned model

Rows are true labels, columns are predictions. (`confusion_matrix.png` is a
supplementary copy of this same table.)

| true \ pred | analysis | hot_take | reaction | total |
|---|---|---|---|---|
| **analysis** | `_` | `_` | `_` | `_` |
| **hot_take** | `_` | `_` | `_` | `_` |
| **reaction** | `_` | `_` | `_` | `_` |

`‹Read it directionally in one or two sentences. Which single off-diagonal cell is
largest? analysis→hot_take means the model misses subtle argument structure;
hot_take→analysis means it learned "contains digits ⇒ analysis." Those are different
diagnoses and the matrix tells you which one you have.›`

### Sample classifications

Five posts run through the fine-tuned model, label and confidence as reported by the
softmax over the classification head.

> **FILL AFTER RUN** — use `predict()` from `notebook_cells.md`.

| # | Post (truncated) | Predicted | Confidence | True | ✓/✗ |
|---|---|---|---|---|---|
| 1 | `‹...›` | `‹label›` | `__%` | `‹label›` | |
| 2 | `‹...›` | `‹label›` | `__%` | `‹label›` | |
| 3 | `‹...›` | `‹label›` | `__%` | `‹label›` | |
| 4 | `‹...›` | `‹label›` | `__%` | `‹label›` | |
| 5 | `‹...›` | `‹label›` | `__%` | `‹label›` | |

**Why #`‹N›` is a reasonable prediction:** `‹Explain the correct one in terms of the
label definition — what in the post triggered it. "The post names a specific usage
figure and its conclusion depends on that figure, which is exactly the analysis
definition, and the model's 94% confidence suggests it's reading the structure rather
than guessing." Don't just say it was right.›`

---

## 7. Error analysis

### Three specific wrong predictions

> **FILL AFTER RUN** — pull these from the actual test-set errors, not from examples you
> invent. Each needs an explanation tied to the data, the label boundary, or the model's
> behavior.

**Error 1 — `‹true›` predicted as `‹pred›`, confidence `__%`**

> `‹paste the post›`

`‹Why it failed. Tie it to something concrete: was the evidence present but phrased
without digits, so the model's digit heuristic missed it? Was the post short enough that
there was little to go on? Did the training set contain few examples of this shape?›`

**Error 2 — `‹true›` predicted as `‹pred›`, confidence `__%`**

> `‹paste the post›`

`‹Why it failed.›`

**Error 3 — `‹true›` predicted as `‹pred›`, confidence `__%`**

> `‹paste the post›`

`‹Why it failed. If you find you labeled two similar posts differently, say so — that's
an annotation-consistency finding, and it's more valuable than a clean story.›`

### Systematic pattern

`‹FILL — the generalizable observation, with counts. Per planning.md §7c: for every
pattern, check how many errors it covers AND how often the same feature appears in
correct predictions. A feature present in 4 of 11 errors and 30 of 28 correct predictions
is not a pattern.›`

**Patterns proposed by the AI pass that I discarded, and why:** `‹List them. The
discards show the verification actually happened, and the rubric's AI-usage point wants
to see what you overrode.›`

---

## 8. What the model learned vs. what I intended

`‹FILL — this is the 2-point section, so make it the most specific writing in the
README. Not "it needs more data."›`

The intent was a model that reads **argument structure**: whether a claim rests on
support that does work. `‹What the decision boundary actually appears to track — the
honest version. Candidates, pick what the evidence supports:›`

- `‹Surface evidence markers rather than evidential force — digits, percent signs, and
  words like "snap share" push toward analysis regardless of whether the evidence
  supports the conclusion. If true, the tell is that "Puka's target share is insane
  but... dude is cooked" gets predicted analysis.›`
- `‹Register and length rather than reasoning — long and calm reads as analysis, short
  and capitalized reads as reaction, and hot_take becomes the residual bucket for
  everything in between. If true, you'd expect hot_take to have the worst F1 of the
  three.›`
- `‹Specific vocabulary that leaked through — if a player name or a week number is
  overrepresented in one class in the training data.›`

`‹Then: what does the gap mean? The stat-flavored rant from §3 is the crux — I wrote
the rule specifically because I anticipated this failure, and the model ‹did / did not›
learn it. If it didn't, that's a real finding: a human-written decision rule doesn't
transfer to the model just because the annotations follow it, unless the training set
contains enough examples of the rule being applied. Say how many such examples were
actually in your training split — if it's four, the model had no chance.›`

`‹Close with what would change it: more examples of the specific hard case, a tighter
label definition, or an admission that 250 examples is not enough to teach evidential
force and the honest ceiling for this approach is lower than I set.›`

---

## 9. Spec reflection

**One way the spec helped.** `‹FILL. Strong candidate: the instruction to read 30–40
posts before committing to labels, and to find one genuinely ambiguous post and write
its decision rule. The stat-flavored rant rule in §2 came directly out of that exercise,
before annotation, which means it was applied consistently across all 250 examples
instead of being invented halfway through and applied to the second half only.›`

**One way the implementation diverged, and why.** `‹FILL. Strong candidate: the spec
says 200 examples; I collected ‹N›, because 200 leaves a 30-example test set where each
class has ~10 examples and a single flip moves per-class F1 by ~0.05 — per-class metrics
I couldn't defend. Second candidate: I filtered bare advice requests out of the sampling
frame rather than building a fourth label, which trades universal coverage for clean
boundaries; the spec's mutual-exclusivity requirement and its ≥90%-exhaustiveness
requirement pull in opposite directions on a sub where a large share of posts are
transactions rather than takes, and I chose exclusivity.›`

---

## 10. AI usage

**Instance 1 — label stress-testing (before annotation).**
I gave Claude the three definitions and the edge-case rule and directed it to generate
ten posts sitting on the boundary between two labels, then classified each myself using
only the written definitions to find cases the definitions couldn't resolve. Eight
resolved cleanly; two broke and each produced a new rule — the generalization rule
("every week" → `hot_take`) and the primary-speech-act rule (advice requests out of
scope). **What I overrode:** Claude proposed "Herbert just doesn't look right, I've
watched every Chargers game" as genuinely ambiguous between `analysis` and `hot_take`,
arguing that sustained personal observation is a form of evidence. I rejected that — an
unfalsifiable impression is exactly what the `hot_take` definition's "vague" clause
covers, and accepting it would collapse the boundary the whole project measures. I left
the definition unchanged and added "unfalsifiable personal impression" to the vague list
to make the ruling explicit.

**Instance 2 — annotation assistance (disclosed).**
I used Claude to **pre-label** batches of ~25 collected examples, given the §2
definitions and decision procedure verbatim. **Every pre-label was read and reviewed by
me**, and the model's suggestion is preserved in the `prelabel` column of the committed
CSV alongside my final `label`, so the override rate is auditable rather than asserted.
I overrode `‹X›` of `‹N›` (`‹__›%`). `‹Say where the overrides clustered and give one
concrete example of a suggestion you changed and why.›`

**Instance 3 — failure-pattern analysis (after evaluation).**
I gave Claude the full set of misclassified test examples and directed it to propose
*systematic* patterns rather than per-example explanations. I then verified each proposed
pattern against the error set by counting how many errors it covered and checking whether
the same feature appeared just as often in correct predictions. `‹Which patterns survived
and which I discarded — see §7.›`

**Instance 4 — scaffolding.**
Claude drafted the collection script, the structure of this README and `planning.md`, and
the Colab cells in `notebook_cells.md`. `‹Say what you changed — every number, every
example, every judgment about your own data is yours, and the sections you rewrote should
be named here.›`

---

## Repo contents

```
planning.md                    design doc, written before data collection
README.md                      this file — the evaluation report
notebook_cells.md              label map, Groq prompt, training args, predict()
scripts/collect_reddit.py      corpus collector (public .json endpoints)
data/raw_unlabeled.csv         collector output, pre-annotation
data/takemeter_labeled.csv     the labeled dataset
evaluation_results.json        exported from Colab
confusion_matrix.png           exported from Colab (supplementary to §6's table)
app.py                         Gradio interface (stretch)
```

## Reproducing

```bash
python3 scripts/collect_reddit.py --out data/raw_unlabeled.csv --target 400
# annotate data/raw_unlabeled.csv -> data/takemeter_labeled.csv
# open the Colab notebook, set runtime to T4 GPU, upload the labeled CSV
# run sections 1, 2, then 5 (baseline), then 3, 4, 6
```

---

## Stretch features

`‹Delete the ones you don't attempt.›`

**Inter-annotator reliability.** `‹Who labeled 30+ examples independently, the agreement
rate (Cohen's κ or percentage), and an analysis of where you disagreed. The disagreements
are the interesting part — if they cluster on one label pair, that pair's definition is
the weak one.›`

**Confidence calibration.** `‹The binned table from notebook_cells.md. Does the 0.9–1.0
bin beat the 0.6–0.7 bin? Fine-tuned DistilBERT on a small set is usually overconfident;
reporting that honestly is worth more than a flattering story.›`

**Error pattern analysis.** `‹See §7's systematic-pattern subsection.›`

**Deployed interface.** `‹Gradio app in app.py; document how to run it and show it in the
demo video.›`
