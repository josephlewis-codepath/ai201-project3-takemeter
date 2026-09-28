# TakeMeter — classifying discourse quality in r/fantasyfootball

A class-weighted fine-tuned DistilBERT that sorts r/fantasyfootball posts into **analysis**, **hot_take**, and **reaction** — three ways a post can relate a claim to its support — benchmarked against a one-shot-per-class `openai/gpt-oss-120b` baseline on the same 57-example held-out test set.

**Joseph Lewis · AI201 Project 3**

📹 **Demo video:** https://youtu.be/8LTiDB6DOrA 
📓 **Colab notebook:** https://colab.research.google.com/drive/17rhO2vDojhJTnsaM5riGDD9gCviQv4vL
📄 **Design doc:** [`planning.md`](planning.md) — written before data collection, updated twice as findings landed
📊 **Dataset:** [`data/takemeter_labeled.csv`](data/takemeter_labeled.csv) · [`data/excluded_log.csv`](data/excluded_log.csv)

---

## Evaluation report summary

| | Fine-tuned DistilBERT (class-weighted) | Baseline: gpt-oss-120b | Δ |
|---|---|---|---|
| Accuracy | **0.667** | 0.596 | **+0.070** |
| **Macro-F1** | **0.644** | 0.556 | **+0.088** |
| Weighted-F1 | 0.667 | 0.580 | +0.087 |
| `analysis` F1 | 0.593 | 0.500 | +0.093 |
| `hot_take` F1 | 0.571 | 0.457 | +0.114 |
| `reaction` F1 | 0.769 | 0.712 | +0.057 |

Majority-class floor (always predict `reaction`): 0.456 accuracy. The fine-tuned model clears it by +0.211.

**Did it hit the §6 targets in planning.md? No — four for four, but two of them narrowly.**

| Target | Result | |
|---|---|---|
| Macro-F1 ≥ 0.70 | 0.644 | miss by 0.056 |
| No class F1 below 0.60 | lowest 0.571 (`hot_take`) | miss by 0.029 |
| `analysis` recall ≥ 0.70 | 0.533 | miss by 0.167 |
| ≥ +0.10 macro-F1 over baseline | +0.088 | miss by 0.012 |

By my own definition this classifier is **not deployable**. Two thresholds were missed by margins smaller than a single test example is worth (1/57 = 0.018), which on a 57-example test set means they aren't distinguishable from noise — but I set them before seeing any data and I'm not moving them now. The honest summary: fine-tuning on 262 examples produced a model that beats a 120B general reasoning model on this task, and is still not good enough to put in front of users.

---

## 1. Community

**r/fantasyfootball.**

It's one of the few large subreddits where the *same event* reliably produces all three kinds of discourse within minutes, in the same thread. A running back gets six carries in the first half, and the replies split into people pulling up his snap share, people declaring him done for the season, and people screaming in all caps because they benched the alternative. Topic held constant, manner of argument varying — which is what makes this a real classification problem rather than a keyword lookup.

Two things make it a good fit beyond that. **Evidence here is concrete and checkable**: fantasy discourse runs on figures that exist — snap percentage, target share, route participation, red zone touches, opponent run-defense rank — so "backed by evidence" is a property I can verify while annotating instead of a vibe I have to intuit. And **the distinction is already a community norm**: users police take quality themselves, with "source: trust me bro," `!RemindMe 1 week`, and the weekly ritual of resurfacing last month's confident bad takes.

The consequence for the model is the interesting part: all three labels share the same vocabulary — player names, team names, "targets," "start" — so the classifier can't win by learning topic words. It has to pick up something about argument structure. Section 8 is about how far it got.

---

## 2. Label taxonomy

One axis: **how the post relates a claim to its support.**

### `analysis`

The post argues toward an evaluative or predictive conclusion and rests that conclusion on something **checkable** — a figure, a named usage/scheme/matchup detail, or a specific verifiable event. Delete the support and the conclusion loses its footing.

> "Bijan ran 71% of routes last week against 44% in weeks 1–3, and Allgeier's snap share fell off a cliff coming out of the bye. That's a real usage change, not a one-week blip."

> "Kittle was limited Wednesday and Thursday and full on Friday. Historically 49ers tight ends with that practice pattern have played around 85% of snaps that week."

### `hot_take`

The post states a confident evaluative claim without checkable support. Support is absent, vague, an unfalsifiable personal impression, or decorative — a figure picked for effect that doesn't establish the claim. Confident qualitative characterization belongs here, however concrete it sounds.

> "Bijan is the overall RB1 rest of season. Take it to the bank."

> "Puka's target share is insane but Stafford throws him into coverage every other play. Dude is cooked in this offense."

### `reaction`

The post exists to express feeling — celebration, anguish, rage, disbelief, mockery, humor — rather than to argue. It may contain an embedded claim and still be a reaction.

> "WHY DID I BENCH HIM. Why. I looked at that lineup for forty minutes."

> "lmaooo my opponent started a guy on bye and still beat me by 40. I'm done."

### The decision procedure

1. **Dominant mode.** Is expressing feeling what the post is *for*? Strip the attitude away — if only attitude is left, it's `reaction`, even with a claim embedded.
2. **Checkable support.** Otherwise, does the claim rest on a figure, a named usage/scheme/matchup detail, or a specific verifiable event? Yes → `analysis`. Absent, vague, qualitative-only, or decorative → `hot_take`.
3. Emotional **and** resting on checkable support → `analysis`. A well-argued rant is still an argument.

Steps 1 and 2 are stated in this sharpened form because the original versions were ambiguous, and a blind double-labeling exercise measured exactly how much (§8, and `planning.md` §7b). **"Specific" is not "checkable":** *"He can't sit in the pocket"* describes something real but nothing anyone could look up.

---

## 3. Data

### Source

Public r/fantasyfootball comments and self-post bodies, collected through Reddit's `.json` endpoints using my own logged-in browser session ([`scripts/browser_collect.js`](scripts/browser_collect.js)), then parsed and cleaned by [`scripts/collect_reddit.py`](scripts/collect_reddit.py). Public content only, nothing behind authentication.

Sampling deliberately spanned five thread types — game and post-game threads, rant threads, top-of-week and top-of-month posts, player discussion posts, and the sub-wide recent comment stream. Pulling only from game threads would have produced an 80% `reaction` dataset and a model that learned nothing.

7,191 comments survived basic cleaning. From those I drew a **450-example annotation pool stratified by length only, never by content keywords** — sampling `analysis` candidates by "has digits" would have baked the digits-equals-analysis shortcut into the training data, which is the exact failure the plan warned about.

Cleaning: URLs, `/u/` handles, `/r/` references and quoted parent comments stripped; posts under 4 words dropped; truncated at 180 words; bots, deleted and removed bodies dropped; near-duplicates removed; bare advice requests filtered as out of scope.

**375 labeled, 75 excluded** with reasons logged in `data/excluded_log.csv` — 19 advice requests, 23 off-domain medical and legal drift, 7 jokes, 7 too thin, 19 assorted meta and promo posts.

### Labeling process

1. **Pre-labeling.** Claude labeled all 375 from the §2 definitions.
2. **Blind double-labeling.** Rather than skim 375 pre-labels — which invites rubber-stamping — I labeled a random **75 blind**, with the model's labels hidden, using only the written procedure. Agreement: **42/75 = 56%, Cohen's κ = 0.324.**
3. **Reconciliation.** That number said the definitions weren't doing the work I claimed. The disagreements were directional, not noise: 12 cases of `hot_take`→my `reaction` (step 1 ambiguity) and 10 of `analysis`→my `hot_take` (step 2 ambiguity). I wrote Rules A and B to encode my boundary, adopted my 75 blind labels as ground truth, and re-examined the other 300 under the new rules.
4. **78 of 375 labels changed (20.8%).** The pre-reconciliation label is preserved in the CSV's `prelabel` column, so the revision is auditable.

This is disclosed again in §10.

### Label distribution

| Label | Count | Share |
|---|---|---|
| `reaction` | 172 | 45.9% |
| `hot_take` | 108 | 28.8% |
| `analysis` | 95 | 25.3% |
| **Total** | **375** | 100% |

Largest class 45.9%, inside the 70% ceiling; smallest 25.3%, above the 20% floor I set myself. Stratified 70/15/15 → **262 train / 56 validation / 57 test**.

### Three examples that were genuinely hard to label

**1. Stats with no claim attached.**

> "28 missed tackles, 247 yards after missed tackles. Yeesh."

Real figures, precisely the kind `analysis` is defined around. But "Yeesh" is the entire payload — there's no claim for the numbers to support. **Decided: `reaction`.** Step 1 resolves it before step 2 is ever reached, which is the clearest case for why the procedure is ordered rather than a checklist.

**2. A specific play described in detail that never becomes a claim.**

> "The Malachi Fields drop was so demoralizing. It was on Winston's 2nd drive, he threw a deep bomb and hit Fields who just dropped it. The Giants OL didn't hold up. I hope they stick with him"

Three verifiable specifics. **Decided: `reaction`.** The post is dismay plus a hope; the details convey how it felt rather than establishing anything. A post can be rich in checkable detail and still not be an argument.

**3. A rant that ends in a prediction.**

> "Fuck you Seahawks... it's wild how having a coaching philosophy of putting Charb in for 3/4th the plays and RZ makes a drastic difference... I expect KW3 to go down with a foot injury by game 5"

That last clause is a durable, falsifiable claim, which by the letter of step 1 should push it out of `reaction`. **Decided: `reaction`** — the prediction is a curse, not a position being defended. This is the hardest of the three and the one I'd expect a second annotator to fight me on.

---

## 4. Fine-tuning

**Base model:** `distilbert-base-uncased` (66M parameters) with a 3-class classification head.
**Platform:** Google Colab, free T4 GPU. **Libraries:** `transformers`, `datasets`, `scikit-learn`.
**Tokenization:** max length 256, truncation, dynamic padding. **Split:** 70/15/15, stratified, seed 42.

| | Final run |
|---|---|
| Epochs | 8 (best validation macro-F1 kept) |
| Learning rate | 3e-5 |
| Batch size | 16 train / 32 eval |
| Loss | cross-entropy with **inverse-frequency class weights** |
| Class weights | `analysis` 1.323 · `hot_take` 1.149 · `reaction` 0.728 |
| Checkpoint metric | validation macro-F1 |
| Seed | 42 |

### The hyperparameter decision: class-weighted loss

The starter defaults (3 epochs, lr 2e-5, unweighted) produced a model that **never predicted `hot_take` once in 57 test examples.** The entire middle column of the confusion matrix was zero. Accuracy read 0.561, which looks survivable until you notice that always guessing `reaction` scores 0.456 — fine-tuning had bought 10 points over a constant while quietly abandoning a third of the task.

That's textbook majority-class collapse: `reaction` is 46% of the training data, `hot_take` is the hardest boundary, and with 262 examples the cheapest available strategy is to treat the hard class as noise.

The fix was a weighted cross-entropy loss with weights set to inverse class frequency, so a `hot_take` error costs about 1.6× a `reaction` error, plus 8 epochs to give the harder boundary time to form.

| | Accuracy | Macro-F1 | `hot_take` F1 |
|---|---|---|---|
| Unweighted, 3 epochs | 0.561 | 0.413 | **0.000** |
| Class-weighted, 8 epochs | **0.667** | **0.644** | **0.571** |

**+0.231 macro-F1** — far more than any epoch-count or learning-rate adjustment was going to produce, because the problem was never capacity. It was the incentive.

**Per-epoch validation macro-F1:**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Val macro-F1 | 0.354 | 0.487 | 0.608 | 0.551 | 0.618 | 0.596 | 0.642 | **0.648** |

Best checkpoint was epoch 8 — **the last one**. The curve never plateaued and never turned over, so the run was stopped by my epoch budget rather than by convergence. I report that as a limitation rather than a tuned result: more epochs would likely have helped, and I don't know where it tops out.

---

## 5. Baseline

**Model: `openai/gpt-oss-120b` via Groq.** Temperature 0, `max_tokens=512`, `reasoning_effort="low"`.

**This is a substitution, and it needs stating.** The spec mandates `meta-llama/llama-4-scout-17b-16e-instruct`. That model has been retired from Groq's free tier, and my account has no Llama chat model of any generation — the only general-purpose instruct models available were `gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b` and `allam-2-7b`. I used the largest.

The direction of that bias matters: gpt-oss-120b is a *reasoning* model roughly 1,800× the size of DistilBERT, so it's plausibly a **stronger** baseline than Llama-4-Scout would have been. The substitution raises the bar my fine-tuned model had to clear, not lowers it.

**One-shot per class, not zero-shot.** The starter's prompt skeleton requires one example post per label, so the baseline saw three illustrative posts. Those examples are hand-written illustrations from `planning.md` §2, and I verified none appears anywhere in the dataset — so nothing from the test set reached the baseline. But calling it "zero-shot" would be inaccurate.

**How results were collected.** Each of the 57 test examples was sent individually with the §2 definitions and full decision procedure in the system prompt, output constrained to a bare label name. Responses were lowercased and matched against the label strings. **57/57 parsed — zero unparseable.** The baseline was run before the class-weighted model was trained, so there was no opportunity to tune it against a known target.

**Per-class, baseline:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | **1.000** | 0.333 | 0.500 | 15 |
| `hot_take` | 0.421 | 0.500 | 0.457 | 16 |
| `reaction` | 0.636 | 0.808 | 0.712 | 26 |

The `analysis` column is the interesting one: **perfect precision, 0.33 recall.** When gpt-oss-120b says "analysis" it is never wrong — it just only says it when a post is unmistakable. It applies the definition conservatively and correctly and misses two thirds of the class. The fine-tuned model trades that away (precision 1.00 → 0.67) for recall (0.33 → 0.53), which is the better trade for a tool meant to surface substantive posts.

### A note on the "unparseable" counter

The milestone says to treat >10% unparseable responses as a prompt problem. Mine reported **57/57 unparseable on three separate runs**, and the prompt was never the cause. The three actual causes were: a retired model ID returning 404 on every call; the prompt skeleton left unfilled, so the model was being asked to output `<label_1>`; and `gpt-oss-120b` being a reasoning model whose 20-token budget was consumed before any visible content was produced. `classify_with_groq()` catches API exceptions and parse failures in the same branch and reports both as a prompt issue. Counting transport errors separately from parse failures would have saved roughly two hours.

---

## 6. Results

### Overall

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| **Fine-tuned DistilBERT (class-weighted)** | **0.667** | **0.644** | **0.667** |
| Baseline — gpt-oss-120b | 0.596 | 0.556 | 0.580 |
| Fine-tuned, unweighted (superseded) | 0.561 | 0.413 | 0.463 |
| *Majority-class floor* | *0.456* | *0.209* | — |

### Per-class — fine-tuned

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.667 | 0.533 | 0.593 | 15 |
| `hot_take` | 0.526 | 0.625 | 0.571 | 16 |
| `reaction` | 0.769 | 0.769 | 0.769 | 26 |

### Confusion matrix — fine-tuned

Rows are true labels, columns are predictions. (`confusion_matrix.png` is a supplementary copy of this table.)

| true \ pred | analysis | hot_take | reaction | total |
|---|---|---|---|---|
| **analysis** | 8 | 4 | 3 | 15 |
| **hot_take** | 3 | 10 | 3 | 16 |
| **reaction** | 1 | 5 | 20 | 26 |

Read directionally: `reaction`→`hot_take` is the single largest off-diagonal cell (5), and `hot_take` is involved on one side or the other in **15 of 19 errors (79%)**. The `analysis`↔`hot_take` pair — the boundary this whole project is about — accounts for 7 of 19 (37%). The `reaction`↔`analysis` pair is nearly clean in one direction (1 error) and modest in the other (3), which says the model has learned that emotional venting and evidence-backed argument are different things. What it hasn't learned is where the middle is.

### Sample classifications

| # | Post (truncated) | True | Predicted | Confidence | ✓ |
|---|---|---|---|---|---|
| 1 | "Dude wouldn't even know the playbook" | reaction | reaction | 0.53 | ✓ |
| 2 | "Sun god and adams here. Riding this high until puka comes back and adams goes back to putt..." | reaction | reaction | 0.90 | ✓ |
| 3 | "Same. My WR room is screwed. I didn't see either injury coming because they were both fine..." | reaction | reaction | 0.85 | ✓ |
| 4 | "They had a great first game. I'm feeling Dexter Lawrence has the defense on an upward mome..." | hot_take | analysis | 0.72 | ✗ |
| 5 | "Just comes down to shit attitude and work ethic. NFL coaches are willing to put up with ba..." | hot_take | analysis | 0.51 | ✗ |

**Why #2 is a reasonable prediction.** "Sun god and adams here. Riding this high until puka comes back and adams goes back to putting up 5 points like weeks 1 lol" carries a real prediction — Adams will regress — and even a figure ("5 points"). A model keying on surface evidence markers would call that `analysis`. It called `reaction` at 0.90, which is correct under Rule A: the post exists to enjoy a good week, and the prediction is self-deprecating rather than defended. That's the model applying step 1 before step 2, which is exactly the ordering the taxonomy specifies.

---

## 7. Error analysis

19 errors out of 57.

### Three specific failures

**1. `hot_take` predicted as `analysis`, confidence 0.946 — the highest-confidence error in the set.**

> "If you're down 63 and you're starting davante, the choice between dart and Stafford is easy. If davante has a big game, Stafford by default is probably going to have a big game as well. Obviously there's a chance davante has 150 yards and 2 tds while Stafford has 200 yards and 2 tds, but Stafford is the better play."

Long, calm, structured, full of numbers — and every one of those numbers is **hypothetical**. "150 yards and 2 tds" is an invented scenario, not a fact about anything that happened. The reasoning is sound but rests on nothing checkable, which is precisely what puts it in `hot_take` under Rule B. The model was more confident here than almost anywhere else in the test set, which tells you it is reading *evidence-shaped text* rather than evidence. This is the clearest single demonstration that "specific" and "checkable" are different properties and the model has only learned the first.

**2. `analysis` predicted as `hot_take`, confidence 0.448 — the lowest-confidence error.**

> "I really don't understand why people are worried. He had 10 fucking targets last game as a TE."

One checkable figure doing all the work: delete "10 targets" and the post is an empty reassurance. That's `analysis` by definition. The model hedged at 0.448 — near-uniform across three classes — because the post is short, profane and conversational in register, and everything else it has learned about `analysis` is long and calm. Its uncertainty is appropriate; its decision boundary is keyed to register rather than to whether the figure supports the claim.

**3. `analysis` predicted as `reaction`, confidence 0.925 — where the label is wrong, not the model.**

> "I had him and rice, needed 47, got 45. Damn"

This is labeled `analysis` in the dataset. It should not be. Three precise numbers, all about one outcome, ending in "Damn" — this is the stat-flavored rant that `planning.md` §3 named as the anticipated hardest case, and Rule A sends it to `reaction` unambiguously. The label came from my blind pass, before Rules A and B existed, and it survived reconciliation because blind-pass labels were adopted as ground truth without re-checking them against the rules they later produced.

**The model got this right at 0.925 confidence and was scored wrong for it.** Counted honestly, that isn't a model error. It's an annotation error — and it points at something systematic.

### The systematic pattern: errors track the annotation method

Following the failure-analysis plan in `planning.md` §7c, I gave the full error set to Claude and asked for patterns, then verified each by counting rather than accepting the story. The pattern that survived: **errors concentrate on the examples I labeled in the blind pass.**

The dataset has two provenances — 75 examples labeled by me blind (before Rules A and B existed), and 300 labeled by applying those rules afterward. 15 of the 57 test examples come from the blind pass.

| Label source | n in test | errors | error rate |
|---|---|---|---|
| Blind pass (pre-rules) | 15 | 9 | **60.0%** |
| Rule-applied (post-rules) | 42 | 10 | **23.8%** |
| Overall | 57 | 19 | 33.3% |

**2.5× the error rate, Fisher exact p = 0.023.** The model is substantially worse on exactly the subset of labels produced by a process that a κ of 0.324 had already flagged as inconsistent, and better on the subset produced by applying explicit written rules.

The most defensible reading is that a meaningful share of the "errors" on blind-pass examples are label noise rather than model failure, as failure 3 demonstrates concretely. The training set carries the same contamination in the same proportion, so the noise both degrades what the model learned and inflates the apparent error rate at evaluation.

**Patterns I discarded.**

- *"Short posts fail more."* The errors span 5 to 60 words, and the shortest test examples are among the most reliably correct.
- *"Sarcasm causes failures."* Only 2 of 19 errors are recognizably sarcastic — no more than their base rate in the test set.
- *"Posts with digits get pulled toward `analysis`."* This is true and important as a mechanism (see failure 1), but as a *population-level pattern* it doesn't hold: several digit-bearing posts are classified correctly, and `reaction`→`analysis` is the rarest cell in the matrix at 1 case. I'm reporting it as a mechanism visible in individual high-confidence failures, not as a regularity across the error set.

---

## 8. What the model learned vs. what I intended

I intended a model that reads **argument structure**: whether a claim rests on support that does work. What it appears to have learned is closer to **register plus evidence-shaped surface form.**

The evidence, in order of strength:

**The highest-confidence errors are all cases where form and substance point in opposite directions.** The 0.946 failure is long, calm and numeric with entirely hypothetical numbers. The 0.925 failure is terse and profane with real numbers. Where a post *looks* like the class it belongs to, the model is right; where looking and being come apart, it follows looking. That is what having learned the proxy rather than the target means.

**`reaction` works and the middle doesn't.** F1 by class: `reaction` 0.769, `analysis` 0.593, `hot_take` 0.571. `reaction` is separable on surface features the model has abundant signal for — profanity, caps, brevity, first-person framing, exclamation. `hot_take` is defined by what it *lacks* (checkable support), and absence has no surface form. With 76 training examples it never acquired a stable shape, which is why it's implicated in 79% of errors and why it collapsed entirely without class weighting.

**The one thing I most wanted it to learn, it didn't.** Rule B — "specific" is not "checkable" — is the distinction that took a 56%-agreement blind pass to discover, and failure 1 shows the model does not hold it. I encoded it in the definitions, in the Groq prompt, and across 375 labels. But a rule stated in a definition is not a rule the model can see; it can only see whatever in the text correlates with the labels. Hypothetical numbers and real numbers look identical to a bag of subword tokens, and nothing in 262 examples taught it otherwise.

**Part of the gap is mine, not the model's.** The provenance analysis says my labels aren't one thing: the blind-pass subset and the rule-applied subset encode measurably different boundaries, and the model was asked to learn their union. I told it to learn a distinction I had only half-articulated at the time I recorded a fifth of the labels.

**What would actually change this**, in order of expected value:

1. **Re-audit the 75 blind-pass labels against Rules A and B.** I know at least one is wrong, it takes half an hour, and it removes measurable noise from both training and test.
2. **Collect targeted `hot_take` examples** — specifically long, calm posts with hypothetical or decorative numbers, since that's the exact configuration the model fails on and there are only 76 `hot_take` examples in training.
3. **More epochs**, since validation macro-F1 was still climbing when the budget ran out.

More data in general is the least useful of these. The problem isn't that the dataset is small; it's that the hard class is both rare and defined negatively.

---

## 9. Spec reflection

**One way the spec helped.** The instruction to define labels with two examples each and name the hardest anticipated edge case *before* annotating forced the stat-flavored-rant rule into existence in advance, which meant it was applied consistently across all 375 examples rather than invented halfway through. More than that, having written definitions is what made the blind double-labeling interpretable: when agreement came back at 56%, I could point at *which clause* was ambiguous in each of the two disagreement buckets and write Rules A and B to fix them. Without the spec's insistence on written definitions up front, I'd have had a bad number and no idea what produced it.

**One way the implementation diverged, and why.** The spec mandates the Groq baseline run on `meta-llama/llama-4-scout-17b-16e-instruct`. That model no longer exists on Groq's free tier and my account has no Llama chat model at all, so the baseline runs on `openai/gpt-oss-120b` — a reasoning model, and plausibly a harder opponent than the spec intended (§5).

A second, smaller divergence: the spec asks for 200 examples and I collected 375. At 200 the test set is 30 examples with per-class support around 10, where a single flip moves per-class F1 by roughly 0.05. Several of my conclusions — the class-weighting comparison, the provenance analysis — depend on per-class numbers, and at 30 test examples I couldn't have defended any of them.

---

## 10. AI usage

**1. Label stress-testing (before annotation).** I gave Claude the three definitions and asked it to generate ten posts sitting on the boundary between two labels, then classified each myself using only the written definitions. Eight resolved cleanly; two broke, and each produced a new rule. **What I overrode:** Claude proposed "Herbert just doesn't look right, I've watched every Chargers game" as genuinely ambiguous, arguing that sustained personal observation is a form of evidence. I rejected that — an unfalsifiable impression is exactly what the `hot_take` "vague" clause covers, and accepting it would collapse the boundary the project measures.

**2. Annotation assistance (full disclosure).** Claude pre-labeled **all 375 examples** from the §2 definitions. Rather than review them one by one, I labeled a random 75 blind and measured agreement: **56%, κ = 0.324.** I then defined Rules A and B to encode my boundary, adopted my 75 blind labels as ground truth, and had Claude re-examine the remaining 300 under those rules. **78 labels changed (20.8%).** Every pre-reconciliation label is preserved in the CSV's `prelabel` column, so the revision is auditable rather than asserted. The 265 rule-applied labels I did not personally re-check are the residual risk here, and §7 quantifies what that kind of risk costs.

**3. Failure-pattern analysis (after evaluation).** I gave Claude all 19 misclassified test examples and asked for *systematic* patterns rather than per-example explanations, then verified each by counting. The provenance pattern survived and is reported with its test statistic; three proposed patterns did not survive and are listed in §7 with the counts that killed them. Claude also wrote the code that computed the provenance split, which I checked by confirming it reproduced the notebook's exact test set before trusting its output.

**4. Infrastructure and drafting.** Claude wrote the collection scripts, the class-weighted training cell and the evaluation cell, and drafted this README and `planning.md` from my decisions and my data. Every number in this document comes from notebook output. The judgments — which boundary to adopt when Claude's and mine disagreed, which patterns to believe, whether to call the classifier deployable — are mine. In the one case where they conflicted directly (§7 failure 3), the record shows the model was right and my label was wrong.

---

## 11. Stretch: confidence calibration

Does a high-confidence prediction actually get it right more often?

| Confidence bin | n | Accuracy | Mean confidence | Gap |
|---|---|---|---|---|
| 0.4 – 0.5 | 3 | 0.333 | 0.477 | −0.14 |
| 0.5 – 0.6 | 9 | 0.667 | 0.557 | +0.11 |
| 0.6 – 0.7 | 8 | 0.625 | 0.659 | −0.03 |
| 0.7 – 0.8 | 5 | 0.400 | 0.735 | −0.34 |
| 0.8 – 1.0 | 32 | 0.750 | 0.878 | −0.13 |

**Expected calibration error: 0.131.** Weakly informative and poorly calibrated. The trend across the full range is positive — the bottom bin is 33% accurate and the top is 75% — so confidence carries *some* signal. But it isn't monotonic: the 0.7–0.8 bin is the *least* accurate in the table at 40%, worse than the 0.5–0.6 bin.

And the model is systematically overconfident where it matters most. 32 of 57 predictions land in the top bin averaging 0.878 confidence and are right 75% of the time — a 13-point gap on more than half the test set. Section 7's failure 1 sits inside that bin at 0.946.

Practically: you could not use this model's confidence as a filter. Thresholding at 0.8 would retain 56% of predictions at 75% accuracy versus 67% unfiltered — a real but modest gain, and one that would silently discard correct low-confidence calls like failure 2, where hedging was the appropriate response to a genuinely hard post.

---

## Repo contents

```
planning.md                    design doc — written before collection, updated twice
README.md                      this file — the evaluation report
notebook_cells.md              label map, Groq prompt, training args
data/takemeter_labeled.csv     375 labeled examples (label + prelabel + notes)
data/excluded_log.csv          75 exclusions with reasons
scripts/browser_collect.js     browser-session collector
scripts/collect_reddit.py      JSON parser and cleaner
tools/annotate.html            blind annotation interface
evaluation_results.json        exported from Colab
confusion_matrix.png           exported from Colab (supplementary to §6's table)
```

## Reproducing

```bash
# 1. collect — paste scripts/browser_collect.js into Chrome's console on reddit.com
# 2. parse
python3 scripts/collect_reddit.py --files "saved/*.json" --out data/raw_unlabeled.csv
# 3. annotate, then open the Colab notebook (T4 GPU) and upload data/takemeter_labeled.csv
# 4. run sections 1 → 2 → 5 (baseline first) → 3 → 4, then the class-weighted cells
```
