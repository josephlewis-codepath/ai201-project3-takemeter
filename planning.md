# TakeMeter — Planning

**Author:** Joseph Lewis
**Course:** AI201, Project 3
**Written:** before data collection. Updated before stretch features (see changelog at the bottom).

---

## 1. Community

**Choice: r/fantasyfootball.**

I picked r/fantasyfootball because it is one of the few large subreddits where the *same event* reliably produces all three kinds of discourse I care about, within minutes of each other, in the same thread. When a running back gets 6 carries in the first half, the replies split into people pulling up his snap share and route participation, people declaring him a bust for the rest of the season, and people screaming in all caps because they benched the alternative. That co-occurrence is what makes the classification task non-trivial: the topic is held constant and only the *manner of argument* varies, so a model can't win by learning topic keywords.

Three more reasons it fits:

- **Evidence here is concrete and checkable.** Fantasy football discourse runs on numbers that exist — snap percentage, target share, route participation, red zone touches, opponent rushing defense rank, air yards. "Backed by evidence" is a property I can verify when annotating, not a vibe I have to intuit. In a subreddit like r/movies, "evidence" would be much harder to pin down.
- **The distinction is a live community norm.** Users in this sub police take quality themselves — "source: trust me bro," "!RemindMe 1 week," and the weekly ritual of quoting last month's confident bad takes are all native behaviors. A classifier that separates argued claims from asserted ones is measuring something the community already measures informally.
- **Volume and access.** ~3M members, public, and the game threads and post-game threads generate thousands of comments per Sunday. Collecting 250+ public comments across a range of thread types is easy and requires no authentication.

A note on what this means for the model: because all three labels share the same vocabulary (player names, team names, "targets," "start"), the model cannot shortcut via topic. It has to learn something about argument structure. That is exactly the gap I want to measure in the evaluation.

---

## 2. Labels

Three labels, on a single axis: **how the post relates a claim to its support.**

### `analysis`

> The post argues toward an evaluative or predictive conclusion using at least one specific, checkable piece of support — a statistic, a usage figure (snaps, targets, routes, touches), a named matchup or scheme detail, or a historical comparison with particulars — and that support does real work, meaning the conclusion would lose its footing if you deleted it.

**Example A**
> "Bijan ran 71% of routes last week against 44% in weeks 1–3, and Allgeier's snap share fell off a cliff coming out of the bye. That's a real usage change, not a one-week blip. I'm buying at whatever the cost is."

**Example B**
> "Kittle was limited Wednesday and Thursday and full on Friday. Historically 49ers tight ends with that practice pattern have played around 85% of snaps that week. I'm starting him over the safer floor guys."

*Why these are analysis:* both cite specific figures, and both conclusions depend on those figures. Strip the numbers out of A and you are left with "I like Bijan," which is a different post.

### `hot_take`

> The post states a confident evaluative claim about a player, team, or strategy without genuinely supporting it. Support is absent, vague ("he's been trash," "I've watched every game and he doesn't look right"), or decorative — a figure selected for rhetorical effect that does not actually establish the claim being made.

**Example A**
> "Bijan is the overall RB1 rest of season. Take it to the bank."

**Example B**
> "Puka's target share is insane but Stafford throws him into coverage every other play. Dude is cooked in this offense."

*Why these are hot_take:* A asserts with nothing behind it. B is the more interesting case — it contains a real usage concept ("target share is insane"), but that evidence *contradicts* the conclusion rather than supporting it, and "throws him into coverage every other play" is an unfalsifiable impression. The number is decorative.

### `reaction`

> The post is an immediate emotional response to something that just happened — a touchdown, an injury, a blown lead, a lineup decision that went wrong. It expresses feeling (celebration, anguish, rage, disbelief, humor) rather than making or defending a claim about player value that could still be evaluated next week.

**Example A**
> "WHY DID I BENCH HIM. Why. I looked at that lineup for forty minutes."

**Example B**
> "lmaooo my opponent started a guy on bye and still beat me by 40. I'm done."

*Why these are reaction:* neither contains a claim that could be true or false a week from now. They are about a moment.

### Mutual exclusivity check

The decision procedure I will apply to every example, in this order:

1. **Is there a durable evaluative claim** — something that could be judged right or wrong next week? If no → `reaction`.
2. **If yes, is there specific checkable support that the claim actually rests on?** If yes → `analysis`. If the support is absent, vague, or decorative → `hot_take`.
3. **If a post is emotionally charged AND carries real supporting evidence, evidence wins** → `analysis`. (Passion is not disqualifying; a well-argued rant is still an argument.)

This ordering makes the labels exclusive by construction: step 1 splits reaction from everything else, step 2 splits the remainder.

---

## 3. Hard edge cases

### The one I expect to hurt most: the stat-flavored rant

A post that is emotionally driven and scoped to what just happened, but that *contains a number*:

> "Started Kupp over Nacua and lost by 2. Three targets. THREE. In a game they threw it 48 times."

This has real figures in it. It is also unambiguously venting. A model that learns "digits ⇒ analysis" will get this wrong, and so will a careless annotator.

**Decision rule:** a number only counts as support if it is deployed *toward a claim*, not as an intensifier for a complaint about an outcome. Here the "3" and the "48" amplify the frustration; they don't establish anything about Kupp going forward. → **`reaction`**.

The same post with one clause added flips the label:

> "...that's the third straight week he's under 5 targets while Nacua's route share climbs. He's droppable."

Now the figures support a forward-looking claim. → **`analysis`**.

**How I'll handle it during annotation:** every example containing a digit gets the step-1 question applied explicitly before I look at the number at all. I will also deliberately over-sample stat-flavored rants during collection, so the training set contains enough of them for the model to learn that digits are not sufficient. I expect this to be my largest error bucket regardless.

### Second hard case: the generalized complaint

> "8 targets, 2 catches. Every week."

"Every week" turns a complaint into a pattern claim, which makes it durable, which pushes it out of `reaction`. But the support is a single game's box score.

**Decision rule:** an explicit generalization ("every week," "all season," "he always") makes the post a durable claim → `hot_take`, because one game's line does not establish a season-long pattern. Without the generalization it stays `reaction`.

### Third hard case: the reasoned advice request

> "Trade eval: I give Chase + Mixon, I get Jefferson + Irving. My RB room after is Irving/Hunt/Dowdle, WR is ADP filler. Thoughts?"

Well-reasoned, full of specifics, but its primary speech act is asking, not claiming.

**Decision rule:** a post whose primary speech act is a request for input is **out of scope** and gets filtered at collection time, unless it advances and defends a position of its own ("I think I win this because Jefferson's target share in the second half of last year..."), in which case it is `analysis`.

### Scope note on exhaustiveness

The three labels need to cover ≥90% of what I sample without an "other" bucket. They do — *given my sampling frame*. I am deliberately not sampling from the Daily Start/Sit megathread or the Trade Advice threads, because those are dominated by bare requests ("Gibbs or Achane?") that carry neither a claim nor an emotion and belong to a different axis entirely. I am sampling from game threads, post-game threads, general discussion posts, rant threads, and top weekly posts, where the three labels apply cleanly. This is a scoping decision, and I am stating it rather than pretending the taxonomy is universal: TakeMeter classifies *discourse*, not *transactions*, and the sub contains both.

**Added after annotation — the off-domain problem I did not anticipate.** Injury threads are a large share of the sub, and the comments under them drift hard into amateur medicine (labrum repair failure rates, iliopsoas anatomy, personal surgery recovery timelines) and, in the Josh Jacobs threads, into criminal procedure and CBA suspension rules. These are argued, evidence-bearing comments — they would classify cleanly as `analysis` by the decision procedure, because the procedure is deliberately about argument structure and says nothing about topic.

I excluded them anyway, and the reason is distributional rather than definitional: if a meaningful slice of my `analysis` class is medical and legal argumentation while `hot_take` and `reaction` are almost entirely football, the model can reach high accuracy by learning topic vocabulary instead of argument structure — which is precisely the thing this project is trying to measure. Letting them in would have quietly invalidated the evaluation.

**Exclusion rule:** a comment is out of scope if it is about neither football nor fantasy — pure medical, legal, or procedural argument, off-topic jokes, personal anecdotes unrelated to the game, subreddit and platform meta, and megathread boilerplate. A medical or legal claim that is *about a specific player's outlook* stays in and is labeled normally, because that is fantasy discourse. Every exclusion is logged with its reason in `data/excluded_log.csv` rather than silently dropped, so the rate and the reasons are auditable.

Two smaller categories also surfaced: **news reports and promo posts** carrying no claim of the author's own (a beat reporter's injury note pasted verbatim), and **stat drops with no claim attached** — these were rarer than expected but the first is excluded as non-discourse while the second is `reaction`, since step 1 finds no durable claim.

---

## 4. Data collection plan

**Source.** Public r/fantasyfootball comments and self-post bodies, pulled through Reddit's public `.json` endpoints. No authentication, no private content. Mix of:

| Thread type | Why | Expected label skew |
|---|---|---|
| Sunday game threads / post-game threads | Highest emotional density | heavy `reaction` |
| Weekly "Rant" / "Who's dropping" threads | Complaints that generalize | `reaction` + `hot_take` |
| Top posts of the week/month | Longer, argued posts | heavy `analysis` |
| General discussion / player-specific posts | Mixed | all three |
| Recent sub-wide comment stream | Unfiltered realism | all three |

Sampling across all five is deliberate: if I only pulled game threads I would get an 80% `reaction` dataset and a model that learned nothing.

**Target volume: 260 examples, not 200.** The split is 70/15/15, so 200 examples leaves a 30-example test set — roughly 10 per class, where a single flip moves per-class F1 by ~0.05 and any per-class number is close to meaningless. 260 gets the test set to ~39. I would rather spend an extra 45 minutes annotating than report per-class metrics I can't defend.

**Cleaning rules, applied before annotation:**
- Strip URLs, usernames (`/u/...`), and subreddit references.
- Drop comments under 4 words (no signal for any label) and truncate over ~180 words. (Originally 120; raised because DistilBERT tokenizes to 256 tokens anyway, so a 120-word cap was throwing away long argued posts for no benefit. The change affects ~1.5% of the corpus.)
- Drop deleted/removed bodies, bot comments (AutoModerator, RemindMeBot), and pure-image/link posts.
- Deduplicate on normalized text.
- Drop bare advice requests per the scope note above.

**Per-label targets.** Roughly 33% each, with a hard floor of 20% and a hard ceiling of 55% (well inside the rubric's 70% cap). `reaction` is the one at risk of running hot because game-thread comments are so plentiful; `analysis` is the one at risk of running thin because argued posts are rarer per unit of text.

**If a label is underrepresented after the first pass:** I will do targeted collection rather than reweighting. For a thin `analysis` class I'll pull from top-of-month posts and the sub's longer-form player breakdowns, where the base rate of argued posts is much higher. For a thin `hot_take` class I'll pull from preseason bust/breakout threads and the "unpopular opinion" style posts. Targeted collection changes the distribution honestly; class weights would just paper over a dataset that doesn't contain the examples.

**Annotation record-keeping.** The CSV carries five columns: `text`, `label`, `notes`, `prelabel`, `source_thread`. `prelabel` stores the LLM's suggestion so I can compute how often I overrode it, which is both a useful signal about label clarity and the honest way to disclose AI assistance.

---

## 5. Evaluation metrics

**Primary: macro-F1.**
Three classes that will not be perfectly balanced, and I care about all three equally. Overall accuracy would let a model that nails the largest class and whiffs on the smallest post a respectable number. Macro-F1 averages the per-class F1s without weighting by support, so a class the model can't learn drags the headline number down instead of hiding inside it. That is the behavior I want from a metric on this task.

**Secondary: per-class precision and recall, with special attention to `analysis` recall.**
The intended downstream use is a tool that surfaces substantive posts — a weekly digest of the best-argued takes in the sub. For that use the two error types are not symmetric. A false positive costs a reader ten seconds of skimming. A false negative means a genuinely good post never gets surfaced, which is the entire failure of the product. So `analysis` recall is the metric that maps to real usefulness, and I'll report it separately rather than letting it average away.

**Diagnostic: the confusion matrix, read directionally.**
I specifically want the `analysis` ↔ `hot_take` cells. That pair *is* the project — it's the boundary where "did the evidence do work?" lives. The direction matters too: `analysis` predicted as `hot_take` means the model is missing subtle argument structure; `hot_take` predicted as `analysis` means it learned "contains digits ⇒ analysis," which is precisely the shortcut my stat-flavored-rant edge case predicts.

**Comparative: macro-F1 delta over the Groq zero-shot baseline.**
The absolute fine-tuned number means little on its own. Llama-4-Scout with my label definitions in the prompt already knows what an argument is; if 182 training examples don't beat it, then either my labels encode nothing beyond what a general model infers from the definitions, or my annotation is noisy. Either way the delta is the number that tells me whether the fine-tuning was worth doing.

**Why not accuracy alone:** covered above, but concretely — if my dataset lands at 45/30/25 and the model predicts `reaction` for everything, that's 45% accuracy on a 3-class task where chance is 33%. It looks like learning. Macro-F1 would be 0.21 and would not.

---

## 6. Definition of success

**Threshold for "this classifier is good enough to deploy in a real community tool":**

1. **Macro-F1 ≥ 0.70** on the held-out test set.
2. **No single class F1 below 0.60.** Stated as a floor because with ~39 test examples the macro average can be carried by two strong classes while a third is unusable, and a digest tool that silently never surfaces one category is broken even if the average looks fine.
3. **`analysis` recall ≥ 0.70** — the digest catches roughly 7 of every 10 genuinely substantive posts. Below that, a user is better served scrolling the sub chronologically, which means the tool has no reason to exist.
4. **At least +0.10 macro-F1 over the Groq zero-shot baseline.** If fine-tuning buys less than that, the honest conclusion is that the prompt was the better product and I should say so in the report rather than shipping a model.

**What I'd accept at lower numbers and why:** if macro-F1 lands between 0.60 and 0.70 but the errors concentrate in the `analysis`/`hot_take` pair specifically, I'd call that a *usable* classifier with a known blind spot, because `reaction` filtering alone (pulling the venting out of a feed) is independently useful. If the errors are spread evenly across all three pairs, that's a taxonomy problem, not a model problem, and the right response is to revisit the label definitions rather than collect more data.

**How I'll know I got a suspiciously good result:** anything above 0.95 macro-F1 on a subjective 3-class task with 182 training examples means test leakage, near-duplicate posts across splits, or labels so easy they aren't measuring discourse quality. I'll check for near-duplicates across the split before believing a high number.

---

## 7. AI Tool Plan

### 7a. Label stress-testing — **doing this before annotation**

I gave Claude my three definitions and the edge-case rule and asked it to generate ten posts sitting on the boundary between two labels, then tried to classify each one myself using only the written definitions. The goal was to find cases the definitions couldn't resolve.

Eight of ten classified cleanly. Two broke, and both produced a new rule:

- **"8 targets, 2 catches. Every week."** — I couldn't decide between `reaction` and `hot_take`, because the definitions didn't say what to do with an explicit generalization attached to a single game's line. → Added the generalization rule in §3.
- **"Trade eval: I give Chase + Mixon... Thoughts?"** — reasoned, specific, and not a claim. → Added the primary-speech-act rule and the scope note in §3.

I overrode Claude on one of its suggested boundary cases: it proposed "Herbert just doesn't look right, I've watched every Chargers game" as ambiguous between `analysis` and `hot_take` on the grounds that sustained personal observation is a form of evidence. I don't accept that — an unfalsifiable impression is exactly what the `hot_take` definition's "vague" clause is for, and calling it evidence would collapse the boundary I'm trying to measure. Definition unchanged; I added "unfalsifiable personal impression" to the vague list to make the ruling explicit.

### 7b. Annotation assistance — **yes, with full review and tracked overrides**

I will use Claude to pre-label the collected examples in batches, given the §2 definitions and the §3 decision rules verbatim. Every pre-label gets read and corrected by me; the point of pre-labeling is to change my task from "produce a label" to "audit a label," which is faster and, on the boundary cases, sharper — disagreeing with a stated label forces me to articulate why.

Tracking: the CSV keeps the model's suggestion in a `prelabel` column alongside my final `label`. The override rate is then computable directly, and I'll report it in the README's AI usage section. If the override rate is very low (<5%) I'll treat that as a warning sign that I'm rubber-stamping rather than reviewing, and re-audit a random sample blind.

This is disclosed in the README's AI usage section.

### 7c. Failure analysis — **after evaluation, verified before it's written up**

I'll paste the full set of misclassified test examples (text, true label, predicted label, confidence) into Claude and ask it to propose systematic patterns rather than per-example explanations — specifically: label pairs, post length, presence of digits, sarcasm, hedging language, and whether the post is scoped to a single game.

Verification step, because this is the place where an LLM will happily invent a tidy story: for every pattern proposed, I go back to the error set and count how many errors it actually covers, and check whether the same feature appears just as often in the *correct* predictions. A pattern that describes 4 of 11 errors but also 30 of 28 correct predictions isn't a pattern. Anything that fails that check gets discarded, and I'll note in the README which proposed patterns I threw out — the discards are as informative as the keeps.

---

## Changelog

- *(initial)* Written before data collection, per Milestone 2.
- *(post-collection)* Reddit closed unauthenticated `.json` access mid-project and gated API app creation behind an approval queue, so collection moved to a browser-session script (`scripts/browser_collect.js`) run against my own logged-in session. Sampling frame and cleaning rules are unchanged apart from the word cap below.
- *(post-annotation)* Added the off-domain exclusion rule in §3 and raised the word cap to 180. Final dataset: 375 examples at 34% `analysis` / 29% `hot_take` / 37% `reaction`; 75 candidates excluded with logged reasons.
- *(to update)* Before starting stretch features.
