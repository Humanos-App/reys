# How Reys is trained

Reys is a small Decision Language Model. It reads the sentence a person approves together with the rules that
actually govern an AI agent's permission, and answers four yes/no questions about whether the two match:

1. **A limit that isn't enforced**: the words promise a limit the rules don't check.
2. **A change nobody mentioned**: the rules allow a write the words don't mention.
3. **A promise it can't keep**: the words promise something the rules never allow.
4. **A deletion nobody mentioned**: the rules allow something destructive the words never say.

Each answer is a calibrated probability. Reys never grants or blocks anything; the rules still decide.

This page explains how the model and its data are made. It's a research prototype, and the numbers below are
early.

## The model

- **Encoder:** [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), the ModernBERT architecture
  pretrained on many languages (322M parameters). We need English and Portuguese.
- **Decision head:** the typed decision head from [Laya](https://huggingface.co/convaiinnovations/laya-multilingual):
  each question is written into the input with two answer markers ("yes" / "no"), and the head scores the
  markers. One forward pass per question; no text is generated.
- **What's frozen:** the token embeddings (197M of the 307M encoder parameters are a vocabulary table our data
  can't teach). Everything else is trained.

## The data

There was no dataset for this, so it's generated, and every label is proven rather than guessed.

**Where the permissions come from.** 326 real MCP tools from 14 services (Slack, Gmail, GitHub, Stripe, Notion,
Google Calendar, Google Drive, Atlassian, Linear, Canva, DocuSign…), each labelled read, write or destructive,
plus single-purpose actions (paying an invoice, booking a trip, buying cloud credits, refunds, trading).

**How one example is made:**

1. **Structure, in code.** A published action (its parameters and rules), the values a person fixes, and either
   nothing wrong or exactly one deliberate break: a rule that doesn't enforce a stated limit, a grant with a
   tool or value the text doesn't mention, a text that promises something the rules forbid, a destructive
   operation left open.
2. **Minimal pairs.** Every example comes as a pair: the correct permission and the same one with one thing
   changed (the rules, the values or the text). "Decoy" pairs change something harmless. A model that reads
   only the words, or only the rules, gets half of each pair wrong.
3. **The sentence.** A writer model (Gemini 3 Flash) writes the description the person would see, in an agent's
   voice: starting from a plausible task, stating limits in passing, about a quarter in Portuguese. Issuer
   descriptions are derived from the rules exactly as the platform derives them.
4. **The round trip.** A second model (Claude Haiku 4.5) reads the sentence back and says what it states. If that
   doesn't match what the writer was asked to say, the text is rejected: the example is dropped, or keeps an
   earlier text that passed (8–17% of examples per version).
5. **The label is proven.** A checker runs the VIA SDK's own rule evaluator over candidate calls. A "yes" means it
   found a concrete call the rules allow that contradicts the sentence, and that call is stored with the label.
   No model decides what's true.
6. **Shortcut gates.** Before any training, baselines that see only the words, only the rules, or only word
   overlap must not beat "always answer no". If one does, the generator is fixed, not the model.

**Size:**

| split | authorizations | descriptions | question-answer pairs |
|---|---|---|---|
| train | 3,374 | 6,330 | 25,320 |
| validation | 450 | 836 | 3,344 |
| test, same services | 410 | 758 | 3,032 |
| test, three services never trained on (Linear, Canva, DocuSign) | 792 | 1,532 | 6,128 |
| **total** | **5,026** | **9,456** | **37,824** |

Training covers 17 domains: 9 real services and 8 custom actions. Writing all the text, across every dataset
version, cost about $11 in API calls.

## Training

- **Loss:** a proper scoring rule over the two answer markers of each question, so the model is rewarded for
  honest probabilities, not only for the right side of 0.5.
- **Schedule:** 5 epochs, AdamW, warmup then cosine decay, effective batch 8.
- **Model selection:** the epoch with the best **validation Brier score**, not accuracy. Accuracy ignores
  calibration and picked the wrong epoch twice.
- **Calibration:** one temperature fitted on validation only, never on a test set.
- **Hardware:** one H100, 17 minutes.

## The math

What Reys computes and optimises, exactly. (Unlike write-ups that reconstruct other decision models from the
outside, this is what our code does.)

**Input.** A mandate is $x = (d, r)$: the description $d$ the person reads and the rules $r$ that govern the
permission (tools, fixed values, rule expressions), serialised to text. There are four questions
$i \in \{\text{limit}, \text{change}, \text{promise}, \text{deletion}\}$ and each label $y_i \in \{0, 1\}$ is
proven by the checker, never annotated.

**One answer.** Question $i$ is written in front of the mandate with two answer markers:

$$[\mathrm{CLS}]\; q_i\; [\mathrm{SEP}]\; [\mathrm{MASK}]\,\text{no}\; [\mathrm{MASK}]\,\text{yes}\; [\mathrm{SEP}]\; x\; [\mathrm{SEP}]$$

The encoder $f_\theta$ gives a hidden state $h$ at each marker, and a small scorer $g$ turns each into a number:
$z_i^{\text{no}}, z_i^{\text{yes}}$. With two options the softmax is a sigmoid of the margin
$s_i = z_i^{\text{yes}} - z_i^{\text{no}}$:

$$p_i(x) = P(y_i = 1 \mid x) = \sigma(s_i / T)$$

The four answers are **independent** yes/no probabilities, not one choice among four: a mandate can fail two
questions at once.

**Loss.** A strictly proper scoring rule per answer (Laya's): the log score plus half the spherical score, so the
expected reward is maximised only by reporting the true probability. With $q = (1 - p_i, p_i)$ and the true
option $y_i$:

$$\mathcal{L}_{\text{point}} = -\frac{1}{4N}\sum_{n}\sum_{i}\Big[\log q_{y_i} + \tfrac12\,\frac{q_{y_i}}{\lVert q \rVert_2}\Big]$$

**Selection and calibration.** After each epoch we score validation by Brier,
$\frac{1}{M}\sum (p - y)^2$, and keep the best epoch. Then one temperature $T$ is fitted on validation only, by
minimising cross-entropy over a grid $T \in [0.5, 5]$. The token embeddings (197M of 307M encoder parameters) are
frozen; everything else is trained with AdamW, warmup then cosine decay.

**Reading the mandate once (packed layout).** The layout above reads $x$ four times. The packed layout reads it
once, with the four question blocks after it:

$$[\mathrm{CLS}]\; x\; [\mathrm{SEP}] \;\big|\; B_1 \;\big|\; B_2 \;\big|\; B_3 \;\big|\; B_4$$

and an attention mask shaped like a tree. With $\mathrm{seg}(t) = 0$ for mandate tokens and $k$ for block $B_k$,
token $a$ may attend to token $b$ iff

$$\mathrm{seg}(b) = 0 \;\lor\; \mathrm{seg}(a) = \mathrm{seg}(b)$$

and, in the encoder's local layers, also $|\mathrm{pos}(a) - \mathrm{pos}(b)| \le 64$. Every block takes the
same position ids, right after the mandate, so each sits where it would if it were alone. All four questions fit
in ~730 tokens instead of 4 × ~390 (39 ms instead of 68 ms on a MacBook GPU). Because the mandate tokens no longer
see a question, the model has to be trained in this layout, not only served in it.

**Using the pairs (tried, not shipped).** Every training example comes as a minimal pair $(x^c, x^b)$: the correct
mandate and the same one with one thing changed. The pointwise loss scores each side alone. The pair term
adds what the pair *means*. Let $\delta_i = y_i^b - y_i^c$:

- where the change flips the answer ($\delta_i = \pm 1$), a Bradley–Terry term asks the side that should say
  "yes" to score above the other: $-\log \sigma\big(\delta_i\,(s_i^b - s_i^c)\big)$;
- where it doesn't ($\delta_i = 0$: decoys, and the questions the change doesn't touch), the harmless change
  shouldn't move the answer: $\big(\sigma(s_i^b) - \sigma(s_i^c)\big)^2$.

$$\mathcal{L} = \mathcal{L}_{\text{point}} + \lambda\, \mathcal{L}_{\text{pair}}, \qquad \lambda = 0.5$$

This uses no new data, only the structure the generator already builds. It targets the weakest number below:
pairs where only the words changed.

*First result* (packed layout, 8 epochs, $\lambda = 0.5$): it does what it was built for. Text-only pairs both
right go from 44% to 64%, and on real requests accuracy rises from 87% to 90% with calibration error 0.13 → 0.09.
But the sealed hand-written set falls to the "always fine" line (86.7%) and the real limit question to 75%, and the
fitted temperature nearly doubles: the Bradley–Terry term is unbounded in the margin, so raw scores get extreme.
Not shipped. Next: a smaller $\lambda$ or a bounded margin.

**What the math doesn't fix.** Temperature fitted on synthetic validation doesn't transfer to real requests
(overconfidence below). That's a shift between the data and the world, not a flaw in the loss, and the fix is
real, human-reviewed examples.

## Evaluation

Nothing we evaluate on was used to tune the model or the generator:

- **Unseen services:** three services left out of training entirely.
- **Hand-written mandates:** written by people, not the generator. One batch was **sealed** (checksummed) before
  training: we score on it but never read its individual errors, so it stays a test.
- **Real requests:** descriptions from real mandates on our own platform. This set never leaves our machines.

The honest baseline is "always answer fine": most descriptions are honest, so it scores high. The job is to catch
the few that aren't without crying wolf.

| | Reys | always "fine" |
|---|---|---|
| real agent-written requests, limit question (20) | 85% | 65% (today's keyword warnings: 50%) |
| sealed hand-written mandates, all four questions | 89.8% | 86.7% |
| three unseen services, all four questions | 96% (English) · 94% (Portuguese) | 86% · 80% |

On the same sealed set, a general-purpose hosted decision model used out of the box (Jev 1.13) scores 71.5%, and
the untrained Laya checkpoint 15.2%.

Speed, all four questions for one mandate on a MacBook: ~60 ms on the Apple GPU, ~210 ms on CPU. No per-check
cost, and the permission never leaves the machine.

## What we still get wrong

We measure more than accuracy, and it shows where the work is:

- **Pairs.** On the synthetic test sets, 73–75% of minimal pairs have both sides fully right, but only 44–50% when
  the only change is in the **text**. Reys is better at noticing a change in the rules than in the words.
- **Calibration outside our data.** In-distribution, a "0.9" means 90% (calibration error ≈ 0.01). On real and
  hand-written permissions Reys is overconfident: on real requests, answers given ~0.97 are right about 63% of the
  time. Warning thresholds therefore can't be set from synthetic validation; they need a human-reviewed real set.
- **Limits on brand-new permission shapes** are still close to the "always fine" line.

## What we learned

1. **The input matters more than the model.** The biggest jump on real data came from showing the model what
   the service *can* change and delete, not only the tools the sentence names (real accuracy 74% → 82%, same
   examples).
2. **The encoder isn't the bottleneck.** Laya's decision pretraining, plain mmBERT-base and the larger
   ModernBERT-large land within noise of each other; the large one is ~2.4× slower and loses Portuguese.
3. **Change the prompt, not the writer.** A writer model at 9× the price wrote nearly the same texts; asking it to
   start from a plausible task doubled the variety.
4. **Seal a test set before you look.** Once a test set guides your fixes, it stops being a test.
5. **Make every shortcut fail** before training, or the model learns the shortcut instead of the question.

## Next

- A human-reviewed test set built from real agent tasks, to measure, and to set warning thresholds.
- More text-only minimal pairs, where Reys is weakest.
- Reading the permission once for all four questions instead of once per question. A first version is
  ~1.75× faster (39 ms instead of 68 ms on a MacBook GPU) and matches on most test sets, but not yet on real
  requests, so it isn't the shipped model yet.
