# Reys

> *Sê todo em cada coisa. Põe quanto és*
> *no mínimo que fazes.*
>
> — Ricardo Reis

<p align="center">
  <img src="./Reys.gif" width="100%" alt="Reys">
</p>

When an AI agent asks for permission, a person approves a **sentence**. What the agent can actually do is decided
by **rules**. Reys is a small Decision Language Model by [Humanos](https://humanos.tech) that checks the two
match, by answering four questions about every permission:

1. **A limit that isn't enforced**: the words promise a limit the rules don't check.
2. **A change nobody mentioned**: the rules allow a write the words don't mention.
3. **A promise it can't keep**: the words promise something the rules never allow.
4. **A deletion nobody mentioned**: the rules allow something destructive the words never say.

322M parameters, ~60 ms for all four questions on a laptop, no LLM call. It never grants or blocks anything:
the rules still decide, and Reys checks that the words describe them honestly. It's a research prototype;
the numbers are early.

**Read more:**
- [Meet Reys](https://humanos-app.github.io/reys/): what it does, in plain language
- [Technical details](https://humanos-app.github.io/reys/technical.html): real outputs, results against Jev
  and Laya, and how it was built

<p align="right">
  <sub>LISBONAI · 2026</sub>
</p>
