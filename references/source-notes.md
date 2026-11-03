# Source and evidence boundaries

Read when explaining the research, choosing a reproduction target, or deciding whether a claimed capability is supported.

Primary source: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, November 2026. [Blog](https://rikiyat.github.io/blog/weight-space-programming/), accessed 2026-11-03. [Released experiment repository](https://github.com/RikiyaT/lm_understand-logs). Pin a repository commit before reproducing it; the main branch can change.

Source revisions at authoring: experiment logs [`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f); Anthropic skill-creator [`ca45fc496a3436251f191354091d557da31c15b5`](https://github.com/anthropics/skills/blob/ca45fc496a3436251f191354091d557da31c15b5/skills/skill-creator/SKILL.md). The blog itself is an unversioned page.

The table summarizes authors' reported results, not experiments executed by this skill's author:

| Experiment | Evidence allowed | Reported finding | Engineering implication |
|---|---|---|---|
| 1: game identification | Raw weights; matched architectures and I/O | Correct game for all four policies | Control metadata/shape leakage; weights may encode task structure |
| 2: diffusion shape | Weights, no forward pass | Identified trained geometric shapes | Structural hypotheses can precede inference; still verify separately |
| 3: character-language model | Middle transformer blocks, weight algebra, no layer calls | Identified Python/romaji/English and copying heads | Weight products can suggest circuits; coordinate labels and embeddings were absent |
| 4: car policy repair | Weights, scene, one recorded failed episode | Edited 3.5%; success in 1,024 evaluated episodes | Local control logic can be written into a small pathway; test beyond one trace |
| 5: MinAtar construction | Rules, I/O and architecture; no feedback/tools/training | Generated policies beat PPO with 100M steps in three games | Translate game computations into layers; this is not evidence for arbitrary tasks |
| 6: Go construction | Fixed architecture, CPU tensor tools, 0–50 bounded feedback passes | Stronger policies with feedback | Count games/positions and total compute, not only calls |
| 7: sycophancy | Weights only, then attention-assisted probes | Weights-only attempts failed; attention-assisted identification succeeded | Do not infer complex behavior conclusively from weight differences alone |
| 8: reasoning-loop repair | Practice problems, tensor tools, bounded forward passes; no training | Best Qwen edit changed 131,072 weights; MATH-500 7.9% → 48.1% | Removing an obstruction can expose an existing ability; evaluate completion and capability separately |
| Gemma follow-up | Similar edit approach on a model that rarely loops | No established improvement; changes within about one point | Local weight surgery may not supply missing reasoning ability |

The best Qwen run modified selected output columns in two MLPs (64 columns in each, layers 15 and 19). The architecture, dtype, selected direction and exact checkpoint matter. [Experiment 8 README](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md) distinguishes the 10-, 25- and 50-pass edits; they are not the same intervention. The 50-pass result used temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, thinking enabled and a 32,768-token limit. A different protocol is not a direct replication.

The repository includes a correction note for baseline GPQA reporting; check its current evaluation files and notes before quoting that score. Use exact revisions and raw evaluation data when precision matters.

The post's ideas about general unlearning, cross-architecture circuit grafting, hidden-capability auditing and recursive improvement are outlook, not validated general procedures. This skill deliberately makes no guarantees about them. "Any AI can load these instructions" does not mean every AI can interpret every network correctly.

This skill is an original engineering workflow informed by the sources. No third-party network, benchmark corpus or experiment code is bundled. If importing such artifacts later, preserve their licenses: the source repository distinguishes CC BY 4.0 logs/data/networks, MIT scripts and third-party material with separate terms.

Authoring method: [Anthropic skill-creator](https://skills.sh/anthropics/skills/skill-creator), selected after inspecting its source and the skill search results (422.5K installs on 2026-11-03), with the local official Codex skill-creator used for portable layout and validation. Popularity supports selection, not a proof of quality.
