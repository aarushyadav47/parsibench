# ParsiBench

**Method-Supervised Tool Use for LLM Agents**

ParsiBench is a deterministic benchmark and training harness for evaluating and improving **tool-use efficiency** in LLM agents. It targets a common failure mode in deployed agents: getting the correct outcome while using a **wasteful or redundant method** (unnecessary tool calls, repeated queries, oversampling).

The core idea is **process supervision for tool use**: we label each tool call as *necessary*, *avoidable*, or *redundant* using task “certificates,” then train models and controllers that learn **when not to call tools** while maintaining task success.

---

## Why this exists

Most agent evaluations emphasize final task completion. In production, the cost often comes from:

* repeated tool calls (re-fetching the same information)
* “panic search” loops
* calling tools after the answer is already supported
* oversampling in statistical decision tasks

ParsiBench makes those method errors measurable and optimizable.

---

## What’s in this repo

### Benchmark

Three deterministic task families (v0):

* **`kb` (Knowledge Boundary):** some questions are answerable from the prompt alone; others require retrieval via tools.
* **`evmin` (Evidence Minimal Retrieval):** tasks require exactly **m = 2 or 3** evidence spans, each located in different docs.
* **`sample95` (Noisy Sampling @ 95%):** decide whether stream A or B has higher mean by at least δ with 95% confidence, using sequential sampling and early stopping.

### Tools

A small deterministic sandbox:

* `kb_search(query, k)` (TF-IDF retrieval over per-task docs)
* `kb_fetch(doc_id, start, end)` (exact substring span)
* `sim_sample(stream, n)` (seeded Gaussian sampler)

### Agents

* `react`: LLM baseline with tool calling
* `voi`: VoI-style gating: stop calling tools when expected benefit is low
* `prm`: Tool-call PRM reranker that discourages avoidable/redundant calls
* `voi_prm`: combined VoI gate + PRM-scored action selection

### Labels and training

* **Certified per-call labels** (v0): `necessary`, `avoidable`, `redundant`
* Training scripts for:

  * a **Tool-call PRM** (multiclass classifier)
  * a **NeedTool** model (binary classifier for “tool needed now”)

---

## Results (example)

Below are the target “ideal” outcomes once the benchmark enforces tool necessity and evidence coverage:

|   Agent | Family   |   Success | Avg Tool Calls | Avg Avoidable Calls |
| ------: | :------- | --------: | -------------: | ------------------: |
|   react | kb       | 0.98–1.00 |        0.6–0.9 |           0.10–0.25 |
| voi_prm | kb       | 0.98–1.00 |    **0.5–0.8** |       **0.05–0.15** |
|   react | evmin    | 0.95–1.00 |        2.8–4.2 |             0.6–1.8 |
| voi_prm | evmin    | 0.95–1.00 |    **2.2–3.2** |         **0.0–0.6** |
|   react | sample95 | 0.85–0.97 |            3–8 |                 1–4 |
| voi_prm | sample95 | 0.83–0.95 |      **2.5–6** |         **0.8–2.5** |

The key story: **voi_prm shifts the success–cost Pareto frontier** by preserving accuracy while reducing avoidable tool calls.

---

## Method: Certified labels

ParsiBench v0 uses “certificates” embedded in each task instance to label tool calls.

### Label definitions

* **necessary:** contributes required evidence or samples up to the minimal threshold
* **avoidable:** not needed (task solvable without it) or called after sufficiency reached
* **redundant:** repeats a semantically equivalent earlier call (after canonicalization)

### Certified labeling rules (high level)

* **kb:** if `requires_tool=False`, any tool call is avoidable; if `True`, the first successful retrieval/fetch is necessary.
* **evmin:** a fetch from a required doc that has not been covered yet is necessary; additional fetches are avoidable/redundant.
* **sample95:** sampling is necessary until `n_min` is reached; additional samples are avoidable.

v0 intentionally prioritizes **determinism and dense supervision** over realism. Counterfactual labeling is a planned extension.

---

## Agent details

### `react` baseline

An LLM agent that:

* chooses to call tools or answer
* uses function calling for tool execution
* produces a final structured JSON answer:

  * `final_answer`
  * `evidence_refs`
  * `confidence`
  * `stop_reason`

### `voi` (Value-of-Information gating)

A lightweight controller that decides whether calling a tool is worth it based on:

* predicted tool necessity (`NeedTool`)
* current progress and uncertainty proxies
* a cost term

### `prm` (Tool-call PRM reranker)

A trained classifier that scores a *proposed* tool call as likely:

* necessary vs avoidable vs redundant

We use PRM scores to pick among candidate tool actions.

### `voi_prm`

Combines both:

1. VoI decides whether to stop
2. PRM helps choose the best next tool call when the agent should continue

---

## Evaluation metrics

Per episode:

* `success`
* `tool_calls_total`
* `avoidable_calls_total`
* `redundancy_count`
* `time_ms`

Aggregate:

* success rate
* average tool calls
* average avoidable calls
* Pareto frontier (success vs tool calls)

---

## Reproducibility

ParsiBench is deterministic end-to-end for fixed seeds:

* Task generators are seeded.
* Tool outputs are deterministic (TF-IDF retrieval + seeded sampling).
* All outputs are written as JSONL traces.
* Recommended: run LLM calls with deterministic decoding settings when available.

---

## Project structure

* `parsibench/tasks/` task generators
* `parsibench/tools/` sandbox tools
* `parsibench/agents/` agent implementations
* `parsibench/eval/` runner, labels, metrics, report
* `parsibench/train/` PRM + NeedTool training pipelines
* `data/` generated datasets
* `runs/` JSONL traces
* `artifacts/` trained models
* `report/` plots and summary

---

## Common pitfalls (and how to avoid them)

### sample95 “free accuracy”

If `react / sample95` achieves high success with **0 tool calls**, you have leakage or a broken evaluator.
Fix by enforcing: success requires sampling, and ensure the prompt does not contain winner signals.

### evmin “everything labeled avoidable”

If nearly all evmin calls are labeled avoidable while success is 1.0, your span overlap matching is too strict or evidence is not required for success.
Fix by:

* requiring evidence coverage in evaluation, and
* treating “fetch from required doc not yet covered” as necessary (ignore span overlap in v0).

---

## Extending ParsiBench

Planned extensions:

* **Counterfactual necessity labeling** (remove a tool call and test if success remains reachable)
* **More realistic tools** (SQL, file I/O, multi-document synthesis)
* **Dynamic tool costs** (latency/cost shifting) and robustness evaluation
* **Leaderboard + standardized submission format**

---

## Citation

If you use this repo in your work, please cite:

```
@misc{parsibench2025,
  title={ParsiBench: Method-Supervised Tool Use for LLM Agents},
  author={Aarush Yadav},
  year={2025},
  howpublished={GitHub repository}
}
```

---

## License

MIT (or your preferred license).

---

## Contact

Open an issue for questions, reproduction problems, or benchmark extensions.
