"""Prompt builders + parsing for LEVER MODE (ticket D9).

When the session's derived `SearchSpace` is anything other than the single
free-form `kernel_source` lever of the `triton_source` backend, the
archetype harnesses stop asking for full kernel modules and instead ask for
a JSON list of typed interventions over the advertised levers — the same
``{"target": {"kind", "selector"}, "payload", "rationale"}`` vocabulary the
canonical `propose_candidate` tool validates (rejections come back as
retryable tool errors, i.e. feedback).

The framing mirrors the prompt content the pydantic_ai integration uses for
these backends (`integrations/python/api.py::_system_instructions` — "You
are a compiler-heuristic researcher. Replace baseline compiler decisions
with verified faster ones.") and presents the same evidence-carrying lever
catalog `inspect_search_space` serializes. Only the candidate
representation and its prompts change — the archetype TOPOLOGY (SR chain,
best-of-N fan-out, evolution archive, UCB1 bandit, CASCADE composite) is
identical to source mode by construction (see `candidate_codec.py`).
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Cap on the number of levers rendered into one prompt; the catalog notes
#: how many were elided. Both shipped lever backends derive well under this
#: (inductor ~40 curated knob/fx/lowering levers, triton ≤ ~40 pipeline
#: passes), so the cap only guards against pathological spaces.
MAX_PROMPT_LEVERS = 64

_INTERVENTION_SHAPE = (
    '{"target": {"kind": "<lever kind>", "selector": "<lever selector>"}, '
    '"payload": <value within the lever\'s range>, '
    '"rationale": "<one line: why this decision should be faster>"}'
)

RETRY_FORMAT_MESSAGE = (
    "Output exactly one fenced json code block containing ONLY a JSON array "
    f"of 1-3 interventions, each shaped {_INTERVENTION_SHAPE}."
)

MISSING_CANDIDATE_FEEDBACK = (
    "Your previous reply contained no parseable JSON intervention list. "
    + RETRY_FORMAT_MESSAGE
)

NOVELTY_FEEDBACK = (
    "Novelty filter: that intervention plan is identical (modulo rationale "
    "wording and ordering) to a candidate already REJECTED this run. "
    "Propose different levers or different values."
)


# ----------------------------------------------------------- lever catalog


def _range_detail(rng: dict[str, Any]) -> str:
    kind = str(rng.get("kind", "?"))
    if kind == "int_freeform":
        return (
            f"int_freeform min={rng.get('min')} max={rng.get('max')} "
            f"step={rng.get('step')}"
        )
    if kind == "structured_json":
        examples = list(rng.get("examples") or [])[:2]
        hint = str(rng.get("schema_hint") or "")
        parts = [f"structured_json examples={json.dumps(examples)}"]
        if hint:
            parts.append(f"schema={hint}")
        return " ".join(parts)
    candidates = rng.get("candidates")
    if candidates is not None:
        return f"{kind} candidates={json.dumps(candidates)}"
    return kind


def format_lever(lever: dict[str, Any]) -> str:
    """One catalog entry: target, typed range, default, evidence summary."""

    target = lever.get("target") or {}
    rng = lever.get("range") or {}
    evidence = lever.get("evidence") or {}
    signal = str(evidence.get("signal") or "")[:160]
    lines = [
        f"- target {target.get('kind')}:{target.get('selector')} "
        f"[{_range_detail(rng)}] default={json.dumps(lever.get('default'))}",
        f"  {str(lever.get('description') or '')[:240]}",
    ]
    if evidence.get("rule") or signal:
        lines.append(f"  evidence: {evidence.get('rule', '')} — {signal}")
    return "\n".join(lines)


def search_space_block(context: dict[str, Any]) -> str:
    """The derived lever catalog, capped at `MAX_PROMPT_LEVERS` entries."""

    levers = list(context.get("levers") or [])
    shown = levers[:MAX_PROMPT_LEVERS]
    elided = len(levers) - len(shown)
    body = "\n".join(format_lever(lv) for lv in shown)
    note = (
        f"\n(+{elided} more levers not shown — any advertised lever may be "
        "targeted.)"
        if elided > 0
        else ""
    )
    return (
        f"Derived decision levers ({len(levers)} total; every range is "
        "derived from workload analysis or device capability, never "
        f"hand-coded):\n{body}{note}"
    )


def _payload_example(lever: dict[str, Any]) -> Any:
    """A plausible non-default payload drawn from the lever's typed range."""

    rng = lever.get("range") or {}
    default = lever.get("default")
    kind = rng.get("kind")
    if kind == "bool":
        return (not default) if isinstance(default, bool) else True
    if kind in ("int_range", "float_range", "enum"):
        for candidate in rng.get("candidates") or []:
            if candidate != default:
                return candidate
        return default
    if kind == "int_freeform":
        return rng.get("min")
    if kind == "structured_json":
        examples = rng.get("examples") or []
        return examples[0] if examples else default
    return default


def example_interventions_json(levers: list[dict[str, Any]]) -> str:
    """A synthesized, catalog-grounded example of the required output."""

    picked = levers[:2] or [
        {
            "target": {"kind": "knob", "selector": "example.flag"},
            "range": {"kind": "bool"},
            "default": False,
        }
    ]
    example = [
        {
            "target": {
                "kind": (lv.get("target") or {}).get("kind", ""),
                "selector": (lv.get("target") or {}).get("selector", ""),
            },
            "payload": _payload_example(lv),
            "rationale": "why this compiler decision should be faster here",
        }
        for lv in picked
    ]
    return json.dumps(example, indent=1)


# ------------------------------------------------------------ base prompt


def _workload_header(context: dict[str, Any]) -> str:
    device = context.get("device") or {}
    baseline = context.get("baseline_median_ms")
    analysis = context.get("analysis_summary") or {}
    analysis_text = json.dumps(analysis, default=str)[:1200]
    return (
        f"Workload: id={context.get('workload_id')}, "
        f"kind={context.get('workload_kind')}, "
        f"backend_id={context.get('backend_id')}.\n"
        f"Task: {context.get('task_description')}\n"
        f"Baseline median latency: {baseline} ms — the EMPTY intervention "
        "list, i.e. the compiler's stock heuristics. Speedups are measured "
        "against it.\n"
        f"Device: {device.get('arch')} ({device.get('name')}).\n"
        f"Analysis summary: {analysis_text}"
    )


def base_prompt(context: dict[str, Any]) -> str:
    """The lever-mode counterpart of `prompts.base_prompt`."""

    levers = list(context.get("levers") or [])
    return f"""You are a compiler-heuristic researcher. Replace baseline compiler decisions with verified faster ones.

{_workload_header(context)}

The program/kernel source is FIXED — do NOT write kernel or module code. Your only decision space is the typed compiler levers below: each intervention overrides one compiler decision for one candidate compile, and the backend validates every intervention (invalid ones are rejected with an explanation and cost no budget).

{search_space_block(context)}

Propose ONE candidate: a JSON array of 1-3 interventions, each shaped
{_INTERVENTION_SHAPE}
Rules:
- target kind/selector should name an advertised lever (the backend may accept off-catalog targets of a known kind, but expect rejections otherwise).
- payload must fit the lever's typed range: bool levers take true/false, int/float/enum levers take one of the listed candidates, int_freeform takes any step-aligned integer in [min, max], structured_json follows the schema hint.
- Prefer levers whose evidence ties them to this workload's measured signals.
- Output exactly ONE fenced json code block containing ONLY the JSON array, nothing else.

Example of the required output format (values illustrative, drawn from this catalog):
```json
{example_interventions_json(levers)}
```"""


#: Lever-space optimization directions for menu-dropout prompt variation —
#: the lever-mode counterpart of `prompts.OPTIMIZATION_MENU`, phrased over
#: lever KINDS so the same menu applies to knob spaces and pass pipelines.
OPTIMIZATION_MENU: tuple[str, ...] = (
    "Flip boolean levers that gate fusion, autotuning, padding, or layout "
    "decisions away from their defaults.",
    "Scale ONE numeric lever (thresholds, pipeline depth such as "
    "num_stages, tile/block parameters) up or down, sweeping powers of two "
    "around the default.",
    "Pick a non-default member of an enum lever (alternative algorithm, "
    "backend, or search-space setting).",
    "Skip/disable one pass or decision whose evidence shows it modified "
    "the IR, to test whether it pays for itself on this workload.",
    "Parameterize the pass/knob the evidence marks as most load-bearing "
    "for this workload's shapes.",
    "Combine two levers whose evidence cites the same signal (e.g. fusion "
    "plus reordering) into one candidate.",
)


def menu_dropout_prompt(
    context: dict[str, Any],
    *,
    rng: Any,
    keep_probability: float = 0.5,
) -> str:
    """`base_prompt` + a randomly-thinned lever-space optimization menu."""

    kept = [item for item in OPTIMIZATION_MENU if rng.random() < keep_probability]
    if not kept:
        kept = [rng.choice(OPTIMIZATION_MENU)]
    menu = "\n".join(f"- {item}" for item in kept)
    return (
        base_prompt(context)
        + f"\nFocus especially on these optimization directions:\n{menu}\n"
    )


#: The 5 lever-space strategy arms (UCB1 bandit) — same arity and role as
#: the source-mode `bandit.STRATEGY_ARMS`, phrased over lever kinds.
STRATEGY_ARMS: tuple[tuple[str, str], ...] = (
    (
        "flip_flags",
        "FLIP-FLAGS: toggle one or two boolean levers (fusion, autotune, "
        "padding, reordering flags) away from their defaults.",
    ),
    (
        "scale_numeric",
        "SCALE-NUMERIC: change exactly one numeric lever's value — double "
        "or halve a threshold/depth (e.g. num_stages, fusion thresholds) "
        "relative to the incumbent plan or the default.",
    ),
    (
        "switch_enum",
        "SWITCH-ENUM: select a different candidate of one enum lever "
        "(alternative algorithm/backend/search-space choice).",
    ),
    (
        "skip_decision",
        "SKIP-DECISION: disable/skip one pass or optimization decision the "
        "evidence shows is active on this workload, to test whether it "
        "pays for itself.",
    ),
    (
        "combine_levers",
        "COMBINE: pair the two levers whose evidence looks most "
        "load-bearing for this workload into one 2-intervention plan.",
    ),
)


# ----------------------------------------------------- parsing / encoding


def _coerce_interventions(parsed: Any) -> list[dict[str, Any]] | None:
    """Coerce parsed JSON into the canonical intervention-list shape.

    Accepts a bare array, ``{"interventions": [...]}``, or entries using
    either the nested ``{"target": {"kind", "selector"}}`` shape or the
    flat ``target_kind``/``target_selector`` shape the session tool takes.
    Returns None when anything is malformed (the retryable-format path).
    """

    if isinstance(parsed, dict) and isinstance(parsed.get("interventions"), list):
        parsed = parsed["interventions"]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return None
    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        target = item.get("target")
        if isinstance(target, dict):
            kind = target.get("kind")
            selector = target.get("selector", "")
        else:
            kind = item.get("target_kind")
            selector = item.get("target_selector", "")
        if not kind or not isinstance(kind, str):
            return None
        out.append(
            {
                "target": {"kind": kind, "selector": str(selector or "")},
                "payload": item.get("payload"),
                "rationale": str(item.get("rationale") or ""),
            }
        )
    return out


def extract_interventions(text: str) -> str | None:
    """Parse an intervention list out of LLM text → canonical JSON string.

    Mirrors `prompts.extract_code`'s contract: returns the candidate text
    (here: a re-serialized canonical JSON array) or None when no parseable
    candidate exists — None routes to the retry-format feedback path.
    Fenced blocks are tried largest-first, then any bare top-level JSON
    array in the text.
    """

    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    bare = re.findall(r"\[.*\]", text, re.DOTALL)
    for block in sorted(blocks, key=len, reverse=True) + bare:
        try:
            parsed = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        interventions = _coerce_interventions(parsed)
        if interventions:
            return json.dumps(interventions, indent=1)
    return None


def propose_args(candidate: str, *, description: str) -> dict[str, Any]:
    """Canonical-JSON candidate → `propose_candidate` tool arguments."""

    interventions = json.loads(candidate)
    return {
        "interventions": [
            {
                "target_kind": iv["target"]["kind"],
                "target_selector": iv["target"]["selector"],
                "payload": iv.get("payload"),
                "rationale": iv.get("rationale") or description,
            }
            for iv in interventions
        ],
        "description": description,
        "expected_effect": "",
    }


def normalize_interventions(candidate: str) -> str:
    """Novelty-filter normalization (C7) for intervention plans.

    Rationale text is dropped (rewording a rejected plan does not make it
    novel) and interventions are sorted by (kind, selector, payload), so
    permuted duplicates collide. Unparseable text falls back to whitespace
    collapse, mirroring `normalize_module_source`'s defensive path.
    """

    try:
        interventions = json.loads(candidate)
        stripped = sorted(
            (
                {
                    "target": iv.get("target"),
                    "payload": iv.get("payload"),
                }
                for iv in interventions
            ),
            key=lambda iv: json.dumps(iv, sort_keys=True, default=str),
        )
        return json.dumps(stripped, sort_keys=True, default=str)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return " ".join(str(candidate).split())


# --------------------------------------------------------------- feedback


def rejection_feedback(error: str) -> str:
    """Feedback when `propose_candidate` rejects the plan (validate path)."""

    return (
        "Your intervention plan was rejected before reaching the compiler: "
        f"{error}\n"
        "Fix it against the advertised lever catalog and output the "
        "corrected JSON array in one fenced json code block."
    )


def feedback_for_run_result(result: dict[str, Any]) -> str:
    """Structured verdict feedback — the lever-mode counterpart of
    `prompts.feedback_for_run_result` (same three verdicts)."""

    if not result.get("compile_ok"):
        return (
            "Your intervention plan made the compile FAIL. Error:\n"
            f"{result.get('compile_diagnostics')}\n"
            "Choose different levers/values and output the corrected JSON "
            "array in one fenced json code block."
        )
    if result.get("correctness_ok") is False:
        gate_lines = [
            w
            for w in (result.get("compile_warnings") or [])
            if "FAILED" in str(w)
        ]
        detail = (
            " ".join(gate_lines)
            or f"max_abs_diff={result.get('max_abs_diff')}"
        )
        return (
            "Your plan compiled but CHANGED NUMERICS vs the baseline: "
            f"{detail} Pick levers that preserve semantics and output a new "
            "JSON array in one fenced json code block."
        )
    cand_ms = result.get("median_ms")
    speedup = result.get("speedup_vs_baseline")
    ref_ms = (
        cand_ms * speedup
        if isinstance(cand_ms, (int, float)) and isinstance(speedup, (int, float))
        else None
    )
    if cand_ms is None or ref_ms is None:
        return (
            "Your plan ran but produced no timing signal. Propose a "
            "different intervention plan in one fenced json code block."
        )
    return (
        f"Your plan is VALID: latency {cand_ms:.4f} ms vs baseline "
        f"{ref_ms:.4f} ms (speedup {speedup:.3f}x). Make it faster — try "
        "different levers, different values on the same levers, or a "
        "2-lever combination — and output the improved JSON array in one "
        "fenced json code block."
    )


# ---------------------------------------- evolution / bandit / cascade


def contrastive_pair_prompt(
    context: dict[str, Any],
    *,
    best: Any,
    divergent: Any,
    mode: str,
) -> str:
    """Crossover/mutation prompt over two measured intervention plans."""

    instruction = {
        "crossover": (
            "Produce a CROSSOVER child: combine the strengths of plan A "
            "and plan B into one faster intervention plan."
        ),
        "mutation": (
            "Produce a MUTATION child: keep plan A's working levers but "
            "change one significant decision (a different value, an "
            "added/removed lever) to try to beat it."
        ),
    }[mode]
    return (
        base_prompt(context)
        + f"""
Two parent intervention plans from the current population, with measured results:

Plan A (best overall) — {best.summary}:
```json
{best.source}
```

Plan B (divergent) — {divergent.summary}:
```json
{divergent.source}
```

{instruction}
Output exactly ONE fenced json code block with the full intervention array, nothing else.
"""
    )


def strategy_prompt(
    context: dict[str, Any],
    *,
    incumbent: str,
    incumbent_speedup: float,
    strategy_text: str,
) -> str:
    """One bandit pull's prompt: incumbent plan + the chosen strategy arm."""

    return (
        base_prompt(context)
        + f"""
Current incumbent intervention plan (validated, speedup {incumbent_speedup:.3f}x vs baseline; [] = the stock compiler heuristics):
```json
{incumbent}
```

Apply exactly this optimization strategy to the incumbent:
{strategy_text}

Output exactly ONE fenced json code block with the full improved intervention array, nothing else.
"""
    )


def incumbent_block(incumbent: Any, *, last_delta_pct: float | None) -> str:
    """C3 — current best plan + timing + last attempt's delta."""

    if incumbent is None:
        return (
            "\nNo candidate has validated yet — the incumbent is the "
            "baseline compiler configuration itself (the empty intervention "
            "list, speedup 1.000x by definition).\n"
        )
    delta_line = (
        f"Your last attempt was {last_delta_pct:+.2f}% vs this incumbent.\n"
        if last_delta_pct is not None
        else ""
    )
    return f"""
Current incumbent (best validated intervention plan) — {incumbent.summary}:
```json
{incumbent.source}
```
{delta_line}"""


def plan_prompt(context_block: str) -> str:
    """C1 phase 1 — one named lever change, justified, NO JSON yet."""

    return (
        context_block
        + "\nState a PLAN for the next candidate: name exactly ONE lever "
        "change to apply (e.g. 'set knob inductor.max_autotune = true', "
        "'skip ttgir:tritongpu-prefetch', 'pipeline num_stages 3 -> 2') and "
        "justify it in at most 3 sentences. Do NOT write the JSON yet."
    )


def implement_prompt(context_block: str, plan: str) -> str:
    """C1 phase 2 — implement exactly the stated lever change."""

    return (
        context_block
        + f"\nYour plan for this attempt:\n{plan}\n\n"
        "Implement exactly this plan. Output exactly ONE fenced json code "
        "block with the full intervention array, nothing else."
    )


def judge_prompt(
    task_description: str, candidates: list[tuple[str, str]]
) -> str:
    """C5 — rank validated plans by PREDICTED gain (no timings shown)."""

    blocks = "\n\n".join(
        f"Candidate {cid}:\n```json\n{source}\n```"
        for cid, source in candidates
    )
    ids = [cid for cid, _ in candidates]
    return f"""You are judging compiler intervention plans for this task: {task_description}

All candidate plans below compiled and validated. Rank them by PREDICTED
performance gain (fastest first), considering which compiler decisions
(fusion, scheduling, pipeline depth, layout, autotuning) plausibly dominate
this workload's runtime.

{blocks}

Respond with ONLY a JSON array of candidate ids, best first, e.g.
["{ids[0]}", ...]. No other text."""


PROPOSE_INSTRUCTION = (
    "\nPropose the next improved intervention plan. Output exactly ONE "
    "fenced json code block with the full intervention array, nothing else."
)
