"""Step 2a: turn a free-text objective into a TaskSpec.

The heuristic parser here is transparent pattern matching: every task it assigns
is backed by the phrases that triggered it, recorded in ``TaskSpec.evidence`` and
hence in the decision log. When an LLM is available it refines this spec
(planner/llm.py), and the refinement is validated against the same schema.

Precedence when several tasks match follows how specific the request is: asking
to impute or detect anomalies names a concrete non-generative end, so it wins
over a generic "generate"; "rebalance rare classes" is more specific than
"augment", which is more specific than "generate".
"""

from __future__ import annotations

import re

from .schema import DataProfile, TaskSpec

PATTERNS: dict[str, list[str]] = {
    "imputation": [r"imput\w*", r"missing (?:values?|entries|data|measurements?)",
                   r"fill(?:ing)? in", r"complete the (?:missing|data)"],
    "anomaly_detection": [r"anomal\w*", r"outliers?", r"novelty", r"\bunusual\b",
                          r"\babnormal\b", r"\bfraud\w*"],
    "uncertainty_quantification": [r"uncertaint\w*", r"confidence intervals?",
                                   r"credible intervals?", r"prediction intervals?",
                                   r"calibrat\w*"],
    "representation_learning": [r"representations?", r"embeddings?", r"latent space",
                                r"low[- ]dimensional", r"feature learning"],
    "class_balancing": [r"re-?balanc\w*", r"imbalanc\w*", r"under-?represented",
                        r"minority", r"\brare\b", r"oversampl\w*", r"balance the classes"],
    "augmentation": [r"augment\w*", r"more training data", r"\bscarce\b",
                     r"small labell?ed", r"improve\w* (?:a |the )?(?:downstream )?classif\w*",
                     r"boost\w*"],
    "generation": [r"generat\w*", r"synthe\w+", r"simulat\w*", r"sample new"],
}
PRECEDENCE = ["imputation", "anomaly_detection", "uncertainty_quantification",
              "class_balancing", "augmentation", "representation_learning", "generation"]

CONDITIONAL = [r"class-?conditional", r"\bconditional\b", r"per[- ]class", r"for each class",
               r"for (?:each|every|rare|specific) (?:cell )?types?", r"cell[- ]types?"]
CLASSIFICATION = [r"classif\w*", r"\bpredict\w*", r"diagnos\w*", r"\bdownstream\b"]
TSTR = [r"synthetic (?:data )?alone", r"train(?:ed|ing)? (?:only )?on synthetic",
        r"\btstr\b", r"synthetic-only"]
PRIVACY = [r"privacy", r"\bshar(?:e|ing)\b", r"de-?identif\w*", r"release"]


def _find(patterns: list[str], text: str) -> list[str]:
    return [m.group(0) for p in patterns for m in re.finditer(p, text)]


def _label_budget(text: str) -> tuple[float | None, int | None]:
    """'~5% of training patches', '1% of labels' -> 0.05; '500 per class' -> 500."""
    frac = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:the\s+)?(?:\w+\s+){0,2}"
                  r"(?:labels?|labell?ed|training|patches|samples|images|cells|data)", text)
    if m and re.search(r"label|subset|available|scarce|only", text):
        frac = float(m.group(1)) / 100
    per_class = None
    m = re.search(r"(\d[\d,]*)\s+(?:labell?ed\s+)?\w*\s*per class", text)
    if m:
        per_class = int(m.group(1).replace(",", ""))
    return frac, per_class


def parse_objective(objective: str, profile: DataProfile | None = None) -> TaskSpec:
    text = objective.lower()
    evidence = {task: hits for task, pats in PATTERNS.items() if (hits := _find(pats, text))}

    matched = [t for t in PRECEDENCE if t in evidence]
    task = matched[0] if matched else "generation"
    secondary = [t for t in matched[1:]]

    target_classes: list[str] = []
    if profile and profile.labels:
        for name in profile.labels.class_names:
            # Whole-word match only, and never bare digits ("5%" is not class "5").
            if not name.isdigit() and re.search(rf"\b{re.escape(name.lower())}\b", text):
                target_classes.append(name)

    frac, per_class = _label_budget(text)
    wants_rare = bool(_find(PATTERNS["class_balancing"], text))
    conditional = bool(_find(CONDITIONAL, text)) or task == "class_balancing" or bool(target_classes)

    return TaskSpec(
        raw_objective=objective,
        task_type=task,
        secondary_tasks=secondary,
        conditional=conditional,
        target_classes=target_classes,
        wants_rare_classes=wants_rare,
        label_budget=frac,
        label_budget_per_class=per_class,
        downstream_task="classification" if _find(CLASSIFICATION, text) or task in
        ("class_balancing", "augmentation") else "none",
        wants_tstr=bool(_find(TSTR, text)),
        privacy_sensitive=bool(_find(PRIVACY, text)),
        evidence=evidence,
        source="heuristic",
    )
