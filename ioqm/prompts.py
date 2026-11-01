"""
Structured prompts for IOQM.

Ground truth now lives in a PromptSpec (canonical class -> count), never in the
prompt text or the image filename. Prompt text is *rendered* from a spec, so the
same spec can be rendered several ways (joiner, context template, numerals)
for the prompt-form sensitivity study.

Also contains:
  - parse_prompt: a parser for legacy (2023) prompt strings and filenames,
    built on the canonical class table.
  - build_v3: the nested-exhaustive v3 prompt set with its train / test-ID /
    test-OOD splits.
  - stratified_subset: draws the ~3k-prompt evaluation subset.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .classes import SURFACE_FORMS_BY_LENGTH, SURFACE_TO_NAME, _norm, names, prompt_noun

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
WORD_TO_NUMBER = {v: k for k, v in NUMBER_WORDS.items()}
WORD_TO_NUMBER.update({"a": 1, "an": 1})

SPLITS = ("train", "test_id", "test_single", "test_ood_pair", "test_ood_triple", "test_ood_count")


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    objects: Tuple[Tuple[str, int], ...]   # ((canonical_name, count), ...) in prompt order
    split: str = "unassigned"
    dataset: str = "v3"

    @property
    def expected(self) -> Dict[str, int]:
        return dict(self.objects)

    @property
    def n_types(self) -> int:
        return len(self.objects)

    @property
    def max_count(self) -> int:
        return max(c for _, c in self.objects)

    @property
    def total_instances(self) -> int:
        return sum(c for _, c in self.objects)

    @property
    def class_set(self) -> Tuple[str, ...]:
        return tuple(sorted(n for n, _ in self.objects))

    def to_json(self) -> dict:
        d = asdict(self)
        d["objects"] = [list(o) for o in self.objects]
        return d

    @staticmethod
    def from_json(d: Mapping) -> "PromptSpec":
        return PromptSpec(
            prompt_id=d["prompt_id"],
            objects=tuple((str(n), int(c)) for n, c in d["objects"]),
            split=d.get("split", "unassigned"),
            dataset=d.get("dataset", "v3"),
        )


def make_prompt_id(objects: Sequence[Tuple[str, int]], dataset: str = "v3") -> str:
    key = dataset + "|" + "|".join(f"{n}:{c}" for n, c in objects)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def make_spec(objects: Sequence[Tuple[str, int]], dataset: str = "v3", split: str = "unassigned") -> PromptSpec:
    objs = tuple((names([n])[0], int(c)) for n, c in objects)
    if len({n for n, _ in objs}) != len(objs):
        raise ValueError(f"duplicate class in prompt: {objs}")
    if any(c < 1 for _, c in objs):
        raise ValueError(f"counts must be >= 1: {objs}")
    return PromptSpec(make_prompt_id(objs, dataset), objs, split, dataset)


# ---------------------------------------------------------------- rendering

def render(
    spec: PromptSpec,
    template: str = "{objects}",
    joiner: str = " and ",
    numerals: str = "digits",
) -> str:
    """
    template: must contain "{objects}". Examples: "{objects}",
              "a photo of {objects}", "{objects} on a table."
    joiner:   " and " or ", "
    numerals: "digits" ("3 cups") or "words" ("three cups")
    """
    if "{objects}" not in template:
        raise ValueError("template must contain {objects}")
    parts = []
    for name, count in spec.objects:
        num = str(count) if numerals == "digits" else NUMBER_WORDS[count]
        parts.append(f"{num} {prompt_noun(name, count)}")
    return template.format(objects=joiner.join(parts))


# ------------------------------------------------------- legacy prompt parsing

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


def parse_prompt(text: str, strict: bool = True) -> Dict[str, int]:
    """
    Parse "5 wine glasses and 5 cups and 4 umbrellas" (or a legacy filename
    stem with underscores, or a prompt with a context suffix such as
    "... on a table.") into {canonical_name: count}.

    The object noun is found by longest match against the canonical surface
    forms, starting right after each number. Words that are neither a number
    nor part of a class name ("and", "on", "a", "table") are skipped.
    """
    text = text.replace("_", " ")
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    out: Dict[str, int] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        count = int(tok) if tok.isdigit() else WORD_TO_NUMBER.get(tok) if tok not in ("a", "an") else None
        if count is None:
            i += 1
            continue
        # longest surface form starting at tokens[i+1]
        match_name, match_len = None, 0
        for n_words in range(min(4, len(tokens) - i - 1), 0, -1):
            window = tokens[i + 1:i + 1 + n_words]
            if any(t.isdigit() or t in NUMBER_WORDS.values() for t in window):
                continue  # a class name never spans the next count
            cand = _norm(" ".join(window))
            if cand in SURFACE_TO_NAME:
                match_name, match_len = SURFACE_TO_NAME[cand], n_words
                break
        if match_name is None:
            if strict:
                raise ValueError(f"no COCO class after count {tok!r} in {text!r}")
            i += 1
            continue
        if match_name in out:
            raise ValueError(f"class {match_name!r} appears twice in {text!r}")
        out[match_name] = count
        i += 1 + match_len
    if strict and not out:
        raise ValueError(f"no objects parsed from {text!r}")
    return out


# ------------------------------------------------------------- v3 builder

def enumerate_objects(classes: Sequence[str], n_types: int, max_count: int) -> Iterable[Tuple[Tuple[str, int], ...]]:
    for combo in combinations(classes, n_types):
        for counts in product(range(1, max_count + 1), repeat=n_types):
            yield tuple(zip(combo, counts))


def at_least_two_at_max(max_count: int) -> Callable[[Sequence[Tuple[str, int]]], bool]:
    """Default held-out count rule: two or more types at the maximum count."""
    return lambda objects: sum(1 for _, c in objects if c == max_count) >= 2


def build_v3(
    pair_classes: Sequence[str],
    triple_classes: Sequence[str],
    max_count: int = 5,
    ood_pair_frac: float = 0.2,
    ood_pair_frac_in_triple: float = 0.1,
    ood_triple_frac: float = 0.2,
    test_id_frac: float = 0.2,
    ood_count_rule: Optional[Callable[[Sequence[Tuple[str, int]]], bool]] = None,
    forced_ood_pairs: Iterable[Iterable[str]] = (),
    seed: int = 0,
) -> List[PromptSpec]:
    """
    Nested exhaustive design:
      singles  over pair_classes                    (|P| * max_count)
      pairs    over pair_classes                    (C(|P|,2) * max_count^2)
      triples  over triple_classes (subset of P)    (C(|T|,3) * max_count^3)

    Splits (assigned in this priority order):
      test_single     every single-class prompt. Gives per-class detectability
                      and per-class threshold calibration. Not OOD.
      test_ood_pair   the prompt CONTAINS a held-out class pair. Held-out pairs
                      never co-occur in ANY train prompt, including inside
                      triples. Tests: a pair never seen together.
                      Held-out pairs = forced_ood_pairs (e.g. the rarest COCO
                      co-occurrences) + a random ood_pair_frac of pairs not
                      inside triple_classes + a random ood_pair_frac_in_triple
                      of pairs inside triple_classes. That second fraction is
                      lower because each held pair there removes
                      (|T|-2) triple sets from train.
      test_ood_triple a triple set whose three pairs are all seen in train,
                      but which is itself never seen (a random ood_triple_frac
                      of the remaining triple sets). Tests: a new triple made
                      of familiar pairs.
      test_ood_count  the count tuple matches ood_count_rule (default: two or
                      more types at max_count).
      test_id / train the rest, split at random by test_id_frac. test_id class
                      sets and count patterns all appear in train.
    """
    pair_classes = names(pair_classes)
    triple_classes = names(triple_classes)
    if not set(triple_classes) <= set(pair_classes):
        raise ValueError("triple_classes must be a subset of pair_classes")
    rule = ood_count_rule or at_least_two_at_max(max_count)
    rng = random.Random(seed)
    tset = set(triple_classes)

    pair_sets = [tuple(sorted(c)) for c in combinations(pair_classes, 2)]
    in_triple = [p for p in pair_sets if set(p) <= tset]
    outside = [p for p in pair_sets if not set(p) <= tset]
    held_pairs = {tuple(sorted(names(c))) for c in forced_ood_pairs}
    if any(len(p) != 2 for p in held_pairs):
        raise ValueError("forced_ood_pairs must be pairs")
    held_pairs |= set(rng.sample(outside, round(ood_pair_frac * len(outside))))
    held_pairs |= set(rng.sample(in_triple, round(ood_pair_frac_in_triple * len(in_triple))))

    def has_held_pair(cset: Tuple[str, ...]) -> bool:
        return any(tuple(sorted(p)) in held_pairs for p in combinations(cset, 2))

    triple_sets = [tuple(sorted(c)) for c in combinations(triple_classes, 3)]
    clean_triples = [t for t in triple_sets if not has_held_pair(t)]
    held_triples = set(rng.sample(clean_triples, round(ood_triple_frac * len(clean_triples))))

    specs: List[PromptSpec] = []
    for n_types, pool in ((1, pair_classes), (2, pair_classes), (3, triple_classes)):
        for objs in enumerate_objects(pool, n_types, max_count):
            cset = tuple(sorted(n for n, _ in objs))
            if n_types == 1:
                split = "test_single"
            elif has_held_pair(cset):
                split = "test_ood_pair"
            elif cset in held_triples:
                split = "test_ood_triple"
            elif rule(objs):
                split = "test_ood_count"
            else:
                split = "test_id" if rng.random() < test_id_frac else "train"
            specs.append(make_spec(objs, "v3", split))
    return specs


def stratified_subset(
    specs: Sequence[PromptSpec],
    n: int,
    key: Callable[[PromptSpec], Tuple] = lambda s: (s.split, s.n_types, s.max_count),
    seed: int = 0,
) -> List[PromptSpec]:
    """Draw about n prompts with roughly equal allocation per stratum. A stratum
    smaller than its share is taken whole, and the leftover is spread over the
    other strata."""
    rng = random.Random(seed)
    strata: Dict[Tuple, List[PromptSpec]] = {}
    for s in specs:
        strata.setdefault(key(s), []).append(s)
    remaining, budget, chosen = dict(strata), n, []
    while remaining and budget > 0:
        share = budget // len(remaining) or 1
        small = {k: v for k, v in remaining.items() if len(v) <= share}
        if not small:
            for k in sorted(remaining):
                chosen.extend(rng.sample(remaining[k], min(share, len(remaining[k]), budget)))
                budget -= min(share, len(remaining[k]))
                if budget <= 0:
                    break
            break
        for k, v in small.items():
            chosen.extend(v)
            budget -= len(v)
            del remaining[k]
    return sorted(chosen, key=lambda s: s.prompt_id)


# ------------------------------------------------------------------ I/O

def save_specs(specs: Iterable[PromptSpec], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in specs:
            f.write(json.dumps(s.to_json()) + "\n")


def load_specs(path: Path) -> List[PromptSpec]:
    with open(path) as f:
        return [PromptSpec.from_json(json.loads(line)) for line in f if line.strip()]


def legacy_specs(prompt_lines: Iterable[str], dataset: str) -> List[PromptSpec]:
    """Convert a 2023 prompts_v1.txt / prompts_v2.txt into specs, keeping the
    original prompt order."""
    out = []
    for line in prompt_lines:
        line = line.strip()
        if not line:
            continue
        objs = list(parse_prompt(line).items())
        out.append(make_spec(objs, dataset=dataset, split="legacy"))
    return out
