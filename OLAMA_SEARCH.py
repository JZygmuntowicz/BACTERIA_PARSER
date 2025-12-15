#!/usr/bin/env python3
"""
OLAMA_SEARCH_modified.py — original features plus automatic routing of JSON outputs
into taxonomy subfolders (species, genus, family, order) under ollama_output/.

Routing logic:
- For each species group we infer the originating source file by taking the snippet
  with highest similarity (if available) and reading its filename.
- We attempt to infer taxonomy level from that filename by looking for keywords
  'Species','Genus','Family','Order' (case-insensitive). If none matched -> 'unclear'.
- Output JSONs are written to ollama_output/<level>/<sanitized_species>.json

If multiple source files exist for a species, the one with the highest similarity wins.
If no source info available -> goes into ollama_output/unclear/

"""

import os
import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict

# ======================================================================
# CONFIG
# ======================================================================

MODEL = "mistral:7b-instruct"
OLLAMA_BIN = os.path.expanduser("~/ollama/bin/ollama")

INPUT_DIR = Path("taxonomy_index_hybrid")
PROMPT_FILE = Path("prompt_no_triggers.txt")
INTERMEDIATE_DIR = Path("ollama_intermediate")
LOG_DIR = Path("ollama_logs_no_triggers")
OLLAMA_OUTPUT = Path("ollama_output_no_triggers")  # new base output dir

# create directories
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OLLAMA_OUTPUT, exist_ok=True)

# create taxonomy subfolders
TAXONOMY_SUBFOLDERS = ["species", "genus", "family", "order", "unclear"]
for sub in TAXONOMY_SUBFOLDERS:
    os.makedirs(OLLAMA_OUTPUT / sub, exist_ok=True)

MAX_CHUNK_CHARS = 7000
MAX_FINAL_CONTEXT_CHARS = 10000

EXTRACTION_RETRIES = 2
FINAL_RETRIES = 2
SLEEP_BETWEEN_CALLS = 0.4

# ======================================================================
# PROMPTS
# ======================================================================

EXTRACTION_PROMPT_TEMPLATE = """
You are a strict extractor. Given the following excerpts, RETURN ONLY a JSON object
with a single key "evidence" whose value is a JSON array of verbatim sentences/fragments
indicating multicellularity or unicellularity. No paraphrase. Verbatim only.

If none exist, return: {{"evidence": []}}

Return JSON:
{{
  "evidence": ["...", "..."]
}}

EXCERPTS:
{excerpts}

IMPORTANT: Output MUST be valid JSON and NOTHING ELSE.
"""

# ======================================================================
# UTILS
# ======================================================================

def run_ollama_with_input(prompt, timeout=None):
    result = subprocess.run(
        [OLLAMA_BIN, "run", MODEL],
        input=prompt,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout
    )
    return result.stdout


def safe_json_load(s):
    try:
        return json.loads(s)
    except:
        return None


def sanitize_filename(s):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)

# ======================================================================
# SNIPPET HANDLING
# ======================================================================

def load_all_snippets(path):
    all_snips = []
    for p in sorted(path.glob("*_hybrid_matches.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except:
            continue
        for d in data:
            d.setdefault("name", "")
            d.setdefault("snippet", "")
            d.setdefault("similarity", 0.0)
            d["source_file"] = p.name
            all_snips.append(d)
    return all_snips


def group_by_species(snippets):
    g = defaultdict(list)
    for s in snippets:
        g[s["name"]].append(s)
    return g


def chunk_snippets(snips):
    snips = sorted(snips, key=lambda x: x.get("similarity", 0), reverse=True)
    chunks, cur, size = [], [], 0
    for s in snips:
        t = s["snippet"]
        if cur and (size + len(t) > MAX_CHUNK_CHARS):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(t)
    if cur:
        chunks.append(cur)
    return chunks


def build_excerpts_block(snips):
    parts = []
    for s in snips:
        hdr = f"[{s['name']}] (source: {s['source_file']}, sim={s['similarity']:.3f})"
        parts.append(f"{hdr}:\n{s['snippet'].strip()}")
    return "\n\n".join(parts)


def dedup(lst):
    out, seen = [], set()
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def trim_context(evidence):
    joined = " ||| ".join(evidence)
    if len(joined) <= MAX_FINAL_CONTEXT_CHARS:
        return joined
    out, size = [], 0
    for ev in evidence:
        need = len(ev) + 5
        if size + need > MAX_FINAL_CONTEXT_CHARS:
            break
        out.append(ev)
        size += need
    return " ||| ".join(out)

# ======================================================================
# NEW: infer source file and taxonomy level
# ======================================================================

def infer_source_file(snippets):
    """
    Choose the most representative source file for a species group.
    Strategy: pick the snippet with highest similarity and return its source_file.
    """
    if not snippets:
        return None
    best = max(snippets, key=lambda x: x.get("similarity", 0.0))
    return best.get("source_file")


def infer_taxonomy_level_from_source(source_filename):
    """
    Inspect the source filename and try to map it to one of: species, genus,
    family, order. Return lowercase level string or 'unclear' if none matched.
    """
    if not source_filename:
        return "unclear"
    s = source_filename.lower()
    if "species" in s:
        return "species"
    if "genus" in s:
        return "genus"
    if "family" in s:
        return "family"
    if "order" in s:
        return "order"
    return "unclear"


def output_path_for_species(species, snippets):
    src = infer_source_file(snippets)
    level = infer_taxonomy_level_from_source(src)
    folder = OLLAMA_OUTPUT / level
    os.makedirs(folder, exist_ok=True)
    filename = sanitize_filename(species) + ".json"
    return folder / filename, level, src

# ======================================================================
# EXTRACTION PASS
# ======================================================================

def extract_evidence_for_species(species, snippets):
    extracted = []
    chunks = chunk_snippets(snippets)
    for i, ch in enumerate(chunks, start=1):
        block = build_excerpts_block(ch)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(excerpts=block)

        for attempt in range(EXTRACTION_RETRIES + 1):
            try:
                time.sleep(SLEEP_BETWEEN_CALLS)
                out = run_ollama_with_input(prompt)
            except subprocess.CalledProcessError as e:
                (LOG_DIR / f"extract_err_{sanitize_filename(species)}_{i}.log"
                 ).write_text(e.stderr, encoding="utf-8")
                continue

            parsed = safe_json_load(out)
            if isinstance(parsed, dict) and "evidence" in parsed:
                for ev in parsed["evidence"]:
                    if isinstance(ev, str) and ev.strip():
                        extracted.append(ev.strip())
                break

            (INTERMEDIATE_DIR / f"extract_raw_{sanitize_filename(species)}_{i}.txt"
             ).write_text(out, encoding="utf-8")

    return dedup(extracted)

# ======================================================================
# FINAL CURATOR
# ======================================================================

def run_final_curator(species, merged_context, sources):
    if not PROMPT_FILE.exists():
        raise FileNotFoundError("missing prompt_no_triggers.txt")

    template = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = template.format(
        context=merged_context,
        bacteria_name=species,
        sources=sources
    )
    prompt += (
        "\n\nIMPORTANT REMINDER:\n"
        "Return exactly one JSON object with keys: taxonomy_level, species, multicellular, "
        "source, extracted_text, model_explanation.\nJSON ONLY.\n"
    )

    for attempt in range(FINAL_RETRIES + 1):
        try:
            time.sleep(SLEEP_BETWEEN_CALLS)
            out = run_ollama_with_input(prompt)
        except subprocess.CalledProcessError:
            continue

        parsed = safe_json_load(out)
        required = {"taxonomy_level", "species", "multicellular",
                    "source", "extracted_text", "model_explanation"}

        if isinstance(parsed, dict) and required.issubset(parsed.keys()):
            return parsed

        (INTERMEDIATE_DIR / f"final_raw_{sanitize_filename(species)}_{attempt}.txt"
         ).write_text(out, encoding="utf-8")

    return None

# ======================================================================
# MAIN
# ======================================================================

def main():

    if not Path(OLLAMA_BIN).exists():
        raise FileNotFoundError(f"Ollama binary missing: {OLLAMA_BIN}")

    all_snippets = load_all_snippets(INPUT_DIR)
    species_groups = group_by_species(all_snippets)

    print(f"Loaded {len(species_groups)} species.")

    for idx, (species, snippets) in enumerate(species_groups.items(), start=1):
        print(f"[{idx}/{len(species_groups)}] {species}")

        evidence = extract_evidence_for_species(species, snippets)
        print(f"  extracted: {len(evidence)}")

        out_path, inferred_level, inferred_src = output_path_for_species(species, snippets)

        if not evidence:
            result = {
                "taxonomy_level": inferred_level if inferred_level != "unclear" else "none",
                "species": species,
                "multicellular": "UNCLEAR",
                "source": inferred_src or ", ".join(sorted(set(s["source_file"] for s in snippets))),
                "extracted_text": "",
                "model_explanation": "no high-weight evidence"
            }
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  saved UNCLEAR (no evidence) -> {out_path}")
            continue

        merged = trim_context(evidence)
        sources = ", ".join(sorted(set(s["source_file"] for s in snippets)))

        final = run_final_curator(species, merged, sources)

        if final is None:
            final = {
                "taxonomy_level": inferred_level if inferred_level != "unclear" else "none",
                "species": species,
                "multicellular": "UNCLEAR",
                "source": sources,
                "extracted_text": " ||| ".join(evidence[:5]),
                "model_explanation": "final curator failed"
            }
        else:
            # overwrite taxonomy level if the curator did NOT supply one; otherwise keep what model said
            if not final.get("taxonomy_level"):
                final["taxonomy_level"] = inferred_level if inferred_level != "unclear" else "none"

        out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  saved OK -> {out_path}")

    print("\nDone. Output saved in:", OLLAMA_OUTPUT)


if __name__ == "__main__":
    main()
