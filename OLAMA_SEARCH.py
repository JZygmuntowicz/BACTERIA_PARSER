#!/usr/bin/env python3
"""
OLAMA_SEARCH.py — multi-pass extraction + strict trigger validation + malformed word normalization.

Features:
- Loads snippets from taxonomy_index_hybrid/*_hybrid_matches.json
- Performs multi-pass evidence extraction on chunks
- Detects morphology triggers (filaments, hyphae, heterocysts, etc.)
- NEW: fixes malformed / merged / broken words automatically
- Validates triggers with mini-prompt (Option A: only "True" accepted)
- Final curator step with prompt.txt
- Strict JSON-only output
- One JSON per species saved to ollama_results/
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
PROMPT_FILE = Path("prompt.txt")
OUTPUT_DIR = Path("ollama_results")
INTERMEDIATE_DIR = Path("ollama_intermediate")
LOG_DIR = Path("ollama_logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MAX_CHUNK_CHARS = 7000
MAX_FINAL_CONTEXT_CHARS = 10000

EXTRACTION_RETRIES = 2
VALIDATION_RETRIES = 2
FINAL_RETRIES = 2
SLEEP_BETWEEN_CALLS = 0.4

# ======================================================================
# TRIGGER LIST
# ======================================================================

MORPHOLOGY_TRIGGERS = [
    "filament", "filaments",
    "trichome", "trichomes",
    "heterocyst", "heterocysts",
    "akinete", "akinetes",
    "hormogonia",
    "mycelium", "mycelia", "aerial mycelium", "substrate mycelium",
    "hypha", "hyphae", "aerial hyphae",
    "fruiting body", "fruiting bodies",
    "spore", "spores", "myxospore", "myxospores",
    "sheath", "sheaths",
    "multicellular aggregate", "multicellular aggregates",
    "magnetosome", "magnetosomes",
    "biofilm matrix",
    "quorum-sensing", "quorum sensing", "subpopulations",
    "synchronous cell-division",
    "hyphal branches", "branches",
    "sporangia",
    "microcolony", "microcolonies"
]

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

TRIGGER_VALIDATION_PROMPT = """
You are a conservative microbiology curator. Determine whether the sentence describes a
morphological feature of the specified bacterium.

Return ONLY:
{{"relevant": "True"}}  OR  {{"relevant": "False"}}  OR  {{"relevant": "UNCLEAR"}}

Rules:
- "True" only if the sentence clearly attributes the structure to the named bacterium.
- "False" if structure refers to another organism or is generic.
- "UNCLEAR" if ambiguous.
- Do NOT guess or add missing context.

[Bacterium]: {name}
[Sentence]: {sentence}
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
    try: return json.loads(s)
    except: return None

def sanitize_filename(s):
    return "".join(c if c.isalnum() or c in ("_","-") else "_" for c in s)

# ======================================================================
# MALFORMED WORD NORMALIZATION
# ======================================================================

def normalize_malformed_words(text):
    """
    Fix broken words, missing spaces, hyphen splits, and run-together terms.
    """
    t = text.replace("  ", " ")
    t = t.replace("-\n", "")
    t = t.replace("\n", " ")

    # Remove internal spacing for triggers e.g. "hetero cysts" -> "heterocysts"
    no_space = t.replace(" ", "")
    for trig in MORPHOLOGY_TRIGGERS:
        trig_clean = trig.replace(" ", "")
        if trig_clean in no_space:
            # Replace glued version
            t = t.replace(trig_clean, trig)
            # Replace space-separated glitched version
            parts = list(trig)
            spaced = " ".join(parts)
            if spaced in t:
                t = t.replace(spaced, trig)

    return t

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
            d.setdefault("name","")
            d.setdefault("snippet","")
            d.setdefault("similarity",0.0)
            d["source_file"] = p.name
            all_snips.append(d)
    return all_snips

def group_by_species(snippets):
    g = defaultdict(list)
    for s in snippets:
        g[s["name"]].append(s)
    return g

def chunk_snippets(snips):
    snips = sorted(snips, key=lambda x: x.get("similarity",0), reverse=True)
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
    parts=[]
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
        need = len(ev)+5
        if size+need > MAX_FINAL_CONTEXT_CHARS: break
        out.append(ev)
        size += need
    return " ||| ".join(out)

# ======================================================================
# EXTRACTION PASS
# ======================================================================

def extract_evidence_for_species(species, snippets):
    extracted=[]
    chunks = chunk_snippets(snippets)
    for i, ch in enumerate(chunks, start=1):
        block = build_excerpts_block(ch)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(excerpts=block)

        for attempt in range(EXTRACTION_RETRIES+1):
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
                    if isinstance(ev,str) and ev.strip():
                        extracted.append(ev.strip())
                break

            (INTERMEDIATE_DIR / f"extract_raw_{sanitize_filename(species)}_{i}.txt"
            ).write_text(out, encoding="utf-8")

    return dedup(extracted)

# ======================================================================
# TRIGGER VALIDATION
# ======================================================================

def validate_trigger_context(species, sentence):
    prompt = TRIGGER_VALIDATION_PROMPT.format(name=species, sentence=sentence)
    for attempt in range(VALIDATION_RETRIES+1):
        try:
            time.sleep(SLEEP_BETWEEN_CALLS)
            out = run_ollama_with_input(prompt)
        except subprocess.CalledProcessError:
            continue

        parsed = safe_json_load(out)
        if isinstance(parsed, dict) and "relevant" in parsed:
            if parsed["relevant"] in ("True","False","UNCLEAR"):
                return parsed["relevant"]

    return "UNCLEAR"

def detect_validated_triggers(species, evidence_sentences):
    found=set()
    for ev in evidence_sentences:
        cleaned = normalize_malformed_words(ev).lower()
        for trig in MORPHOLOGY_TRIGGERS:
            if trig in cleaned:
                verdict = validate_trigger_context(species, ev)
                if verdict == "True":
                    found.add(trig)
    return sorted(found)

# ======================================================================
# FINAL CURATOR
# ======================================================================

def run_final_curator(species, merged_context, sources):
    if not PROMPT_FILE.exists():
        raise FileNotFoundError("missing prompt.txt")

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

    for attempt in range(FINAL_RETRIES+1):
        try:
            time.sleep(SLEEP_BETWEEN_CALLS)
            out = run_ollama_with_input(prompt)
        except subprocess.CalledProcessError:
            continue

        parsed = safe_json_load(out)
        required = {"taxonomy_level","species","multicellular","source","extracted_text","model_explanation"}

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

        if not evidence:
            result = {
                "taxonomy_level": "none",
                "species": species,
                "multicellular": "UNCLEAR",
                "source": ", ".join(sorted(set(s["source_file"] for s in snippets))),
                "extracted_text": "",
                "model_explanation": "no high-weight evidence",
                "morphology_trigger": []
            }
            (OUTPUT_DIR / f"{sanitize_filename(species)}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print("  saved UNCLEAR (no evidence)")
            continue

        triggers = detect_validated_triggers(species, evidence)
        print(f"  validated triggers: {triggers}")

        merged = trim_context(evidence)
        sources = ", ".join(sorted(set(s["source_file"] for s in snippets)))

        final = run_final_curator(species, merged, sources)

        if final is None:
            final = {
                "taxonomy_level": "none",
                "species": species,
                "multicellular": "UNCLEAR",
                "source": sources,
                "extracted_text": " ||| ".join(evidence[:5]),
                "model_explanation": "final curator failed",
                "morphology_trigger": triggers
            }
        else:
            if triggers:
                final["multicellular"] = "True"
            final["morphology_trigger"] = triggers

        (OUTPUT_DIR / f"{sanitize_filename(species)}.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

        print("  saved OK")

    print("\nDone. Output saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
