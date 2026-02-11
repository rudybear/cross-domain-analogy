"""
Deep Dive: Music Theory ↔ Internal Combustion Engine
=====================================================
Expanded concept sets, clean domains, targeted LLM analysis.
Explores the oscillatory/cyclical structural isomorphism.
"""

import json
import sys
import numpy as np
from scipy.spatial.distance import cosine
from itertools import product

# ---------------------------------------------------------------------------
# Clean, expanded domain definitions
# ---------------------------------------------------------------------------

MUSIC_CONCEPTS = [
    # Oscillation / wave fundamentals
    "frequency", "amplitude", "wavelength", "oscillation", "vibration",
    "waveform", "phase", "period", "cycle",
    # Core music theory
    "rhythm", "tempo", "pitch", "melody", "harmony", "chord",
    "resonance", "timbre", "octave", "scale", "beat",
    # Dynamics & energy
    "crescendo", "diminuendo", "fortissimo", "pianissimo",
    # Structure
    "measure", "bar", "downbeat", "syncopation", "polyrhythm",
    # Interference patterns
    "overtone", "harmonic", "fundamental", "dissonance", "consonance",
    # Damping / decay
    "sustain", "decay", "attack", "release", "damping",
    # Energy transfer
    "acoustic_energy", "sound_pressure", "propagation",
]

ENGINE_CONCEPTS = [
    # Oscillation / cycle fundamentals
    "RPM", "stroke", "cycle", "crankshaft", "reciprocation",
    "rotation", "frequency", "timing",
    # Core engine components
    "piston", "cylinder", "valve", "camshaft", "spark_plug",
    "connecting_rod", "flywheel",
    # Thermodynamic cycle
    "intake", "compression", "combustion", "exhaust",
    "expansion", "ignition", "detonation",
    # Energy & dynamics
    "torque", "horsepower", "power_stroke", "efficiency",
    "thermal_energy", "pressure", "temperature",
    # Fuel system
    "fuel", "air_fuel_mixture", "injection", "carburetor",
    # Damping / waste
    "friction", "heat_loss", "vibration_damping", "exhaust_gas",
    "emissions", "muffler",
    # Control
    "throttle", "governor", "idle", "redline",
]

# Relations to probe — designed to capture oscillatory patterns
RELATIONS = [
    "produces", "requires", "enables", "causes",
    "transforms_into", "oscillates_with", "amplifies",
    "dampens", "regulates", "drives", "follows",
    "peaks_at", "dissipates", "synchronizes_with",
    "contains", "opposes", "sustains",
]


def main():
    from sentence_transformers import SentenceTransformer
    import requests

    print("=" * 70)
    print("  DEEP DIVE: Music Theory ↔ Internal Combustion Engine")
    print("  Exploring oscillatory/cyclical structural isomorphism")
    print("=" * 70)

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cache = {}

    def embed(text):
        if text not in cache:
            cache[text] = model.encode(text.replace("_", " "), normalize_embeddings=True)
        return cache[text]

    def embed_ctx(concept, domain):
        key = f"_ctx_{domain}_{concept}"
        if key not in cache:
            cache[key] = model.encode(
                f"{concept.replace('_', ' ')} in the context of {domain}",
                normalize_embeddings=True
            )
        return cache[key]

    # -----------------------------------------------------------------------
    # Step 1: Embed all concepts in both domains
    # -----------------------------------------------------------------------
    print(f"\nEmbedding {len(MUSIC_CONCEPTS)} music concepts...")
    music_vecs = {c: embed_ctx(c, "music theory and acoustics") for c in MUSIC_CONCEPTS}

    print(f"Embedding {len(ENGINE_CONCEPTS)} engine concepts...")
    engine_vecs = {c: embed_ctx(c, "internal combustion engines") for c in ENGINE_CONCEPTS}

    # -----------------------------------------------------------------------
    # Step 2: Find concept-level mappings (which music concept ↔ which engine concept)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONCEPT MAPPING: Music ↔ Engine")
    print("=" * 70)

    mappings = []
    for mc in MUSIC_CONCEPTS:
        for ec in ENGINE_CONCEPTS:
            sim = 1 - cosine(music_vecs[mc], engine_vecs[ec])
            mappings.append((sim, mc, ec))

    mappings.sort(reverse=True)

    print(f"\n  {'Music Concept':<25} {'Engine Concept':<25} {'Similarity':>10}")
    print(f"  {'-'*25} {'-'*25} {'-'*10}")

    # Show top mappings, but also ensure diversity (don't repeat concepts)
    shown_music = set()
    shown_engine = set()
    count = 0
    for sim, mc, ec in mappings:
        if mc in shown_music or ec in shown_engine:
            continue
        print(f"  {mc.replace('_', ' '):<25} {ec.replace('_', ' '):<25} {sim:10.3f}")
        shown_music.add(mc)
        shown_engine.add(ec)
        count += 1
        if count >= 25:
            break

    # -----------------------------------------------------------------------
    # Step 3: Relational structure comparison
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  RELATIONAL STRUCTURE COMPARISON")
    print("=" * 70)

    def score_relation(subj, pred, obj, domain):
        s = embed_ctx(subj, domain)
        p = embed(pred)
        o = embed_ctx(obj, domain)
        predicted = s + p
        norm = np.linalg.norm(predicted)
        if norm > 0:
            predicted = predicted / norm
        return float(np.dot(predicted, o))

    # Extract top relations in each domain
    print("\n  Top Music Theory relations:")
    music_rels = []
    for pred in RELATIONS:
        for subj in MUSIC_CONCEPTS:
            for obj in MUSIC_CONCEPTS:
                if subj == obj:
                    continue
                score = score_relation(subj, pred, obj, "music theory")
                if score > 0.35:
                    music_rels.append((score, subj, pred, obj))

    music_rels.sort(reverse=True)
    seen = set()
    for score, s, p, o in music_rels[:20]:
        pair = (s, o)
        if pair in seen:
            continue
        seen.add(pair)
        print(f"    ({s} --{p}--> {o})  score={score:.3f}")

    print("\n  Top Engine relations:")
    engine_rels = []
    for pred in RELATIONS:
        for subj in ENGINE_CONCEPTS:
            for obj in ENGINE_CONCEPTS:
                if subj == obj:
                    continue
                score = score_relation(subj, pred, obj, "internal combustion engines")
                if score > 0.35:
                    engine_rels.append((score, subj, pred, obj))

    engine_rels.sort(reverse=True)
    seen = set()
    for score, s, p, o in engine_rels[:20]:
        pair = (s, o)
        if pair in seen:
            continue
        seen.add(pair)
        print(f"    ({s} --{p}--> {o})  score={score:.3f}")

    # -----------------------------------------------------------------------
    # Step 4: Cross-domain relation alignment
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  CROSS-DOMAIN RELATION ALIGNMENTS")
    print("  (same relational pattern in both domains)")
    print("=" * 70)

    # Compare relation vectors across domains
    alignments = []
    for m_score, ms, mp, mo in music_rels[:50]:
        m_rel_vec = embed_ctx(mo, "music") - embed_ctx(ms, "music")
        for e_score, es, ep, eo in engine_rels[:50]:
            e_rel_vec = embed_ctx(eo, "engines") - embed_ctx(es, "engines")
            sim = 1 - cosine(m_rel_vec, e_rel_vec)
            alignments.append((sim, ms, mp, mo, es, ep, eo, m_score, e_score))

    alignments.sort(reverse=True)

    print(f"\n  {'Sim':>5}  {'Music Relation':<45} {'Engine Relation':<45}")
    print(f"  {'-'*5}  {'-'*45} {'-'*45}")

    shown = set()
    count = 0
    for sim, ms, mp, mo, es, ep, eo, mscore, escore in alignments:
        key = (ms, mo, es, eo)
        if key in shown:
            continue
        shown.add(key)
        m_str = f"({ms} --{mp}--> {mo})"
        e_str = f"({es} --{ep}--> {eo})"
        print(f"  {sim:5.3f}  {m_str:<45} {e_str:<45}")
        count += 1
        if count >= 25:
            break

    # -----------------------------------------------------------------------
    # Step 5: LLM Deep Analysis via Ollama
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  LLM DEEP ANALYSIS: Why do these domains share structure?")
    print("=" * 70)

    url = "http://localhost:11434/api/generate"

    # Gather the best alignments for the prompt
    top_alignments_text = ""
    shown2 = set()
    ct = 0
    for sim, ms, mp, mo, es, ep, eo, _, _ in alignments:
        key = (ms, mo, es, eo)
        if key in shown2:
            continue
        shown2.add(key)
        top_alignments_text += f"  Music: ({ms} --{mp}--> {mo})  ↔  Engine: ({es} --{ep}--> {eo})  similarity={sim:.3f}\n"
        ct += 1
        if ct >= 15:
            break

    # Gather concept mappings for the prompt
    top_mappings_text = ""
    shown_m = set()
    shown_e = set()
    ct2 = 0
    for sim, mc, ec in mappings:
        if mc in shown_m or ec in shown_e:
            continue
        shown_m.add(mc)
        shown_e.add(ec)
        top_mappings_text += f"  {mc} ↔ {ec} (similarity={sim:.3f})\n"
        ct2 += 1
        if ct2 >= 15:
            break

    prompts = [
        {
            "title": "FUNDAMENTAL ISOMORPHISM",
            "prompt": f"""You are a cross-disciplinary scientist analyzing a deep structural analogy
between Music Theory and Internal Combustion Engines.

My embedding analysis found these concept mappings (music concept ↔ engine concept):
{top_mappings_text}

And these structural relation alignments:
{top_alignments_text}

Questions:
1. What is the FUNDAMENTAL reason these two domains share structure? What deep principle connects them?
2. List 5 specific structural isomorphisms (A in music = B in engine) with explanations.
3. What does this analogy PREDICT? What property of one domain should exist in the other but hasn't been explored?
4. Are there known physics/engineering connections that explain this (e.g., both involve wave mechanics)?

Be specific and analytical. Max 400 words."""
        },
        {
            "title": "OSCILLATORY PATTERNS",
            "prompt": f"""Analyze the oscillatory/cyclical patterns shared between Music Theory and Internal Combustion Engines:

Music: rhythm, tempo, beat, measure, cycle, frequency, period, phase, syncopation
Engine: RPM, stroke, cycle, crankshaft, timing, reciprocation, rotation, frequency

Both domains feature:
- Periodic repetition (beats vs. power strokes)
- Frequency as a key parameter (pitch vs. RPM)
- Phase relationships (polyrhythm vs. cylinder firing order)
- Energy envelopes (attack-decay-sustain-release vs. intake-compression-power-exhaust)
- Harmonic content (overtones vs. engine harmonics)

Questions:
1. Map the 4-stroke engine cycle (intake, compression, power, exhaust) to the ADSR envelope (attack, decay, sustain, release). Does this mapping reveal anything non-obvious?
2. What is the engine equivalent of "harmony" (multiple frequencies combining)?
3. What is the musical equivalent of "engine knock" (detonation at wrong timing)?
4. Could optimization techniques from one domain transfer to the other?

Be specific with technical details. Max 400 words."""
        },
        {
            "title": "ENERGY & DISSIPATION ANALOGY",
            "prompt": f"""Analyze the energy flow and dissipation patterns shared between Music and Engines:

Music energy flow: acoustic_energy → sound_pressure → propagation → damping → decay
Engine energy flow: thermal_energy → pressure → expansion → friction → heat_loss

Both have:
- Energy input (plucking a string vs. fuel ignition)
- Resonant amplification (body resonance vs. turbocharger)
- Useful output (sound vs. mechanical work)
- Waste/losses (damping vs. friction/heat)
- Efficiency as key metric

Questions:
1. What is the musical equivalent of "thermal efficiency"?
2. What is the engine equivalent of "resonance"?
3. Both domains have "overtones" - unwanted secondary frequencies. How do they compare?
4. Could the concept of "impedance matching" (acoustic) improve engine design, or vice versa?
5. What non-obvious dependency does this reveal?

Be specific. Max 300 words."""
        },
    ]

    for p in prompts:
        print(f"\n  --- {p['title']} ---")
        try:
            resp = requests.post(url, json={
                "model": "qwen2.5:14b",
                "prompt": p["prompt"],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 600}
            }, timeout=300)
            if resp.status_code == 200:
                analysis = resp.json().get("response", "")
                for line in analysis.split("\n"):
                    print(f"    {line}")
            else:
                print(f"    (Ollama error: {resp.status_code})")
        except Exception as e:
            print(f"    (Error: {e})")

    # -----------------------------------------------------------------------
    # Step 6: Summary mapping table
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  STRUCTURAL ISOMORPHISM MAP")
    print("=" * 70)
    print("""
  Music Theory                    Internal Combustion Engine
  ══════════════════════════════  ══════════════════════════════════
  frequency / pitch         ↔    RPM / engine speed
  rhythm / beat             ↔    power stroke cycle
  tempo                     ↔    RPM (speed of cycle)
  measure / bar             ↔    complete 4-stroke cycle
  amplitude / volume        ↔    torque / power output
  resonance                 ↔    harmonic vibration / turbo resonance
  overtones / harmonics     ↔    engine order harmonics
  dissonance                ↔    engine knock / detonation
  consonance                ↔    smooth running / balanced engine
  damping                   ↔    vibration damping / engine mounts
  sustain                   ↔    flywheel momentum
  attack                    ↔    ignition / power stroke onset
  decay                     ↔    expansion stroke / pressure drop
  release                   ↔    exhaust stroke
  syncopation               ↔    uneven firing order
  polyrhythm                ↔    multi-cylinder phase offset
  timbre                    ↔    exhaust note / engine character
  crescendo                 ↔    acceleration / increasing RPM
  tuning                    ↔    engine tuning / calibration
  acoustic_energy           ↔    thermal_energy
  sound_pressure            ↔    cylinder_pressure
  fundamental frequency     ↔    base firing frequency
""")

    print("\nDone.")


if __name__ == "__main__":
    main()
