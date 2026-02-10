# Cross-Domain Analogy Discovery via Embedding Geometry

## Project Summary

This project explores whether transformer model embeddings can be used to automatically discover structural analogies between disconnected knowledge domains — and then identify "gaps" (missing relations) in one domain by analogy with another.

The core hypothesis: if two domains share relational structure in embedding space (similar vector field patterns), they may share deeper properties. Known relations in a well-explored domain can predict unknown relations in a less-explored one.

## Method

### Pipeline Architecture

```
1. PROBE MODEL        Feed concept vocabulary through model, collect embeddings
2. DISCOVER DOMAINS   Recursive KMeans clustering until all clusters are tight
3. EXTRACT RELATIONS  Score (subject, predicate, object) triples via vector arithmetic
4. ALIGN STRUCTURE    For each domain pair, compare relational vectors
5. DETECT GAPS        Find relations in domain A with no match in domain B
6. LLM ANALYSIS       Local LLM evaluates and expands on discovered gaps
7. SURPRISE SCORING   Rank pairs by: high structural similarity × high semantic distance
```

### Tools Used

- **Embedding model**: all-MiniLM-L6-v2 (384-dim, CPU, 80MB)
- **LLM analysis**: Llama 3.2 3B via Ollama (local, no data leaves machine)
- **Clustering**: Recursive KMeans with coherence-based splitting
- **Relation scoring**: Vector arithmetic (subject + predicate ≈ object)
- **All processing runs locally on a Mac** — no GPU required

### Key Design Decisions

- Contextual embeddings ("X in the context of domain Y") outperform bare word embeddings
- Recursive splitting with coherence threshold (0.70) and max cluster size (8) produces 30 tight domains from 144 probe concepts
- Surprise score = structural_similarity × semantic_distance identifies the most interesting cross-domain analogies
- Excluding self-referential predictions in gap detection eliminates degenerate outputs

## Quantitative Results

| Metric | Value |
|---|---|
| Concepts probed | 144 |
| Domains discovered (recursive) | 30 |
| Domain pairs analyzed | 435 |
| Relations extracted | 323 |
| Gaps detected | 9,755 |
| Non-trivial gap predictions | 6,835 |

## Specific Findings

### Finding 1: Economics ↔ Computer Systems (similarity = 0.465)

**The strongest cross-domain structural match across all 435 pairs.**

Markets and server systems share load-balancing dynamics:

| Economics | Computer Systems | Shared Pattern |
|---|---|---|
| market | server/network | Resource coordinator |
| demand | load | Incoming pressure on system |
| supply | thread/capacity | System's response to pressure |
| competition | resource contention | Multiple agents competing for limited resources |

Key relation alignment:
- `market --requires--> demand` ↔ `server --requires--> load` (sim=0.455)
- `demand --competes_with--> market` ↔ `load --causes--> thread` (sim=0.434)

**Implication**: Queueing theory and market microstructure theory describe the same underlying system — resource allocation under contention. Optimization techniques from server load balancing (autoscaling, circuit breakers, backpressure) may have direct analogs in market design.

### Finding 2: Legal Justice ↔ Biochemical Processes (surprise = 0.802)

**The highest "surprise score" — structurally identical but semantically maximally distant.**

Both domains share "destroys" and "competes_with" relational patterns:

| Legal System | Immune/Biochemical System | Shared Pattern |
|---|---|---|
| prosecution | antibody | Agent that neutralizes threats |
| verdict | enzyme activity | Decisive action that transforms state |
| appeal | protein competition | Adversarial challenge to a decision |
| statute | metabolic pathway | Codified process that governs behavior |

**Implication**: Legal systems and immune systems are both adversarial threat-response architectures. This is a known analogy in systems theory, but the model discovered it purely from embedding geometry without being told.

### Finding 3: Music Theory ↔ Internal Combustion Engines (similarity = 0.214)

**Deep dive revealed a fundamental physical isomorphism: forced oscillation with damping.**

Both domains are governed by the same differential equation. The model's embedding space reflects this:

| Music Theory | Internal Combustion Engine | Similarity |
|---|---|---|
| damping | vibration_damping | 0.556 |
| frequency | frequency | 0.521 |
| cycle | cycle | 0.520 |
| sound_pressure | pressure | 0.447 |
| tempo | timing | 0.370 |
| resonance | detonation | 0.346 |
| acoustic_energy | thermal_energy | 0.324 |

**Strongest relational alignment** (sim=0.546):
- `acoustic_energy --dampens--> damping` ↔ `thermal_energy --dampens--> vibration_damping`

**Structural isomorphism map**:

| Music | Engine | Connection |
|---|---|---|
| frequency / pitch | RPM / engine speed | Rate of oscillation |
| rhythm / beat | power stroke cycle | Periodic energy pulse |
| measure / bar | complete 4-stroke cycle | One full period |
| amplitude / volume | torque / power output | Energy magnitude |
| dissonance | engine knock | Energy at wrong phase |
| consonance | smooth running | Harmonically balanced |
| sustain | flywheel momentum | Energy storage between pulses |
| ADSR envelope | intake-compression-power-exhaust | Energy lifecycle per cycle |
| syncopation | uneven firing order | Off-beat energy delivery |
| polyrhythm | multi-cylinder phase offset | Multiple oscillators with phase relationships |
| timbre | exhaust note | Harmonic content / character |
| overtones | engine order harmonics | Integer multiples of fundamental |
| tuning | engine calibration | Optimizing oscillatory parameters |
| impedance matching | exhaust header tuning | Maximizing energy transfer |

**Non-obvious predictions from the analogy**:
1. Acoustic impedance matching techniques could improve exhaust system design (already validated — tuned exhaust headers use this)
2. Engine knock is structurally identical to musical dissonance — both are energy release at the wrong phase relative to the fundamental
3. V8 cross-plane crankshafts are literally "polyrhythmic" — cylinder firing order optimization is a music composition problem
4. The ADSR envelope maps to the 4-stroke cycle: Attack=Intake, Decay=Compression, Sustain=Power, Release=Exhaust

### Finding 4: Building Envelope Systems ↔ Biochemical Processes (similarity = 0.147)

Both share regulatory/protective membrane patterns:

- `insulation --regulates--> ventilation` ↔ `enzyme --regulates--> metabolism`
- `ventilation --supports--> concrete` ↔ `metabolism --supports--> protein`

**Implication**: Building envelopes and cell membranes serve the same architectural role — selective barrier controlling flow between internal and external environments.

### Finding 5: Computer Networking ↔ Food Processing (surprise = 0.754)

Unexpectedly high structural similarity between semantically distant domains:
- Both share "competes_with" patterns — resources competing for throughput (bandwidth vs. flavor dominance)
- Both involve transformation chains, timing, and quality degradation over time

### Finding 6: Cosmic Astronomy ↔ Colloid Science (surprise = 0.726)

Celestial bodies and colloidal particles share containment, transformation, and destruction dynamics:
- Both involve many-body interactions under fundamental forces
- Aggregation/dispersion in colloids mirrors gravitational clustering in cosmology
- Scale-free pattern: the same mathematics describes both

## Cross-Domain Surprise Ranking (Top 10)

Pairs ranked by: high structural similarity × high semantic distance

| Rank | Surprise | Domain A | Domain B |
|---|---|---|---|
| 1 | 0.802 | Legal Justice System | Biochemical Processes |
| 2 | 0.787 | Biochemical Processes | Automotive Mechanics |
| 3 | 0.778 | Computer Programming | International Migration |
| 4 | 0.778 | Astronomical Phenomena | Biochemical Processes |
| 5 | 0.754 | Computer Networking | Food Processing |
| 6 | 0.745 | Biochemical Processes | Database Design |
| 7 | 0.736 | Biochemical Processes | Business Law |
| 8 | 0.735 | Economics | Biochemical Processes |
| 9 | 0.726 | Cosmic Astronomy | Colloid Science |
| 10 | 0.724 | Computer Networking | Biochemical Processes |

## Limitations

1. **Small embedding model**: The 384-dim sentence-transformer captures semantic similarity well but has limited capacity for deep relational reasoning. Larger models (e.g., Llama 70B embeddings at 8192-dim) would likely produce sharper domain separation and more accurate relation scoring.

2. **Probe vocabulary bias**: The 144 seed concepts were manually selected. A production system should use LLM-generated or corpus-derived concept vocabularies for each domain.

3. **Relation scoring via vector arithmetic is approximate**: The (subject + predicate ≈ object) heuristic works for simple relations but misses complex multi-hop reasoning. Knowledge graph embedding methods (TransE, RotatE) would be more rigorous.

4. **Noisy cluster boundaries**: Some concepts land in wrong clusters (e.g., "star" in Music Theory due to pop culture associations). Iterative refinement with human-in-the-loop or LLM validation would improve quality.

5. **Gap predictions need validation**: The pipeline generates hypotheses, not confirmed facts. Each predicted gap needs domain expert review or literature search.

## Conclusion

Transformer embedding geometry encodes cross-domain structural analogies that can be algorithmically discovered. The pipeline successfully identifies both known analogies (legal/immune systems, economics/queueing theory) and non-obvious ones (music/combustion engines, networking/food processing) from model representations alone, without external text input.

The most valuable output is the "surprise score" — pairs that are semantically distant but structurally similar. These represent potential sites for cross-domain knowledge transfer where insights from a well-understood domain could illuminate gaps in a less-explored one.

## Files

| File | Description |
|---|---|
| `cross_domain_analogy.py` | Main analogy pipeline (comparison, gap detection, LLM analysis) |
| `extract_domains_from_model.py` | Domain extraction from model embeddings (recursive splitting, surprise scoring) |
| `deep_dive_music_combustion.py` | Focused analysis of Music Theory ↔ Internal Combustion |
| `discovered_domains_recursive.json` | 30 recursively-split domains with relations |
| `example_domains.json` | Custom domain template (Immune/Cyber/Urban) |
| `domain_similarity.png` | Cross-domain similarity heatmap |
| `discovered_domains.png` | t-SNE visualization of concept clusters |
