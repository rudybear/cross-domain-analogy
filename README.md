# Cross-Domain Analogy Discovery via Embedding Geometry

Automatically discover structural analogies between disconnected knowledge domains using transformer embedding geometry — then identify "gaps" (missing relations) in one domain by analogy with another.

**Runs entirely locally on a Mac. No GPU required. No data leaves your machine.**

## The Idea

If two domains share similar relational patterns in vector space, they may share deeper properties. Known relations in a well-explored domain can predict unknown relations in a less-explored one.

The pipeline found that **V8 engine firing order optimization is literally a polyrhythm composition problem**, that **markets behave like servers under load**, and that **the legal system is structurally identical to the immune system** — all purely from embedding geometry, without being told.

## Pipeline

```
1. PROBE MODEL        Feed concept vocabulary through model, collect embeddings
2. DISCOVER DOMAINS   Recursive KMeans clustering until all clusters are tight
3. EXTRACT RELATIONS  Score (subject, predicate, object) triples via vector arithmetic
4. ALIGN STRUCTURE    For each domain pair, compare relational vectors
5. DETECT GAPS        Find relations in domain A with no match in domain B
6. LLM ANALYSIS       Local LLM evaluates and expands on discovered gaps
7. SURPRISE SCORING   Rank pairs by: high structural similarity × high semantic distance
```

## Quickstart

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/cross-domain-analogy.git
cd cross-domain-analogy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Step 1: Extract domains from model representations
python extract_domains_from_model.py --recursive --max-size 8 --min-coherence 0.70

# Step 2: Run cross-domain analogy pipeline
python cross_domain_analogy.py --custom discovered_domains_recursive.json --ollama

# Step 3: Deep dive into a specific pair
python deep_dive_music_combustion.py
```

**Optional**: Install [Ollama](https://ollama.ai) for local LLM analysis:
```bash
brew install ollama
ollama pull llama3.2:3b
```

## What It Found

### Music Theory is Isomorphic to Internal Combustion Engines

Both are governed by forced oscillation with damping — the same differential equation:

| Music | Engine | Connection |
|---|---|---|
| frequency / pitch | RPM / engine speed | Rate of oscillation |
| rhythm / beat | power stroke cycle | Periodic energy pulse |
| amplitude / volume | torque / power output | Energy magnitude |
| dissonance | engine knock | Energy at wrong phase |
| sustain | flywheel momentum | Energy storage between pulses |
| ADSR envelope | intake-compression-power-exhaust | Energy lifecycle per cycle |
| syncopation | uneven firing order | Off-beat energy delivery |
| polyrhythm | multi-cylinder phase offset | Multiple oscillators with phase relationships |
| timbre | exhaust note | Harmonic content / character |
| impedance matching | exhaust header tuning | Maximizing energy transfer |

### Markets Are Servers (similarity = 0.465)

The strongest cross-domain structural match across all 435 pairs:

- `market --requires--> demand` ↔ `server --requires--> load`
- Demand = load. Supply = capacity. Competition = resource contention.
- Autoscaling and circuit breakers have direct analogs in market design.

### The Legal System is an Immune System (surprise = 0.802)

Highest "surprise score" — structurally identical but semantically maximally distant:

| Legal System | Immune System | Shared Pattern |
|---|---|---|
| prosecution | antibody | Agent that neutralizes threats |
| verdict | enzyme activity | Decisive action that transforms state |
| appeal | protein competition | Adversarial challenge to a decision |
| statute | metabolic pathway | Codified process that governs behavior |

## Quantitative Results

| Metric | Value |
|---|---|
| Concepts probed | 144 |
| Domains discovered | 30 |
| Domain pairs analyzed | 435 |
| Relations extracted | 323 |
| Gaps detected | 9,755 |
| Non-trivial gap predictions | 6,835 |

## How It Works

- **Embedding model**: all-MiniLM-L6-v2 (384-dim, CPU, 80MB)
- **LLM analysis**: Llama 3.2 3B via Ollama (local)
- **Relation scoring**: Vector arithmetic — `subject + predicate ≈ object` in embedding space
- **Contextual embeddings**: "X in the context of domain Y" outperforms bare word embeddings
- **Surprise score**: `structural_similarity × semantic_distance` — surfaces the most interesting cross-domain analogies

## Files

| File | Description |
|---|---|
| `extract_domains_from_model.py` | Domain extraction from model embeddings (recursive splitting, surprise scoring) |
| `cross_domain_analogy.py` | Main analogy pipeline (comparison, gap detection, LLM analysis) |
| `deep_dive_music_combustion.py` | Focused analysis of Music Theory ↔ Internal Combustion |
| `example_domains.json` | Custom domain template (Immune/Cyber/Urban) |
| `FINDINGS.md` | Detailed findings document |

## Limitations

- Small embedding model (384-dim) — larger models would produce sharper results
- 144 seed concepts were manually selected — a production system should use LLM-generated vocabularies
- Relation scoring via vector arithmetic is approximate — knowledge graph methods (TransE, RotatE) would be more rigorous
- Gap predictions are hypotheses, not confirmed facts — each needs expert review

## License

MIT
