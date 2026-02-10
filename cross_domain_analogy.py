"""
Cross-Domain Analogy Discovery Pipeline
========================================
Discovers structural similarities between disconnected knowledge domains
using embedding geometry, then identifies gaps (missing relations) by analogy.

Runs locally on CPU with small models. No GPU required.

Usage:
    python cross_domain_analogy.py                    # uses built-in examples
    python cross_domain_analogy.py --ollama           # uses local LLM for deeper analysis
    python cross_domain_analogy.py --custom domains.json  # your own domains
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Step 0: Domain Knowledge Representation
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    subject: str
    predicate: str
    obj: str
    domain: str = ""

    def as_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}"

    def __repr__(self):
        return f"({self.subject} --{self.predicate}--> {self.obj})"


@dataclass
class Domain:
    name: str
    description: str
    entities: list[str]
    relations: list[Relation] = field(default_factory=list)

    def add_relation(self, subject: str, predicate: str, obj: str):
        self.relations.append(Relation(subject, predicate, obj, self.name))


def get_example_domains() -> list[Domain]:
    """Built-in example domains to demonstrate cross-domain analogy."""

    # --- Domain 1: Automotive ---
    auto = Domain(
        name="Automotive",
        description="How cars work — energy, propulsion, maintenance",
        entities=["car", "engine", "fuel", "combustion", "exhaust",
                  "movement", "brake", "friction", "heat", "wear",
                  "mechanic", "repair", "oil", "lubrication", "efficiency"],
    )
    auto.add_relation("car", "requires", "fuel")
    auto.add_relation("fuel", "enables", "combustion")
    auto.add_relation("combustion", "produces", "movement")
    auto.add_relation("combustion", "produces", "exhaust")
    auto.add_relation("combustion", "produces", "heat")
    auto.add_relation("engine", "requires", "oil")
    auto.add_relation("oil", "provides", "lubrication")
    auto.add_relation("lubrication", "reduces", "friction")
    auto.add_relation("friction", "causes", "wear")
    auto.add_relation("brake", "uses", "friction")
    auto.add_relation("brake", "enables", "stopping")
    auto.add_relation("mechanic", "performs", "repair")
    auto.add_relation("efficiency", "reduces", "fuel")
    auto.add_relation("exhaust", "is_waste_of", "combustion")

    # --- Domain 2: Cell Biology ---
    bio = Domain(
        name="Cell Biology",
        description="How cells work — metabolism, energy, maintenance",
        entities=["cell", "mitochondria", "glucose", "metabolism",
                  "free_radicals", "movement", "apoptosis", "membrane",
                  "ATP", "enzyme", "repair", "antioxidant", "efficiency"],
    )
    bio.add_relation("cell", "requires", "glucose")
    bio.add_relation("glucose", "enables", "metabolism")
    bio.add_relation("metabolism", "produces", "ATP")
    bio.add_relation("metabolism", "produces", "free_radicals")
    bio.add_relation("metabolism", "produces", "heat")
    bio.add_relation("mitochondria", "requires", "enzyme")
    bio.add_relation("enzyme", "provides", "catalysis")
    bio.add_relation("ATP", "enables", "movement")
    bio.add_relation("free_radicals", "causes", "damage")
    bio.add_relation("antioxidant", "reduces", "free_radicals")
    bio.add_relation("apoptosis", "enables", "stopping")
    bio.add_relation("cell", "performs", "repair")
    bio.add_relation("efficiency", "reduces", "glucose")
    bio.add_relation("free_radicals", "is_waste_of", "metabolism")

    # --- Domain 3: Software Systems ---
    software = Domain(
        name="Software Systems",
        description="How software systems work — resources, processing, maintenance",
        entities=["application", "CPU", "electricity", "processing",
                  "log_noise", "output", "shutdown", "memory",
                  "throughput", "garbage_collector", "maintenance",
                  "cache", "efficiency"],
    )
    software.add_relation("application", "requires", "electricity")
    software.add_relation("electricity", "enables", "processing")
    software.add_relation("processing", "produces", "output")
    software.add_relation("processing", "produces", "log_noise")
    software.add_relation("processing", "produces", "heat")
    software.add_relation("CPU", "requires", "cache")
    software.add_relation("cache", "provides", "speed")
    software.add_relation("throughput", "enables", "output")
    software.add_relation("log_noise", "causes", "storage_bloat")
    software.add_relation("garbage_collector", "reduces", "memory")
    software.add_relation("shutdown", "enables", "stopping")
    software.add_relation("application", "performs", "maintenance")
    software.add_relation("efficiency", "reduces", "electricity")
    software.add_relation("log_noise", "is_waste_of", "processing")

    # --- Domain 4: Economics ---
    econ = Domain(
        name="Economics",
        description="How economies work — resources, production, regulation",
        entities=["economy", "industry", "capital", "production",
                  "pollution", "growth", "recession", "market",
                  "GDP", "regulation", "reform", "subsidy", "efficiency"],
    )
    econ.add_relation("economy", "requires", "capital")
    econ.add_relation("capital", "enables", "production")
    econ.add_relation("production", "produces", "growth")
    econ.add_relation("production", "produces", "pollution")
    econ.add_relation("production", "produces", "inflation")
    econ.add_relation("industry", "requires", "subsidy")
    econ.add_relation("subsidy", "provides", "competitiveness")
    econ.add_relation("GDP", "enables", "growth")
    econ.add_relation("pollution", "causes", "health_costs")
    econ.add_relation("regulation", "reduces", "pollution")
    econ.add_relation("recession", "enables", "stopping")
    econ.add_relation("economy", "performs", "reform")
    econ.add_relation("efficiency", "reduces", "capital")
    econ.add_relation("pollution", "is_waste_of", "production")

    return [auto, bio, software, econ]


# ---------------------------------------------------------------------------
# Step 1: Embed Everything with Sentence Transformers
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Wraps sentence-transformers for encoding entities and relations."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}
        print("  Model loaded (runs on CPU, ~80MB).\n")

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            vecs = self.model.encode(uncached, normalize_embeddings=True)
            for t, v in zip(uncached, vecs):
                self._cache[t] = v
        return np.array([self._cache[t] for t in texts])

    def embed_contextual(self, entity: str, domain_name: str) -> np.ndarray:
        """Embed an entity with domain context for richer semantics."""
        key = f"_ctx_{domain_name}_{entity}"
        if key not in self._cache:
            text = f"{entity} in the context of {domain_name}"
            self._cache[key] = self.model.encode(text, normalize_embeddings=True)
        return self._cache[key]

    def embed_relation(self, rel: Relation) -> dict:
        """Embed a relation as: subject vec, object vec, and relation vector (offset)."""
        # Use contextual embeddings (entity + domain) for richer semantics
        domain = rel.domain
        subj_vec = self.embed_contextual(rel.subject, domain) if domain else self.embed(rel.subject)
        obj_vec = self.embed_contextual(rel.obj, domain) if domain else self.embed(rel.obj)
        pred_vec = self.embed(rel.predicate)
        # Also embed the full relation as a sentence
        full_text = f"In {domain}, {rel.subject} {rel.predicate} {rel.obj}" if domain else rel.as_text()
        full_vec = self.embed(full_text)
        # Relation vector = direction from subject to object in embedding space
        rel_vec = obj_vec - subj_vec
        return {
            "relation": rel,
            "subject_vec": subj_vec,
            "object_vec": obj_vec,
            "predicate_vec": pred_vec,
            "relation_vec": rel_vec,  # the key geometric signal
            "full_vec": full_vec,     # full sentence embedding
        }


# ---------------------------------------------------------------------------
# Step 2: Build Relational Signatures for Each Domain
# ---------------------------------------------------------------------------

@dataclass
class DomainSignature:
    """A domain's 'fingerprint' — the set of relational vectors."""
    domain: Domain
    embedded_relations: list[dict]
    predicate_vectors: dict  # predicate_name -> average relation vector

    def relation_matrix(self) -> np.ndarray:
        """Matrix of all relation vectors (N_relations x embedding_dim)."""
        return np.array([er["relation_vec"] for er in self.embedded_relations])


def build_domain_signature(domain: Domain, engine: EmbeddingEngine) -> DomainSignature:
    embedded = [engine.embed_relation(r) for r in domain.relations]

    # Group relation vectors by predicate and average them
    pred_groups: dict[str, list[np.ndarray]] = {}
    for er in embedded:
        pred = er["relation"].predicate
        pred_groups.setdefault(pred, []).append(er["relation_vec"])

    pred_vectors = {
        pred: np.mean(vecs, axis=0) for pred, vecs in pred_groups.items()
    }

    return DomainSignature(domain, embedded, pred_vectors)


# ---------------------------------------------------------------------------
# Step 3: Cross-Domain Structural Similarity
# ---------------------------------------------------------------------------

def compare_domain_predicates(sig_a: DomainSignature, sig_b: DomainSignature) -> dict:
    """Compare how predicates (relations) align between two domains."""
    results = []
    shared_predicates = set(sig_a.predicate_vectors.keys()) & set(sig_b.predicate_vectors.keys())

    for pred in shared_predicates:
        vec_a = sig_a.predicate_vectors[pred]
        vec_b = sig_b.predicate_vectors[pred]
        sim = 1 - cosine(vec_a, vec_b)
        results.append({
            "predicate": pred,
            "similarity": sim,
            "domain_a": sig_a.domain.name,
            "domain_b": sig_b.domain.name,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "domain_a": sig_a.domain.name,
        "domain_b": sig_b.domain.name,
        "shared_predicates": len(shared_predicates),
        "total_predicates_a": len(sig_a.predicate_vectors),
        "total_predicates_b": len(sig_b.predicate_vectors),
        "predicate_similarities": results,
        "avg_similarity": np.mean([r["similarity"] for r in results]) if results else 0,
    }


def compute_relation_alignment(sig_a: DomainSignature, sig_b: DomainSignature) -> list[dict]:
    """For each relation in domain A, find the most similar relation in domain B."""
    alignments = []

    for er_a in sig_a.embedded_relations:
        best_sim = -1
        best_match = None
        for er_b in sig_b.embedded_relations:
            sim = 1 - cosine(er_a["relation_vec"], er_b["relation_vec"])
            if sim > best_sim:
                best_sim = sim
                best_match = er_b
        alignments.append({
            "source": er_a["relation"],
            "target": best_match["relation"] if best_match else None,
            "similarity": best_sim,
        })

    alignments.sort(key=lambda x: x["similarity"], reverse=True)
    return alignments


# ---------------------------------------------------------------------------
# Step 4: Gap Detection — Find Missing Analogies
# ---------------------------------------------------------------------------

def find_gaps(sig_a: DomainSignature, sig_b: DomainSignature,
              engine: EmbeddingEngine, threshold: float = 0.3) -> list[dict]:
    """
    Find relations in domain A that have NO good match in domain B.
    These are potential 'gaps' — things that might exist in B but aren't modeled.

    Uses contextual embeddings and excludes self-referential predictions.
    """
    alignments = compute_relation_alignment(sig_a, sig_b)
    gaps = []

    domain_a_name = sig_a.domain.name
    domain_b_name = sig_b.domain.name

    for align in alignments:
        if align["similarity"] < threshold:
            rel_a = align["source"]
            # Use contextual embeddings: entity-in-domain, not bare words
            rel_vec = (engine.embed_contextual(rel_a.obj, domain_a_name)
                       - engine.embed_contextual(rel_a.subject, domain_a_name))

            # Find the most similar entity in B to the subject of A
            subj_sims = []
            for entity in sig_b.domain.entities:
                sim = 1 - cosine(
                    engine.embed_contextual(rel_a.subject, domain_a_name),
                    engine.embed_contextual(entity, domain_b_name),
                )
                subj_sims.append((entity, sim))
            subj_sims.sort(key=lambda x: x[1], reverse=True)
            mapped_subject = subj_sims[0][0] if subj_sims else None

            if mapped_subject:
                predicted_vec = engine.embed_contextual(mapped_subject, domain_b_name) + rel_vec
                norm = np.linalg.norm(predicted_vec)
                if norm > 0:
                    predicted_vec = predicted_vec / norm

                # Find nearest entity in B, EXCLUDING the mapped subject itself
                obj_sims = []
                for entity in sig_b.domain.entities:
                    if entity == mapped_subject:
                        continue  # prevent self-referential predictions
                    sim = 1 - cosine(
                        predicted_vec,
                        engine.embed_contextual(entity, domain_b_name),
                    )
                    obj_sims.append((entity, sim))
                obj_sims.sort(key=lambda x: x[1], reverse=True)
                predicted_object = obj_sims[0][0] if obj_sims else None
                prediction_score = obj_sims[0][1] if obj_sims else 0

                # Also get runner-up for context
                runner_up = obj_sims[1][0] if len(obj_sims) > 1 else None
            else:
                predicted_object = None
                prediction_score = 0
                runner_up = None

            gaps.append({
                "source_relation": rel_a,
                "source_domain": domain_a_name,
                "target_domain": domain_b_name,
                "best_match_similarity": align["similarity"],
                "mapped_subject": mapped_subject,
                "predicted_object": predicted_object,
                "runner_up_object": runner_up,
                "prediction_score": prediction_score,
                "predicted_relation": f"({mapped_subject} --{rel_a.predicate}--> {predicted_object})",
                "confidence": 1 - align["similarity"],
            })

    gaps.sort(key=lambda x: x["confidence"], reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# Step 5: (Optional) LLM-Powered Deep Analysis via Ollama
# ---------------------------------------------------------------------------

def analyze_with_ollama(gaps: list[dict], model: str = "llama3.2:3b") -> list[dict]:
    """Use a local LLM (via Ollama) to evaluate and expand on discovered gaps."""
    try:
        import requests
    except ImportError:
        print("  requests not installed, skipping LLM analysis")
        return gaps

    url = "http://localhost:11434/api/generate"
    enriched = []

    for gap in gaps[:5]:  # analyze top 5 gaps
        prompt = f"""You are a cross-domain analogy expert.

Domain A ({gap['source_domain']}) has this known relation:
  {gap['source_relation']}

Domain B ({gap['target_domain']}) appears to be MISSING an analogous relation.
My embedding-based analysis predicts this gap:
  {gap['predicted_relation']}

Questions:
1. Does this predicted analogy make sense? Why or why not?
2. What is the actual analogous concept in {gap['target_domain']}?
3. What NEW INSIGHTS does this cross-domain mapping suggest?
4. Are there any non-obvious dependencies this reveals?

Be specific and concise (max 200 words)."""

        try:
            resp = requests.post(url, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300}
            }, timeout=60)

            if resp.status_code == 200:
                llm_response = resp.json().get("response", "")
                gap["llm_analysis"] = llm_response
            else:
                gap["llm_analysis"] = f"(Ollama returned status {resp.status_code})"
        except requests.exceptions.ConnectionError:
            gap["llm_analysis"] = "(Ollama not running — start with: ollama serve)"
            break
        except Exception as e:
            gap["llm_analysis"] = f"(Error: {e})"

        enriched.append(gap)

    return enriched


# ---------------------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------------------

def print_domain_comparison(comparison: dict):
    print(f"\n{'='*60}")
    print(f"  {comparison['domain_a']}  <-->  {comparison['domain_b']}")
    print(f"{'='*60}")
    print(f"  Shared predicates: {comparison['shared_predicates']}")
    print(f"  Avg similarity:    {comparison['avg_similarity']:.3f}")
    print(f"  Predicate alignment:")
    for p in comparison["predicate_similarities"]:
        bar_len = int(p["similarity"] * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"    {p['predicate']:20s} [{bar}] {p['similarity']:.3f}")


def print_alignments(alignments: list[dict], domain_a: str, domain_b: str, top_n: int = 10):
    print(f"\n  Top relation alignments ({domain_a} -> {domain_b}):")
    print(f"  {'Source':40s} {'Target':40s} {'Sim':>6s}")
    print(f"  {'-'*40} {'-'*40} {'-'*6}")
    for a in alignments[:top_n]:
        src = str(a["source"])
        tgt = str(a["target"])
        print(f"  {src:40s} {tgt:40s} {a['similarity']:6.3f}")


def print_gaps(gaps: list[dict], top_n: int = 10):
    if not gaps:
        print("\n  No significant gaps found (domains are well-aligned).")
        return

    print(f"\n  DISCOVERED GAPS (potential missing relations):")
    print(f"  {'='*70}")
    for i, gap in enumerate(gaps[:top_n]):
        print(f"\n  Gap #{i+1} (confidence: {gap['confidence']:.3f})")
        print(f"    Known in {gap['source_domain']:15s}: {gap['source_relation']}")
        print(f"    Predicted in {gap['target_domain']:11s}: {gap['predicted_relation']}")
        if gap.get("runner_up_object"):
            print(f"    Runner-up prediction:    ...--{gap['source_relation'].predicate}--> {gap['runner_up_object']}")
        if gap.get("prediction_score"):
            print(f"    Prediction score: {gap['prediction_score']:.3f}")
        if "llm_analysis" in gap:
            print(f"    LLM Analysis:")
            for line in gap["llm_analysis"].split("\n"):
                print(f"      {line}")


def plot_domain_similarity_matrix(signatures: list[DomainSignature]):
    """Generate a heatmap of cross-domain similarity."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    n = len(signatures)
    matrix = np.zeros((n, n))
    names = [s.domain.name for s in signatures]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                comp = compare_domain_predicates(signatures[i], signatures[j])
                matrix[i][j] = comp["avg_similarity"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i][j] > 0.6 else "black")

    ax.set_title("Cross-Domain Structural Similarity")
    plt.colorbar(im, label="Avg Predicate Similarity")
    plt.tight_layout()

    out_path = Path(__file__).parent / "domain_similarity.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Heatmap saved to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_custom_domains(path: str) -> list[Domain]:
    """Load domains from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    domains = []
    for d in data["domains"]:
        domain = Domain(
            name=d["name"],
            description=d.get("description", ""),
            entities=d.get("entities", []),
        )
        for r in d.get("relations", []):
            domain.add_relation(r["subject"], r["predicate"], r["object"])
        domains.append(domain)
    return domains


def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Analogy Discovery")
    parser.add_argument("--ollama", action="store_true", help="Use local LLM for analysis")
    parser.add_argument("--ollama-model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--custom", type=str, help="Path to custom domains JSON file")
    parser.add_argument("--gap-threshold", type=float, default=0.3,
                        help="Similarity threshold below which a relation is a 'gap'")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    args = parser.parse_args()

    print("=" * 60)
    print("  CROSS-DOMAIN ANALOGY DISCOVERY PIPELINE")
    print("=" * 60)

    # Load domains
    if args.custom:
        print(f"\nLoading custom domains from: {args.custom}")
        domains = load_custom_domains(args.custom)
    else:
        print("\nUsing built-in example domains...")
        domains = get_example_domains()

    for d in domains:
        print(f"  [{d.name}] {len(d.entities)} entities, {len(d.relations)} relations")

    # Initialize embedding engine
    print()
    engine = EmbeddingEngine()

    # Build domain signatures
    print("Building domain signatures...")
    signatures = [build_domain_signature(d, engine) for d in domains]
    print(f"  Done. {len(signatures)} domains embedded.\n")

    # Compare all domain pairs
    print("\n" + "=" * 60)
    print("  CROSS-DOMAIN STRUCTURAL COMPARISONS")
    print("=" * 60)

    all_comparisons = []
    for sig_a, sig_b in combinations(signatures, 2):
        comp = compare_domain_predicates(sig_a, sig_b)
        all_comparisons.append(comp)
        print_domain_comparison(comp)

        # Show relation alignments
        alignments = compute_relation_alignment(sig_a, sig_b)
        print_alignments(alignments, sig_a.domain.name, sig_b.domain.name)

    # Find gaps
    print("\n\n" + "=" * 60)
    print("  GAP DETECTION — MISSING CROSS-DOMAIN ANALOGIES")
    print("=" * 60)

    all_gaps = []
    for sig_a, sig_b in combinations(signatures, 2):
        print(f"\n--- Gaps: {sig_a.domain.name} → {sig_b.domain.name} ---")
        gaps = find_gaps(sig_a, sig_b, engine, threshold=args.gap_threshold)
        print_gaps(gaps)
        all_gaps.extend(gaps)

        print(f"\n--- Gaps: {sig_b.domain.name} → {sig_a.domain.name} ---")
        gaps_rev = find_gaps(sig_b, sig_a, engine, threshold=args.gap_threshold)
        print_gaps(gaps_rev)
        all_gaps.extend(gaps_rev)

    # Optional: LLM analysis
    if args.ollama and all_gaps:
        print("\n\n" + "=" * 60)
        print("  LLM-POWERED DEEP ANALYSIS (via Ollama)")
        print("=" * 60)
        top_gaps = sorted(all_gaps, key=lambda x: x["confidence"], reverse=True)[:5]
        enriched = analyze_with_ollama(top_gaps, model=args.ollama_model)
        print_gaps(enriched)

    # Visualization
    if not args.no_plot:
        print("\n\nGenerating similarity heatmap...")
        plot_domain_similarity_matrix(signatures)

    # Summary
    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Domains analyzed:     {len(domains)}")
    print(f"  Domain pairs:         {len(all_comparisons)}")
    print(f"  Total gaps found:     {len(all_gaps)}")
    if all_comparisons:
        best = max(all_comparisons, key=lambda x: x["avg_similarity"])
        print(f"  Most similar pair:    {best['domain_a']} <-> {best['domain_b']} "
              f"(sim={best['avg_similarity']:.3f})")
    if all_gaps:
        top = max(all_gaps, key=lambda x: x["confidence"])
        print(f"  Highest-confidence gap:")
        print(f"    {top['source_relation']} ({top['source_domain']})")
        print(f"    → predicted in {top['target_domain']}: {top['predicted_relation']}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
