"""
Extract Domain Knowledge Directly From Model Weights
=====================================================
Mines the model's internal representations to discover:
  1. Natural domain clusters (from embedding geometry)
  2. Entities within each domain
  3. Relational structure between entities

No external text needed — all knowledge comes from the model itself.

Usage:
    python extract_domains_from_model.py                           # uses sentence-transformers (fast, CPU)
    python extract_domains_from_model.py --mode llm                # uses Llama via Ollama (richer)
    python extract_domains_from_model.py --mode transformer        # uses HF transformer hidden states
    python extract_domains_from_model.py --seed "physics,cooking"  # start from seed concepts
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Concept vocabulary — a broad set of concepts to probe the model with
# ---------------------------------------------------------------------------

# These are "probe words" — we feed them to the model and analyze how it
# organizes them internally. The model's representation reveals which
# concepts it considers related (same domain) vs. distant.

DEFAULT_CONCEPTS = [
    # Physical / mechanical (25)
    "engine", "fuel", "combustion", "wheel", "brake", "transmission",
    "friction", "torque", "exhaust", "piston", "gear", "axle",
    "crankshaft", "cylinder", "horsepower", "carburetor", "throttle",
    "flywheel", "camshaft", "spark_plug", "drivetrain", "suspension",
    "differential", "turbocharger", "RPM",
    # Biology / cell biology (25)
    "cell", "mitochondria", "glucose", "enzyme", "membrane", "DNA",
    "protein", "metabolism", "neuron", "antibody", "virus", "organ",
    "ribosome", "chromosome", "nucleus", "cytoplasm", "hemoglobin",
    "synapse", "hormone", "receptor", "pathogen", "immune_response",
    "gene_expression", "stem_cell", "apoptosis",
    # Computing / software (25)
    "CPU", "algorithm", "database", "network", "compiler",
    "thread", "cache", "bandwidth", "server", "encryption", "API",
    "operating_system", "firewall", "latency", "throughput",
    "virtualization", "recursion", "binary_tree", "hash_table",
    "load_balancer", "microservice", "container", "socket",
    "interrupt", "deadlock",
    # Economics (25)
    "inflation", "GDP", "market", "trade", "currency", "debt",
    "supply", "demand", "monopoly", "recession", "investment", "tax",
    "interest_rate", "fiscal_policy", "monetary_policy", "subsidy",
    "tariff", "commodity", "stock_exchange", "bond_market",
    "depreciation", "liquidity", "arbitrage", "dividend", "central_bank",
    # Chemistry (25)
    "molecule", "atom", "reaction", "catalyst", "bond", "electron",
    "oxidation", "acid", "solvent", "polymer", "crystal", "isotope",
    "valence", "covalent_bond", "ionic_bond", "pH", "titration",
    "electrolysis", "distillation", "precipitation", "equilibrium",
    "molar_mass", "reagent", "compound", "alloy",
    # Ecology (25)
    "ecosystem", "predator", "prey", "habitat", "photosynthesis", "extinction",
    "biodiversity", "food_chain", "decomposer", "symbiosis", "migration", "adaptation",
    "carrying_capacity", "trophic_level", "nitrogen_cycle", "carbon_cycle",
    "keystone_species", "invasive_species", "pollination", "biome",
    "deforestation", "coral_reef", "wetland", "succession", "mutualism",
    # Music (25)
    "melody", "harmony", "rhythm", "chord", "tempo", "pitch",
    "scale", "octave", "resonance", "timbre", "crescendo", "syncopation",
    "counterpoint", "modulation", "cadence", "arpeggio", "staccato",
    "legato", "vibrato", "overtone", "dissonance", "consonance",
    "fermata", "polyrhythm", "dynamics",
    # Architecture / construction (25)
    "foundation", "beam", "column", "arch", "concrete", "steel",
    "blueprint", "facade", "insulation", "ventilation", "truss",
    "cantilever", "buttress", "lintel", "rebar", "mortar",
    "load_bearing", "floor_plan", "roof_pitch", "structural_integrity",
    "building_code", "excavation", "scaffolding", "drywall", "masonry",
    # Psychology (25)
    "cognition", "emotion", "perception", "motivation", "anxiety",
    "behavior", "conditioning", "attention", "consciousness", "trauma", "empathy",
    "reinforcement", "cognitive_bias", "working_memory", "long_term_memory",
    "attachment_theory", "introspection", "habituation", "neuroplasticity",
    "psychotherapy", "dopamine", "serotonin", "phobia", "resilience",
    "cognitive_dissonance",
    # Cooking / food science (25)
    "fermentation", "caramelization", "emulsion", "seasoning", "simmering", "marination",
    "dough", "broth", "blanching", "braising", "tempering",
    "maillard_reaction", "deglaze", "roux", "brine", "poaching",
    "sous_vide", "smoking", "curing", "leavening", "gluten",
    "umami", "julienne", "flambe", "reduction_sauce",
    # Astronomy (25)
    "star", "planet", "orbit", "gravity", "nebula", "galaxy",
    "black_hole", "supernova", "asteroid", "comet", "pulsar", "quasar",
    "red_dwarf", "white_dwarf", "neutron_star", "solar_wind",
    "light_year", "redshift", "dark_matter", "dark_energy",
    "exoplanet", "accretion_disk", "cosmic_radiation", "magnetar", "protostar",
    # Law (25)
    "statute", "precedent", "jurisdiction", "liability", "contract", "tort",
    "prosecution", "verdict", "appeal", "arbitration", "injunction", "compliance",
    "due_process", "habeas_corpus", "indictment", "subpoena", "deposition",
    "plaintiff", "defendant", "litigation", "mediation", "felony",
    "misdemeanor", "jurisprudence", "fiduciary",
]

# Predicates to probe relational structure
RELATION_PROBES = [
    "requires", "produces", "causes", "enables", "prevents",
    "contains", "transforms", "regulates", "destroys", "creates",
    "depends_on", "interacts_with", "competes_with", "supports",
]


# ---------------------------------------------------------------------------
# Step 1: Embed all concepts using the model
# ---------------------------------------------------------------------------

class ModelProber:
    """Base class for probing a model's internal representations."""

    def embed_concept(self, concept: str) -> np.ndarray:
        raise NotImplementedError

    def embed_concepts(self, concepts: list[str]) -> np.ndarray:
        return np.array([self.embed_concept(c) for c in concepts])

    def probe_relation(self, subject: str, predicate: str, obj: str) -> float:
        """Score how strongly the model associates this triple."""
        raise NotImplementedError


class SentenceTransformerProber(ModelProber):
    """Probe using sentence-transformers (fast, CPU, ~80MB)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self._cache = {}
        print("  Ready.\n")

    def embed_concept(self, concept: str) -> np.ndarray:
        if concept not in self._cache:
            # Embed the concept as a short descriptive phrase
            # This gives richer representation than a bare word
            text = concept.replace("_", " ")
            self._cache[concept] = self.model.encode(text, normalize_embeddings=True)
        return self._cache[concept]

    def embed_in_context(self, concept: str, context: str) -> np.ndarray:
        key = f"{concept}__{context}"
        if key not in self._cache:
            text = f"{concept.replace('_', ' ')} in the context of {context}"
            self._cache[key] = self.model.encode(text, normalize_embeddings=True)
        return self._cache[key]

    def probe_relation(self, subject: str, predicate: str, obj: str) -> float:
        """Score a relation by embedding the full sentence and comparing."""
        sentence = f"{subject.replace('_', ' ')} {predicate.replace('_', ' ')} {obj.replace('_', ' ')}"
        sent_vec = self.model.encode(sentence, normalize_embeddings=True)
        # Compare to the expected vector (subject + predicate direction ≈ object)
        subj_vec = self.embed_concept(subject)
        obj_vec = self.embed_concept(obj)
        pred_vec = self.model.encode(predicate.replace("_", " "), normalize_embeddings=True)
        # Relation score: how close is (subj + pred) to obj?
        predicted = subj_vec + pred_vec
        predicted = predicted / np.linalg.norm(predicted)
        return float(np.dot(predicted, obj_vec))


class E5MistralProber(ModelProber):
    """Probe using intfloat/e5-mistral-7b-instruct (4096-dim, ~14GB RAM).

    This is a 7B-parameter embedding model built on Mistral-7B, specifically
    trained for high-quality text embeddings. Much richer representations than
    the 384-dim MiniLM but needs more RAM and is slower.
    """

    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct"):
        import torch
        from sentence_transformers import SentenceTransformer

        print(f"Loading E5-Mistral-7B embedding model...")
        print(f"  This needs ~14GB RAM. Loading in float16 for efficiency...")

        self.model = SentenceTransformer(model_name,
                                          model_kwargs={"torch_dtype": torch.float16})
        self._cache = {}
        dim = self.model.get_sentence_embedding_dimension()
        print(f"  Ready. Embedding dimension: {dim}\n")

    def _instruct_text(self, text: str, task: str = "clustering") -> str:
        """E5-Mistral uses instruction-prefixed queries for best results."""
        if task == "clustering":
            return f"Instruct: Identify the knowledge domain of this concept.\nQuery: {text}"
        elif task == "relation":
            return f"Instruct: Determine the semantic relationship.\nQuery: {text}"
        return text

    def embed_concept(self, concept: str) -> np.ndarray:
        if concept not in self._cache:
            text = concept.replace("_", " ")
            instruct = self._instruct_text(text, task="clustering")
            self._cache[concept] = self.model.encode(instruct, normalize_embeddings=True)
        return self._cache[concept]

    def embed_concepts(self, concepts: list[str]) -> np.ndarray:
        """Batch encode for efficiency — much faster than one-by-one."""
        uncached = [c for c in concepts if c not in self._cache]
        if uncached:
            texts = [self._instruct_text(c.replace("_", " "), task="clustering")
                     for c in uncached]
            vecs = self.model.encode(texts, normalize_embeddings=True,
                                      batch_size=8, show_progress_bar=True)
            for c, v in zip(uncached, vecs):
                self._cache[c] = v
        return np.array([self._cache[c] for c in concepts])

    def embed_in_context(self, concept: str, context: str) -> np.ndarray:
        key = f"{concept}__{context}"
        if key not in self._cache:
            text = f"{concept.replace('_', ' ')} in the context of {context}"
            instruct = self._instruct_text(text, task="clustering")
            self._cache[key] = self.model.encode(instruct, normalize_embeddings=True)
        return self._cache[key]

    def probe_relation(self, subject: str, predicate: str, obj: str) -> float:
        subj_vec = self.embed_concept(subject)
        obj_vec = self.embed_concept(obj)
        pred_text = self._instruct_text(predicate.replace("_", " "), task="relation")
        pred_key = f"_pred_{predicate}"
        if pred_key not in self._cache:
            self._cache[pred_key] = self.model.encode(pred_text, normalize_embeddings=True)
        pred_vec = self._cache[pred_key]
        predicted = subj_vec + pred_vec
        predicted = predicted / np.linalg.norm(predicted)
        return float(np.dot(predicted, obj_vec))


class OllamaProber(ModelProber):
    """Probe using a local LLM via Ollama — richer but slower."""

    def __init__(self, model_name: str = "llama3.2:3b"):
        import requests
        self.model_name = model_name
        self.url = "http://localhost:11434/api/embeddings"
        self._cache = {}
        # Test connection
        try:
            resp = requests.post(self.url, json={
                "model": model_name, "prompt": "test"
            }, timeout=30)
            if resp.status_code == 200:
                print(f"Connected to Ollama ({model_name}) for embeddings.\n")
            else:
                print(f"Warning: Ollama returned status {resp.status_code}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            sys.exit(1)

    def embed_concept(self, concept: str) -> np.ndarray:
        import requests
        if concept not in self._cache:
            text = concept.replace("_", " ")
            resp = requests.post(self.url, json={
                "model": self.model_name, "prompt": text
            }, timeout=30)
            self._cache[concept] = np.array(resp.json()["embedding"], dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(self._cache[concept])
            if norm > 0:
                self._cache[concept] /= norm
        return self._cache[concept]

    def probe_relation(self, subject: str, predicate: str, obj: str) -> float:
        subj_vec = self.embed_concept(subject)
        obj_vec = self.embed_concept(obj)
        pred_vec = self.embed_concept(predicate)
        predicted = subj_vec + pred_vec
        predicted = predicted / np.linalg.norm(predicted)
        return float(np.dot(predicted, obj_vec))


class TransformerProber(ModelProber):
    """Probe using HuggingFace transformer hidden states (most powerful)."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading transformer: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        self._cache = {}
        print(f"  Ready. ({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)\n")

    def embed_concept(self, concept: str) -> np.ndarray:
        import torch
        if concept not in self._cache:
            text = concept.replace("_", " ")
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use mean of last hidden state as representation
            hidden = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            hidden = hidden / np.linalg.norm(hidden)
            self._cache[concept] = hidden
        return self._cache[concept]

    def probe_relation(self, subject: str, predicate: str, obj: str) -> float:
        subj_vec = self.embed_concept(subject)
        obj_vec = self.embed_concept(obj)
        pred_vec = self.embed_concept(predicate)
        predicted = subj_vec + pred_vec
        predicted = predicted / np.linalg.norm(predicted)
        return float(np.dot(predicted, obj_vec))


# ---------------------------------------------------------------------------
# Step 2: Cluster concepts to discover domains
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredDomain:
    name: str
    concepts: list[str]
    centroid: np.ndarray
    coherence: float  # how tight the cluster is

    def __repr__(self):
        top = self.concepts[:8]
        return f"Domain('{self.name}', {len(self.concepts)} concepts: {top}...)"


def _build_domain(label: int, concepts: list[str],
                  embeddings: np.ndarray) -> DiscoveredDomain:
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    coherence = float(sims.mean())
    return DiscoveredDomain(
        name=f"Domain_{label}",
        concepts=concepts,
        centroid=centroid,
        coherence=coherence,
    )


def discover_domains_recursive(concepts: list[str], embeddings: np.ndarray,
                               max_size: int = 10,
                               min_size: int = 3,
                               min_coherence: float = 0.65,
                               depth: int = 0,
                               max_depth: int = 5,
                               _label_counter: list | None = None) -> list[DiscoveredDomain]:
    """
    Recursively split clusters until every domain is tight and specific.

    A cluster is split further when:
      - It has more than `max_size` concepts, OR
      - Its coherence is below `min_coherence`
    Splitting stops when:
      - Cluster is small enough AND coherent enough
      - Cluster would split below `min_size`
      - Max recursion depth reached
    """
    if _label_counter is None:
        _label_counter = [0]

    indent = "  " + "  " * depth

    # Base case: small and tight enough, or can't split further
    if len(concepts) <= max_size and depth > 0:
        domain = _build_domain(_label_counter[0], concepts, embeddings)
        if domain.coherence >= min_coherence or len(concepts) <= min_size:
            _label_counter[0] += 1
            print(f"{indent}  LEAF: {len(concepts)} concepts, coherence={domain.coherence:.3f}")
            return [domain]

    if len(concepts) < min_size * 2 or depth >= max_depth:
        domain = _build_domain(_label_counter[0], concepts, embeddings)
        _label_counter[0] += 1
        print(f"{indent}  LEAF (limit): {len(concepts)} concepts, coherence={domain.coherence:.3f}")
        return [domain]

    # Decide how many sub-clusters: split into 2 for binary recursion
    # (cleaner than guessing k)
    n_splits = min(2, max(2, len(concepts) // max_size))
    if len(concepts) > max_size * 3:
        n_splits = 3  # allow 3-way split for very large clusters

    current_coherence = float(cosine_similarity(
        embeddings, embeddings.mean(axis=0).reshape(1, -1)
    ).mean())

    print(f"{indent}Splitting {len(concepts)} concepts (coherence={current_coherence:.3f}) "
          f"into {n_splits} sub-clusters (depth={depth})")

    clusterer = KMeans(n_clusters=n_splits, random_state=42 + depth, n_init=10)
    labels = clusterer.fit_predict(embeddings)

    all_domains = []
    for sub_label in range(n_splits):
        mask = labels == sub_label
        sub_concepts = [c for c, m in zip(concepts, mask) if m]
        sub_embeddings = embeddings[mask]

        if len(sub_concepts) < min_size:
            # Too small to be meaningful, keep as leaf
            domain = _build_domain(_label_counter[0], sub_concepts, sub_embeddings)
            _label_counter[0] += 1
            all_domains.append(domain)
            continue

        # Check if this sub-cluster needs further splitting
        sub_coherence = float(cosine_similarity(
            sub_embeddings, sub_embeddings.mean(axis=0).reshape(1, -1)
        ).mean())

        needs_split = (len(sub_concepts) > max_size or
                       sub_coherence < min_coherence)

        if needs_split and len(sub_concepts) >= min_size * 2:
            # Recurse
            sub_domains = discover_domains_recursive(
                sub_concepts, sub_embeddings,
                max_size=max_size, min_size=min_size,
                min_coherence=min_coherence,
                depth=depth + 1, max_depth=max_depth,
                _label_counter=_label_counter,
            )
            all_domains.extend(sub_domains)
        else:
            domain = _build_domain(_label_counter[0], sub_concepts, sub_embeddings)
            _label_counter[0] += 1
            print(f"{indent}  LEAF: {len(sub_concepts)} concepts, "
                  f"coherence={domain.coherence:.3f}")
            all_domains.append(domain)

    return all_domains


def discover_domains(concepts: list[str], embeddings: np.ndarray,
                     n_domains: int | None = None,
                     min_cluster_size: int = 4,
                     recursive: bool = False,
                     max_size: int = 10,
                     min_coherence: float = 0.65) -> list[DiscoveredDomain]:
    """Cluster concept embeddings to discover natural domains."""

    if recursive:
        print(f"  Recursive splitting: {len(concepts)} concepts "
              f"(max_size={max_size}, min_coherence={min_coherence})")
        domains = discover_domains_recursive(
            concepts, embeddings,
            max_size=max_size, min_size=min_cluster_size,
            min_coherence=min_coherence,
        )
        domains.sort(key=lambda d: d.coherence, reverse=True)
        print(f"\n  Recursive splitting produced {len(domains)} domains.\n")
        return domains

    if n_domains is not None:
        print(f"  Clustering {len(concepts)} concepts into {n_domains} domains (KMeans)...")
        clusterer = KMeans(n_clusters=n_domains, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
    else:
        print(f"  Clustering {len(concepts)} concepts (HDBSCAN, auto domain count)...")
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine")
        labels = clusterer.fit_predict(embeddings)

    domains = []
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_concepts = [c for c, m in zip(concepts, mask) if m]
        cluster_embeddings = embeddings[mask]
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        sims = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        coherence = float(sims.mean())
        domains.append(DiscoveredDomain(
            name=f"Domain_{label}",
            concepts=cluster_concepts,
            centroid=centroid,
            coherence=coherence,
        ))

    domains.sort(key=lambda d: d.coherence, reverse=True)
    print(f"  Found {len(domains)} domains.\n")
    return domains


# ---------------------------------------------------------------------------
# Step 3: Name domains using the LLM (or heuristics)
# ---------------------------------------------------------------------------

def name_domains_heuristic(domains: list[DiscoveredDomain]) -> list[DiscoveredDomain]:
    """Name domains based on the most central concepts."""
    for domain in domains:
        # Use the first few concepts as a rough name
        top_concepts = [c.replace("_", " ") for c in domain.concepts[:3]]
        domain.name = " / ".join(top_concepts)
    return domains


def name_domains_with_ollama(domains: list[DiscoveredDomain],
                             model: str = "llama3.2:3b") -> list[DiscoveredDomain]:
    """Use local LLM to generate meaningful domain names."""
    import requests
    url = "http://localhost:11434/api/generate"

    for domain in domains:
        concept_list = ", ".join(c.replace("_", " ") for c in domain.concepts)
        prompt = (
            f"These concepts belong to the same knowledge domain:\n"
            f"  {concept_list}\n\n"
            f"What is the single best name for this domain? "
            f"Reply with ONLY the domain name (2-4 words), nothing else."
        )
        try:
            resp = requests.post(url, json={
                "model": model, "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 20}
            }, timeout=30)
            if resp.status_code == 200:
                name = resp.json().get("response", "").strip().strip('"').strip(".")
                if name:
                    domain.name = name
        except Exception:
            pass  # fall back to heuristic name

    return domains


# ---------------------------------------------------------------------------
# Step 4: Extract relations within each domain
# ---------------------------------------------------------------------------

@dataclass
class ExtractedRelation:
    subject: str
    predicate: str
    obj: str
    score: float
    domain: str = ""


def extract_relations(domain: DiscoveredDomain, prober: ModelProber,
                      predicates: list[str] = None,
                      top_n: int = 15) -> list[ExtractedRelation]:
    """
    Discover relations between concepts within a domain by probing
    the model's embedding geometry.

    For each (concept_A, predicate, concept_B) triple, score it using
    vector arithmetic: does (A + predicate) ≈ B in embedding space?
    """
    if predicates is None:
        predicates = RELATION_PROBES

    relations = []
    concepts = domain.concepts

    for pred in predicates:
        for subj in concepts:
            for obj in concepts:
                if subj == obj:
                    continue
                score = prober.probe_relation(subj, pred, obj)
                if score > 0.3:  # threshold for plausibility
                    relations.append(ExtractedRelation(
                        subject=subj, predicate=pred, obj=obj,
                        score=score, domain=domain.name,
                    ))

    # Sort by score, deduplicate (keep highest scoring predicate per pair)
    relations.sort(key=lambda r: r.score, reverse=True)

    # Keep only the best predicate for each (subject, object) pair
    seen_pairs = set()
    deduplicated = []
    for rel in relations:
        pair = (rel.subject, rel.obj)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            deduplicated.append(rel)

    return deduplicated[:top_n]


# ---------------------------------------------------------------------------
# Step 5: Visualization
# ---------------------------------------------------------------------------

def visualize_domains(concepts: list[str], embeddings: np.ndarray,
                      domains: list[DiscoveredDomain]):
    """Create a t-SNE scatter plot colored by domain."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    # Build concept-to-domain mapping
    concept_to_domain = {}
    for i, domain in enumerate(domains):
        for c in domain.concepts:
            concept_to_domain[c] = i

    # t-SNE reduction
    print("  Running t-SNE...")
    perplexity = min(30, len(concepts) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, metric="cosine")
    coords = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = cm.tab20(np.linspace(0, 1, len(domains)))

    for i, domain in enumerate(domains):
        mask = [c in domain.concepts for c in concepts]
        domain_coords = coords[mask]
        ax.scatter(domain_coords[:, 0], domain_coords[:, 1],
                   c=[colors[i]], label=domain.name, s=60, alpha=0.7)

        # Label each point
        domain_concepts = [c for c, m in zip(concepts, mask) if m]
        for j, (x, y) in enumerate(domain_coords):
            ax.annotate(domain_concepts[j].replace("_", " "),
                        (x, y), fontsize=7, alpha=0.8,
                        xytext=(5, 5), textcoords="offset points")

    # Also plot unclustered points
    unclustered = [c for c in concepts if c not in concept_to_domain]
    if unclustered:
        mask = [c in unclustered for c in concepts]
        unc_coords = coords[mask]
        ax.scatter(unc_coords[:, 0], unc_coords[:, 1],
                   c="gray", label="unclustered", s=30, alpha=0.3, marker="x")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title("Concept Domains Discovered from Model Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()

    out_path = Path(__file__).parent / "discovered_domains.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Step 6: Export to cross_domain_analogy.py format
# ---------------------------------------------------------------------------

def export_domains_json(domains: list[DiscoveredDomain],
                        all_relations: dict[str, list[ExtractedRelation]],
                        output_path: str):
    """Export discovered domains to JSON for use with cross_domain_analogy.py."""
    data = {"domains": []}
    for domain in domains:
        relations_list = []
        for rel in all_relations.get(domain.name, []):
            relations_list.append({
                "subject": rel.subject,
                "predicate": rel.predicate,
                "object": rel.obj,
            })
        data["domains"].append({
            "name": domain.name,
            "description": f"Auto-discovered domain with {len(domain.concepts)} concepts",
            "entities": domain.concepts,
            "relations": relations_list,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Exported to: {output_path}")
    print(f"  Use with: python cross_domain_analogy.py --custom {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract domains from model")
    parser.add_argument("--mode", choices=["sentence", "e5", "ollama", "transformer"],
                        default="sentence",
                        help="Which model to probe (e5 = e5-mistral-7b-instruct, 4096-dim)")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--transformer-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--n-domains", type=int, default=None,
                        help="Force a specific number of domains (default: auto)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively split clusters until tight and specific")
    parser.add_argument("--max-size", type=int, default=10,
                        help="Max concepts per domain before splitting (default: 10)")
    parser.add_argument("--min-coherence", type=float, default=0.65,
                        help="Min coherence before splitting (default: 0.65)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Comma-separated seed concepts to expand from")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", default="discovered_domains.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  DOMAIN EXTRACTION FROM MODEL REPRESENTATIONS")
    print("=" * 60)

    # Select prober
    if args.mode == "e5":
        prober = E5MistralProber()
    elif args.mode == "ollama":
        prober = OllamaProber(args.ollama_model)
    elif args.mode == "transformer":
        prober = TransformerProber(args.transformer_model)
    else:
        prober = SentenceTransformerProber()

    # Select concepts
    if args.seed:
        seeds = [s.strip() for s in args.seed.split(",")]
        print(f"\nSeed concepts: {seeds}")
        concepts = expand_seeds_with_ollama(seeds, args.ollama_model) if args.mode == "ollama" else DEFAULT_CONCEPTS
    else:
        concepts = DEFAULT_CONCEPTS

    print(f"\nProbing model with {len(concepts)} concepts...")

    # Step 1: Embed all concepts
    embeddings = prober.embed_concepts(concepts)
    print(f"  Embedding shape: {embeddings.shape}")

    # Step 2: Discover domains via clustering
    print("\nDiscovering domains...")
    domains = discover_domains(
        concepts, embeddings,
        n_domains=args.n_domains,
        recursive=args.recursive,
        max_size=args.max_size,
        min_coherence=args.min_coherence,
    )

    # Step 3: Name domains
    print("Naming domains...")
    if args.mode == "ollama":
        domains = name_domains_with_ollama(domains, args.ollama_model)
    else:
        # Try Ollama if available, fall back to heuristic
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                domains = name_domains_with_ollama(domains, "llama3.2:3b")
            else:
                domains = name_domains_heuristic(domains)
        except Exception:
            domains = name_domains_heuristic(domains)

    for domain in domains:
        print(f"\n  [{domain.name}] (coherence: {domain.coherence:.3f})")
        print(f"    Concepts: {', '.join(c.replace('_', ' ') for c in domain.concepts)}")

    # Step 4: Extract relations within each domain
    print("\n\nExtracting relational structure...")
    all_relations = {}
    for domain in domains:
        print(f"\n  [{domain.name}]")
        relations = extract_relations(domain, prober, top_n=15)
        all_relations[domain.name] = relations
        for rel in relations[:10]:
            print(f"    ({rel.subject} --{rel.predicate}--> {rel.obj})  score={rel.score:.3f}")

    # Step 5: Visualize
    if not args.no_plot:
        print("\n\nGenerating visualization...")
        visualize_domains(concepts, embeddings, domains)

    # Step 6: Export
    out_path = Path(__file__).parent / args.output
    export_domains_json(domains, all_relations, str(out_path))

    # Step 6b: Cross-domain surprise analysis
    # Find domain pairs with HIGH structural similarity but LOW semantic similarity
    # These are the most interesting discoveries
    if len(domains) >= 2:
        print(f"\n{'='*60}")
        print(f"  CROSS-DOMAIN SURPRISE ANALYSIS")
        print(f"  (structurally similar BUT semantically distant)")
        print(f"{'='*60}")

        from itertools import combinations
        from scipy.spatial.distance import cosine as cos_dist

        surprises = []
        for d_a, d_b in combinations(domains, 2):
            # Semantic distance: how far apart are the domain centroids?
            semantic_dist = cos_dist(d_a.centroid, d_b.centroid)

            # Structural similarity: compare relation vectors
            rels_a = all_relations.get(d_a.name, [])
            rels_b = all_relations.get(d_b.name, [])
            if not rels_a or not rels_b:
                continue

            # Compare predicate distributions
            preds_a = {}
            for r in rels_a:
                preds_a.setdefault(r.predicate, []).append(r.score)
            preds_b = {}
            for r in rels_b:
                preds_b.setdefault(r.predicate, []).append(r.score)

            shared_preds = set(preds_a.keys()) & set(preds_b.keys())
            if not shared_preds:
                continue

            # Structural similarity = correlation of predicate usage patterns
            pred_scores_a = [np.mean(preds_a.get(p, [0])) for p in shared_preds]
            pred_scores_b = [np.mean(preds_b.get(p, [0])) for p in shared_preds]

            if len(shared_preds) >= 2:
                structural_sim = float(np.corrcoef(pred_scores_a, pred_scores_b)[0, 1])
                if np.isnan(structural_sim):
                    structural_sim = 0.0
            else:
                structural_sim = 1.0 - abs(pred_scores_a[0] - pred_scores_b[0])

            # SURPRISE SCORE = structural_similarity × semantic_distance
            # High when: domains are far apart semantically BUT behave similarly structurally
            surprise = structural_sim * semantic_dist

            surprises.append({
                "domain_a": d_a.name,
                "domain_b": d_b.name,
                "semantic_distance": semantic_dist,
                "structural_similarity": structural_sim,
                "surprise_score": surprise,
                "shared_predicates": list(shared_preds),
                "concepts_a": d_a.concepts[:5],
                "concepts_b": d_b.concepts[:5],
            })

        surprises.sort(key=lambda x: x["surprise_score"], reverse=True)

        print(f"\n  {'Rank':<5} {'Surprise':<10} {'Sem.Dist':<10} {'Struct.Sim':<11} {'Domain A':<28} {'Domain B':<28}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*11} {'-'*28} {'-'*28}")

        for i, s in enumerate(surprises[:20], 1):
            print(f"  {i:<5} {s['surprise_score']:<10.3f} {s['semantic_distance']:<10.3f} "
                  f"{s['structural_similarity']:<11.3f} {s['domain_a']:<28} {s['domain_b']:<28}")

        # Detail the top 5 most surprising pairs
        print(f"\n  TOP SURPRISING ANALOGIES (distant domains, similar structure):")
        print(f"  {'='*70}")
        for i, s in enumerate(surprises[:5], 1):
            print(f"\n  #{i} Surprise={s['surprise_score']:.3f}")
            print(f"     {s['domain_a']}: {', '.join(s['concepts_a'])}")
            print(f"     {s['domain_b']}: {', '.join(s['concepts_b'])}")
            print(f"     Semantic distance:      {s['semantic_distance']:.3f} (1.0 = completely different)")
            print(f"     Structural similarity:  {s['structural_similarity']:.3f} (1.0 = identical pattern)")
            print(f"     Shared predicates:      {', '.join(s['shared_predicates'])}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Model probed:      {args.mode}")
    print(f"  Concepts probed:   {len(concepts)}")
    print(f"  Domains found:     {len(domains)}")
    total_rels = sum(len(v) for v in all_relations.values())
    print(f"  Relations found:   {total_rels}")
    print(f"  Output:            {out_path}")
    print(f"\n  Next step: python cross_domain_analogy.py --custom {out_path}")


def expand_seeds_with_ollama(seeds: list[str], model: str) -> list[str]:
    """Use LLM to expand seed concepts into a full vocabulary."""
    import requests
    url = "http://localhost:11434/api/generate"
    all_concepts = list(seeds)

    for seed in seeds:
        prompt = (
            f"List 15 key concepts/entities that are closely related to '{seed}' "
            f"in its knowledge domain. Return ONLY a comma-separated list of "
            f"single words or short phrases. No numbering, no explanations."
        )
        try:
            resp = requests.post(url, json={
                "model": model, "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            }, timeout=30)
            if resp.status_code == 200:
                text = resp.json().get("response", "")
                concepts = [c.strip().lower().replace(" ", "_")
                            for c in text.split(",")]
                concepts = [c for c in concepts if c and len(c) < 30]
                all_concepts.extend(concepts)
        except Exception:
            pass

    # Deduplicate
    seen = set()
    unique = []
    for c in all_concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


if __name__ == "__main__":
    main()
