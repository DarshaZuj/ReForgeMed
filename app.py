import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import gc
import logging
from typing import Dict, Any, Optional

# -------------------------------------------------------------------
# CONFIGURATION & SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="Zero-Shot Drug Repurposing", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "DummyEncoder"
DEFAULT_ALPHA = 0.5


def log_memory_usage():
    """Log memory usage info."""
    try:
        import psutil, os
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)
        logger.info(f"💾 Memory usage: {mem:.2f} MB")
    except Exception:
        pass


# -------------------------------------------------------------------
# MINIMAL DATA LOADING
# -------------------------------------------------------------------
def load_minimal_dataset() -> Dict[str, Any]:
    """Creates a minimal dummy dataset for demonstration."""
    disease_info = {
        "D001": {"name": "Diabetes Mellitus"},
        "D002": {"name": "Alzheimer’s Disease"},
        "D003": {"name": "Breast Cancer"},
    }
    drug_info = {
        "DB001": {"name": "Metformin", "description": "Antidiabetic drug"},
        "DB002": {"name": "Donepezil", "description": "Alzheimer’s treatment"},
        "DB003": {"name": "Tamoxifen", "description": "Breast cancer therapy"},
        "DB004": {"name": "Aspirin", "description": "Pain reliever"},
    }
    target_info = {"T001": {"gene": "INSR"}, "T002": {"gene": "APP"}, "T003": {"gene": "ESR1"}}

    return {
        "splits": {},
        "drug_info": drug_info,
        "disease_info": disease_info,
        "target_info": target_info,
        "loader": None,
    }


# -------------------------------------------------------------------
# FIXED LOAD FUNCTION
# -------------------------------------------------------------------
def load_data_and_models(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads minimal dataset, builds dummy components (encoder, scorer, KG)
    and returns all required fields for process_query().
    """
    try:
        log_memory_usage()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if model_name is None:
            model_name = st.session_state.get("current_encoder", DEFAULT_ENCODER)
        st.sidebar.info(f"🧠 Using model: {model_name}")

        # free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # load minimal data
        with st.spinner("📥 Loading minimal dataset..."):
            data = load_minimal_dataset()
            num_diseases = len(data.get("disease_info", {}))
            num_drugs = len(data.get("drug_info", {}))
            st.sidebar.success(f"Loaded {num_diseases} diseases and {num_drugs} drugs")
            st.sidebar.warning("⚠️ Using minimal test dataset (demo mode)")

        embedding_dim = 16

        # ----- Dummy Components -----
        class DummyEncoder:
            def __init__(self, dim=embedding_dim, seed=None):
                self.dim = dim
                if seed is not None:
                    np.random.seed(seed)

            def encode(self, texts):
                if isinstance(texts, (list, tuple)):
                    return np.stack([np.random.randn(self.dim).astype(np.float32) for _ in texts])
                else:
                    return np.random.randn(self.dim).astype(np.float32)

            def get_embedding_dimension(self):
                return self.dim

        class DummyScorer:
            def __init__(self, alpha=0.5, device=device):
                self.alpha = alpha
                self.device = device

            def fuse_scores(self, graph_scores, text_scores):
                if not isinstance(graph_scores, torch.Tensor):
                    graph_scores = torch.tensor(graph_scores, dtype=torch.float32, device=self.device)
                if not isinstance(text_scores, torch.Tensor):
                    text_scores = torch.tensor(text_scores, dtype=torch.float32, device=self.device)
                alpha = float(st.session_state.get("alpha", self.alpha))
                return alpha * graph_scores + (1 - alpha) * text_scores

            def rank(self, fused, top_k=10):
                if not isinstance(fused, torch.Tensor):
                    fused = torch.tensor(fused, dtype=torch.float32, device=self.device)
                k = min(top_k, fused.numel())
                scores, idxs = torch.topk(fused, k)
                return scores, idxs

        class DummyKG:
            def __init__(self):
                self.graph = nx.Graph()

            def build_from_drug_ids(self, drug_ids):
                for i, did in enumerate(drug_ids):
                    self.graph.add_node(f"drug_{i}", drug_id=did)
                # connect some drugs
                for i in range(len(drug_ids) - 1):
                    self.graph.add_edge(f"drug_{i}", f"drug_{i+1}")

        # build components
        drug_info = data["drug_info"]
        drug_ids = list(drug_info.keys())
        encoder = DummyEncoder(seed=42)
        scorer = DummyScorer()
        kg = DummyKG()
        kg.build_from_drug_ids(drug_ids)

        try:
            drug_embeddings = encoder.encode([drug_info[d]["description"] for d in drug_ids])
        except Exception as e:
            logger.warning(f"Encoding fallback: {e}")
            drug_embeddings = np.random.randn(len(drug_ids), embedding_dim).astype(np.float32)

        mappings = {
            "drug_id_to_idx": {d: i for i, d in enumerate(drug_ids)},
            "idx_to_drug": {i: d for i, d in enumerate(drug_ids)},
            "disease_id_to_idx": {d: i for i, d in enumerate(data["disease_info"].keys())},
            "idx_to_disease": {i: d for i, d in enumerate(data["disease_info"].keys())},
        }

        return {
            "splits": data["splits"],
            "drug_info": drug_info,
            "disease_info": data["disease_info"],
            "target_info": data["target_info"],
            "loader": data["loader"],
            "gnn": None,
            "device": device,
            "model_name": model_name,
            "drug_embeddings": drug_embeddings,
            "drug_ids": drug_ids,
            "scorer": scorer,
            "encoder": encoder,
            "mappings": mappings,
            "kg": kg,
            "model_trained": False,
            "drug_z_cached": None,
        }

    except Exception as e:
        logger.exception("Error during load_data_and_models")
        st.error(f"❌ Failed to load: {e}")
        st.stop()


# -------------------------------------------------------------------
# QUERY FUNCTION
# -------------------------------------------------------------------
def process_query(query: str, data: Dict[str, Any], top_k: int = 5) -> pd.DataFrame:
    """
    Simulates drug ranking based on a disease query using embeddings and dummy scorer.
    """
    try:
        encoder = data["encoder"]
        scorer = data["scorer"]
        kg = data["kg"]
        drug_embeddings = data["drug_embeddings"]
        drug_ids = data["drug_ids"]

        # encode disease query
        text_emb = encoder.encode(query)
        if text_emb.ndim == 1:
            text_emb = text_emb.reshape(1, -1)
        text_tensor = torch.tensor(text_emb, dtype=torch.float32)

        # compute cosine similarity
        drug_tensor = torch.tensor(drug_embeddings, dtype=torch.float32)
        sim = torch.nn.functional.cosine_similarity(text_tensor, drug_tensor).squeeze()

        # fuse with dummy graph scores (e.g., small random noise)
        graph_scores = torch.randn_like(sim) * 0.05
        fused = scorer.fuse_scores(graph_scores, sim)
        scores, indices = scorer.rank(fused, top_k)

        top_results = []
        for i, idx in enumerate(indices):
            idx = idx.item()
            did = drug_ids[idx]
            score = scores[i].item()
            top_results.append({"Rank": i + 1, "Drug ID": did, "Name": data["drug_info"][did]["name"], "Score": round(score, 4)})

        return pd.DataFrame(top_results)

    except Exception as e:
        logger.exception("Error in process_query")
        st.error(f"❌ Error in processing query: {e}")
        return pd.DataFrame()


# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
def main():
    st.title("💊 Zero-Shot Drug Repurposing (Demo)")
    st.markdown("This demo uses a **minimal dataset** and a **dummy encoder/scorer** to simulate zero-shot drug repurposing.")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.alpha = st.slider("Fusion α (graph/text weight)", 0.0, 1.0, 0.5)

    data = load_data_and_models()

    st.success("✅ Data and dummy models loaded successfully!")

    query = st.text_input("Enter disease name or description:", "Breast cancer")
    top_k = st.slider("Top-K results:", 3, 10, 5)

    if st.button("🔍 Search Drugs"):
        with st.spinner("Searching for potential drug repurposing candidates..."):
            results = process_query(query, data, top_k)
            if not results.empty:
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No results found.")


if __name__ == "__main__":
    main()
