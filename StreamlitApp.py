"""
Zero-Shot Drug Repurposing with GraphSAGE

This Streamlit application enables zero-shot drug repurposing by combining
knowledge graph embeddings with text-based similarity. It uses the TDC PrimeKG
dataset and can be extended with custom models.

Author: Your Name
Date: 2025-10-31
"""

import streamlit as st
import torch
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required packages
try:
    from repurpose_assistant.models.gnn import GraphSAGEZeroShot
    from repurpose_assistant.scoring.zero_shot import ZeroShotScorer
    from repurpose_assistant.data.tdc_loader import load_tdc_repurposing_subset
    from repurpose_assistant.kg.graph_builder import build_kg_from_tdc
    from repurpose_assistant.models.train_gnn import build_adjacency_lists
    from repurpose_assistant.config import MODELS_DIR, TEXT_ENCODER_MODELS, DEFAULT_ENCODER
    from repurpose_assistant.model_utils import load_text_encoder, get_available_models
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    st.error("""
    Missing required dependencies. Please install them using:
    ```
    pip install -r requirements.txt
    ```
    """)
    st.stop()

# Set environment variables for model caching
os.environ['TRANSFORMERS_CACHE'] = str(Path(MODELS_DIR) / '.cache')
os.environ['HF_HOME'] = str(Path(MODELS_DIR) / '.cache')

# Constants
DEFAULT_ALPHA = 0.5
DEFAULT_TEMP = 1.0
DEFAULT_TOP_K = 10

def setup_page() -> None:
    """Configure the Streamlit page settings and navigation."""
    st.set_page_config(
        page_title="ReForgeMed - AI Drug Repurposing",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="💊"
    )
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Custom CSS for the top navigation
    st.markdown("""
    <style>
    /* Navigation bar styling */
    .nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .nav-title {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .nav-logo {
        width: 50px;
        height: 50px;
    }
    
    .nav-links {
        display: flex;
        gap: 1.5rem;
    }
    
    .nav-link {
        color: #4b5563;
        text-decoration: none;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .nav-link:hover {
        background-color: #e5e7eb;
        color: #1f2937;
    }
    
    .nav-link.active {
        background-color: #2563eb;
        color: white;
    }
    
    .main-title {
        font-size: 2rem !important;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    
    .subtitle {
        font-size: 1rem !important;
        color: #6b7280;
        margin: 0;
    }
    
    /* Hide the default Streamlit header */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Adjust main content padding */
    .main .block-container {
        padding-top: 0.5rem;
    }
    
    /* Hide Streamlit's default menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Create navigation columns
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Navigation bar with title and logo
        st.markdown("""
        <div class="nav">
            <div class="nav-title">
                <img src="https://img.icons8.com/color/96/000000/pill.png" class="nav-logo" alt="ReForgeMed Logo">
                <div>
                    <h1 class="main-title">ReForgeMed</h1>
                    <p class="subtitle">AI-Powered Drug Repurposing</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Navigation buttons using st.button
        st.markdown("<div class='nav-links' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
        
        # Home button
        if st.button("🏠 Home", key="nav_home"):
            st.session_state.page = 'home'
            st.rerun()
            
        # About button
        if st.button("ℹ️ About", key="nav_about"):
            st.session_state.page = 'about'
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add a separator
    st.markdown("---")

@st.cache_resource(show_spinner=False)
def load_data_and_models(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load TDC data, build knowledge graph, and initialize models.
    
    Args:
        model_name: Name of the text encoder model to use
        
    Returns:
        Dictionary containing loaded models and data
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        # Get the selected model name from session state or use default
        if model_name is None:
            model_name = st.session_state.get('current_encoder', DEFAULT_ENCODER)
        
        # Set device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Show device info
        if torch.cuda.is_available():
            st.sidebar.success(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.warning("💻 Using CPU (GPU not available)")
        
        # Get model info
        model_info = TEXT_ENCODER_MODELS.get(model_name, TEXT_ENCODER_MODELS[DEFAULT_ENCODER])
        st.sidebar.info(f"🧠 Using model: {model_name}")
        
        # Load TDC data
        with st.spinner("📥 Loading TDC PrimeKG data..."):
            data = load_tdc_repurposing_subset()
            if not all(k in data for k in ['splits', 'drug_info', 'disease_info']):
                raise ValueError("Invalid data format from TDC loader")
                
            splits = data['splits']
            drug_info = data['drug_info']
            disease_info = data['disease_info']
            target_info = data.get('target_info', {})
            loader = data['loader']
        
        # Build knowledge graph
        with st.spinner("🔗 Building knowledge graph..."):
            full_kg_df = loader.load_full_kg()
            kg, mappings = build_kg_from_tdc(drug_info, disease_info, target_info, splits['train'], full_kg_df)
            
        # Build adjacency lists
        with st.spinner("📊 Building adjacency lists..."):
            drug_ids = list(drug_info.keys())
            disease_ids = list(disease_info.keys())
            drug_adj, disease_adj = build_adjacency_lists(kg, mappings, len(drug_ids), len(disease_ids))
        
        # Load text encoder
        with st.spinner(f"🤖 Loading {model_name} model..."):
            encoder, tokenizer, max_length = load_text_encoder(model_name)
        
        # Initialize GNN model
        st.info("🤖 Initializing GraphSAGE model...")
        in_dim = 768  # Default dimension for most sentence transformers
        gnn = GraphSAGEZeroShot(in_dim=in_dim, hid_dims=[256, 128], dropout=0.2)
        
        # Try to load pre-trained model
        model_paths = [
            MODELS_DIR / "graphsage_best.pt",  # Original path
            Path("/mnt/d/Zero-shot/models_store/graphsage_best.pt"),  # WSL path
            Path("D:/Zero-shot/models_store/graphsage_best.pt")  # Windows path
        ]
        
        model_trained = False
        loaded_path = None
        
        for model_path in model_paths:
            st.info(f"🔍 Looking for model at: {model_path}")
            if model_path.exists():
                try:
                    st.info("🔍 Found model file. Loading...")
                    checkpoint = torch.load(str(model_path), map_location=device)
                    
                    # Debug: Print checkpoint keys
                    st.info(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        model_state_dict = checkpoint['model_state_dict']
                        st.info("✅ Found 'model_state_dict' in checkpoint")
                    elif 'state_dict' in checkpoint:
                        model_state_dict = checkpoint['state_dict']
                        st.info("✅ Found 'state_dict' in checkpoint")
                    else:
                        # If the checkpoint is the state dict itself
                        model_state_dict = checkpoint
                        st.info("✅ Checkpoint appears to be a direct state dict")
                    
                    # Debug: Print model state dict keys
                    st.info(f"🔍 Model state dict keys (first 5): {list(model_state_dict.keys())[:5]}...")
                    
                    # Get current model's state dict
                    current_state_dict = gnn.state_dict()
                    
                    # Check for architecture mismatch
                    missing_keys = [k for k in current_state_dict.keys() if k not in model_state_dict]
                    unexpected_keys = [k for k in model_state_dict.keys() if k not in current_state_dict]
                    
                    if missing_keys:
                        st.warning(f"⚠️ Missing keys in checkpoint: {missing_keys}")
                    if unexpected_keys:
                        st.warning(f"⚠️ Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
                    
                    # Create a new state dict with matching keys
                    new_state_dict = {}
                    for name, param in current_state_dict.items():
                        if name in model_state_dict:
                            # Check if shapes match
                            if param.shape == model_state_dict[name].shape:
                                new_state_dict[name] = model_state_dict[name]
                                st.info(f"✅ Loading parameter: {name} (shape: {param.shape})")
                            else:
                                st.warning(f"⚠️ Shape mismatch for {name}: expected {param.shape}, got {model_state_dict[name].shape}")
                                # Initialize with default weights if shapes don't match
                                new_state_dict[name] = param
                        else:
                            st.warning(f"⚠️ Parameter not found in checkpoint: {name}, using default initialization")
                            new_state_dict[name] = param
                    
                    # Load the state dict
                    gnn.load_state_dict(new_state_dict, strict=False)
                    gnn = gnn.to(device)
                    gnn.eval()
                    
                    # Verify model was loaded correctly
                    param_loaded = 0
                    for name, param in gnn.named_parameters():
                        if not torch.any(torch.isnan(param)) and not torch.any(torch.isinf(param)):
                            param_loaded += 1
                        else:
                            st.error(f"❌ Parameter {name} contains NaN or Inf values")
                    
                    # Verify model parameters were loaded
                    total_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
                    st.success(f"✅ Successfully loaded {param_loaded} parameters")
                    st.success(f"✅ Model has {total_params:,} total parameters")
                    
                    # If we got this far, consider the model loaded
                    model_trained = param_loaded > 0
                    if model_trained:
                        loaded_path = model_path
                        st.success(f"✅ Successfully loaded model from {model_path}")
                        break
                    
                except Exception as e:
                    st.error(f"❌ Error loading model from {model_path}: {str(e)}")
                    logger.exception(f"Error loading model from {model_path}")
        
        if not model_trained:
            st.warning("⚠️ Could not load model from any of the following paths:")
            for p in model_paths:
                exists = "✅ Exists" if p.exists() else "❌ Does not exist"
                st.warning(f"  - {p} ({exists})")
            
            # If no model was loaded, initialize a new one
            st.warning("⚠️ No trained model found or model loading failed. Using text similarity only.")
            gnn = gnn.to(device)
        else:
            st.success(f"✅ Using trained model from: {loaded_path}")
        
        # Initialize scorer
        scorer = ZeroShotScorer(alpha=DEFAULT_ALPHA, temperature=DEFAULT_TEMP)
        
        # Pre-compute drug embeddings for fast queries...
        st.info("⚡ Pre-computing embeddings...")
        with torch.no_grad():
            # Encode drugs and diseases
            drug_texts = [drug_info[did]['text'] for did in drug_ids]
            disease_texts = [disease_info[did]['text'] for did in disease_ids]
            
            # Get embeddings using the encoder
            drug_embeddings_np = encoder.encode(drug_texts)
            disease_embeddings_np = encoder.encode(disease_texts)
            
            # Convert to tensors if they're not already
            if not isinstance(drug_embeddings_np, torch.Tensor):
                drug_embeddings = torch.from_numpy(drug_embeddings_np).float().to(device)
            else:
                drug_embeddings = drug_embeddings_np.float().to(device)
                
            if not isinstance(disease_embeddings_np, torch.Tensor):
                disease_embeddings = torch.from_numpy(disease_embeddings_np).float().to(device)
            else:
                disease_embeddings = disease_embeddings_np.float().to(device)
            
            # Ensure proper shape (batch_size, embedding_dim)
            if len(drug_embeddings.shape) == 1:
                drug_embeddings = drug_embeddings.unsqueeze(0)
            if len(disease_embeddings.shape) == 1:
                disease_embeddings = disease_embeddings.unsqueeze(0)
            
            # Cache graph embeddings if model is trained
            if model_trained and hasattr(gnn, 'encode_drug'):
                try:
                    drug_z_cached = gnn.encode_drug(drug_embeddings, drug_adj)
                except Exception as e:
                    logger.warning(f"Could not encode drug embeddings with GNN: {e}")
                    model_trained = False
                    drug_z_cached = None
            else:
                drug_z_cached = None
                
        st.success("🎉 All systems ready!")
        
        return {
            'encoder': encoder,
            'drug_info': drug_info,
            'disease_info': disease_info,
            'drug_ids': drug_ids,
            'disease_ids': disease_ids,
            'drug_embeddings': drug_embeddings,
            'disease_embeddings': disease_embeddings,
            'kg': kg,
            'mappings': mappings,
            'drug_adj': drug_adj,
            'disease_adj': disease_adj,
            'gnn': gnn,
            'scorer': scorer,
            'splits': splits,
            'model_trained': model_trained,
            'drug_z_cached': drug_z_cached,
            'device': device
        }
        
    except Exception as e:
        logger.error(f"Error in load_data_and_models: {str(e)}")
        raise RuntimeError(f"Failed to load models and data: {str(e)}")

def display_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar UI and return user settings.
    
    Returns:
        Dictionary containing user settings
    """
    st.sidebar.header("Model Settings")
    
    # Model selection
    try:
        available_models = get_available_models()
        model_names = list(available_models.keys())
        
        selected_model = st.sidebar.selectbox(
            "Select Text Encoder",
            options=model_names,
            index=model_names.index(st.session_state.get('current_encoder', DEFAULT_ENCODER)),
            format_func=lambda x: f"{x} ({'✅' if available_models[x]['downloaded'] else '⬇️'})"
        )
        
        # Model info
        with st.sidebar.expander("ℹ️ Model Info"):
            st.write(available_models[selected_model]["description"])
            st.write(f"**Status:** {'Downloaded' if available_models[selected_model]['downloaded'] else 'Will download on first use'}")
        
        # Model controls
        if st.sidebar.button("🔄 Load Selected Model") and selected_model != st.session_state.get('current_encoder'):
            with st.spinner(f"Switching to {selected_model}..."):
                st.session_state.current_encoder = selected_model
                st.session_state.data_loaded = False
                st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.header("Search Settings")
        
        # Search parameters
        alpha = st.sidebar.slider(
            "Fusion weight (graph vs text)",
            0.0, 1.0, 
            st.session_state.get('alpha', DEFAULT_ALPHA),
            0.05,
            help="Weight for combining graph and text scores (0 = text only, 1 = graph only)"
        )
        
        temp = st.sidebar.slider(
            "Temperature",
            0.1, 5.0,
            st.session_state.get('temp', DEFAULT_TEMP),
            0.1,
            help="Temperature for score scaling (higher = more diverse results)"
        )
        
        top_k = st.sidebar.number_input(
            "Number of results",
            min_value=1,
            max_value=50,
            value=st.session_state.get('top_k', DEFAULT_TOP_K),
            step=1,
            help="Number of top candidates to show"
        )
        
        # Save settings to session state
        st.session_state.alpha = alpha
        st.session_state.temp = temp
        st.session_state.top_k = top_k
        
        return {
            'model_name': selected_model,
            'alpha': alpha,
            'temperature': temp,
            'top_k': top_k
        }
        
    except Exception as e:
        st.sidebar.error(f"Error in sidebar: {str(e)}")
        return {
            'model_name': DEFAULT_ENCODER,
            'alpha': DEFAULT_ALPHA,
            'temperature': DEFAULT_TEMP,
            'top_k': DEFAULT_TOP_K
        }

def display_main_content(settings: Dict[str, Any]) -> None:
    """
    Render the main content area and handle user interactions.
    
    Args:
        settings: Dictionary containing user settings
    """
    # Query input
    query = st.text_area(
        "Disease description",
        height=120,
        placeholder="Enter disease name/description, symptoms, phenotype notes...",
        help="Describe the disease or condition you want to find treatments for"
    )
    
    col1, col2 = st.columns([3, 2])
    
    if st.button("Find Candidates", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a disease description.")
            return
            
        try:
            # Load data and models
            with st.spinner("🚀 Initializing models and data..."):
                data = load_data_and_models(settings['model_name'])
                st.session_state.data_loaded = True
                
                # Update scorer with current settings
                data['scorer'].alpha = settings['alpha']
                data['scorer'].temperature = settings['temperature']
                
                # Show model status
                if data['model_trained']:
                    st.info("✅ Using trained GraphSAGE model for graph-based scoring")
                else:
                    st.warning("⚠️ No trained model found. Using text similarity only.")
            
            # Process query
            with st.spinner("🔍 Finding potential candidates..."):
                results = process_query(query, data, settings['top_k'])
                
            # Display results
            display_results(results, data, col1, col2)
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.exception("Error in display_main_content")

def process_query(query: str, data: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    """
    Process a disease query and return ranked drug candidates.
    
    Args:
        query: Disease description text
        data: Dictionary containing models and data
        top_k: Number of top results to return
        
    Returns:
        Dictionary containing query results
    """
    try:
        device = data['device']
        gnn = data['gnn']
        drug_embeddings = data['drug_embeddings']
        drug_ids = data['drug_ids']
        drug_info = data['drug_info']
        scorer = data['scorer']
        
        # Encode query and ensure it's a tensor on the correct device
        query_emb_np = data['encoder'].encode([query])
        if not isinstance(query_emb_np, torch.Tensor):
            query_emb = torch.from_numpy(query_emb_np).float().to(device)
        else:
            query_emb = query_emb_np.float().to(device)
        
        # Ensure drug_embeddings is a tensor
        if not isinstance(drug_embeddings, torch.Tensor):
            drug_embeddings = torch.from_numpy(drug_embeddings).to(device)
            
        # Reshape query_emb to match drug_embeddings if needed
        if len(query_emb.shape) == 1:
            query_emb = query_emb.unsqueeze(0)
            
        # Text-based scoring
        with torch.no_grad():
            # Ensure both tensors are 2D for cosine_similarity
            if len(drug_embeddings.shape) == 1:
                drug_embeddings = drug_embeddings.unsqueeze(0)
                
            text_scores = torch.cosine_similarity(
                query_emb, 
                drug_embeddings,
                dim=1
            )
            
            # Graph-based scoring
            graph_scores = None
            if data.get('model_trained', False) and 'drug_z_cached' in data and data['drug_z_cached'] is not None:
                try:
                    # Use pre-computed drug embeddings
                    drug_z = data['drug_z_cached']
                    
                    # Encode query disease (zero-shot)
                    query_disease_adj = {0: []}  # No neighbors for novel disease
                    
                    # Debug: Check model and input shapes
                    st.sidebar.write("---")
                    st.sidebar.write("Graph Model Debug:")
                    st.sidebar.write(f"- Model type: {type(gnn).__name__}")
                    st.sidebar.write(f"- Has encode_disease: {hasattr(gnn, 'encode_disease')}")
                    st.sidebar.write(f"- Input shapes - query_emb: {query_emb.shape}, drug_z: {drug_z.shape if drug_z is not None else 'None'}")
                    
                    if hasattr(gnn, 'encode_disease') and drug_z is not None:
                        query_disease_z = gnn.encode_disease(query_emb.unsqueeze(0), query_disease_adj)
                        
                        # Compute graph scores
                        if query_disease_z is not None and drug_z is not None:
                            graph_scores = torch.mm(drug_z, query_disease_z.t()).squeeze()
                            
                            # Debug: Print raw graph scores
                            st.sidebar.write("---")
                            st.sidebar.write("Graph Scores Debug:")
                            st.sidebar.write(f"- Raw scores shape: {graph_scores.shape}")
                            st.sidebar.write(f"- Raw scores min: {graph_scores.min().item():.4f}, max: {graph_scores.max().item():.4f}, mean: {graph_scores.mean().item():.4f}")
                            
                            # Normalize to [0, 1]
                            if graph_scores.max() > graph_scores.min():
                                graph_scores = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min())
                                st.sidebar.write(f"- Normalized scores min: {graph_scores.min().item():.4f}, max: {graph_scores.max().item():.4f}")
                            else:
                                st.sidebar.warning("Graph scores have no variation (min == max)")
                                graph_scores = torch.zeros_like(graph_scores)  # Fallback to zeros
                except Exception as e:
                    st.sidebar.error(f"❌ Error in graph-based scoring: {str(e)}")
                    logger.exception("Error in graph-based scoring")
                    graph_scores = None
            
            # Fallback if graph scoring failed
            if graph_scores is None:
                st.sidebar.warning("⚠️ Graph scoring not available. Using text similarity only.")
                graph_scores = torch.zeros_like(text_scores)
                data['model_trained'] = False
            else:
                # Fallback: simple neighbor count
                graph_scores = torch.zeros_like(text_scores)
                for i, drug_id in enumerate(drug_ids):
                    if drug_id in data['mappings']['drug_id_to_idx']:
                        drug_idx = data['mappings']['drug_id_to_idx'][drug_id]
                        drug_node = f"drug_{drug_idx}"
                        if drug_node in data['kg'].graph:
                            n_neighbors = len(list(data['kg'].graph.neighbors(drug_node)))
                            graph_scores[i] = min(n_neighbors / 10.0, 1.0)
            
            # Debug: Print score statistics before fusion
            st.sidebar.write("---")
            st.sidebar.write("Before Score Fusion:")
            st.sidebar.write(f"- Text scores shape: {text_scores.shape}, min: {text_scores.min().item():.4f}, max: {text_scores.max().item():.4f}")
            st.sidebar.write(f"- Graph scores shape: {graph_scores.shape}, min: {graph_scores.min().item():.4f}, max: {graph_scores.max().item():.4f}")
            
            # Debug: Print score statistics before fusion
            st.sidebar.write("---")
            st.sidebar.write("Before Score Fusion:")
            st.sidebar.write(f"- Text scores shape: {text_scores.shape}, min: {text_scores.min().item():.4f}, max: {text_scores.max().item():.4f}")
            st.sidebar.write(f"- Graph scores shape: {graph_scores.shape}, min: {graph_scores.min().item():.4f}, max: {graph_scores.max().item():.4f}")
            
            # Fuse scores and get top K
            try:
                fused = scorer.fuse_scores(graph_scores, text_scores)
                scores, indices = scorer.rank(fused, top_k=top_k)
                
                # Debug: Print top scores
                st.sidebar.write("---")
                st.sidebar.write(f"Top {top_k} scores:")
                for i, idx in enumerate(indices.cpu().numpy()):
                    if i < 5:  # Only show first 5 for brevity
                        st.sidebar.write(f"- {i+1}. Drug {drug_ids[idx]}: "
                                      f"Text={text_scores[idx]:.4f}, "
                                      f"Graph={graph_scores[idx].item() if data.get('model_trained', False) else 'N/A'}, "
                                      f"Fused={fused[idx]:.4f}")
                
                # Prepare results
                return {
                    'query': query,
                    'scores': scores.cpu().numpy(),
                    'indices': indices.cpu().numpy(),
                    'text_scores': text_scores.cpu().numpy(),
                    'graph_scores': graph_scores.cpu().numpy() if data.get('model_trained', False) else None,
                    'model_trained': data.get('model_trained', False)
                }
                
            except Exception as e:
                st.sidebar.error(f"❌ Error fusing scores: {str(e)}")
                logger.exception("Error fusing scores")
                
                # Fallback: Use text scores only
                scores, indices = torch.sort(text_scores, descending=True)
                indices = indices[:top_k]
                scores = scores[:top_k]
                
                return {
                    'query': query,
                    'scores': scores.cpu().numpy(),
                    'indices': indices.cpu().numpy(),
                    'text_scores': text_scores.cpu().numpy(),
                    'graph_scores': None,
                    'model_trained': False
                }
            
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        raise RuntimeError(f"Failed to process query: {str(e)}")

def display_results(results: Dict[str, Any], data: Dict[str, Any], col1, col2) -> None:
    """
    Display search results in the UI.
    
    Args:
        results: Dictionary containing query results
        data: Dictionary containing models and data
        col1: First column for layout
        col2: Second column for layout
    """
    drug_ids = data['drug_ids']
    drug_info = data['drug_info']
    splits = data['splits']
    
    with col1:
        st.subheader("Top Candidates")
        for rank, (i, score) in enumerate(zip(results['indices'], results['scores']), 1):
            drug_id = drug_ids[i]
            drug_name = drug_info[drug_id]['name']
            
            # Debug: Print score information
            st.sidebar.write("---")
            st.sidebar.write(f"Debug - Drug {drug_id}:")
            st.sidebar.write(f"- Model trained: {results['model_trained']}")
            st.sidebar.write(f"- Graph scores available: {results['graph_scores'] is not None}")
            if results['graph_scores'] is not None:
                st.sidebar.write(f"- Graph score shape: {results['graph_scores'].shape}")
                st.sidebar.write(f"- Graph score value: {results['graph_scores'][i]}")
            
            # Display drug card
            with st.expander(f"**{rank}. {drug_name}** ({drug_id})"):
                # Display main score
                st.caption(f"📊 **Score:** {score:.4f}")
                
                # Display text score
                st.caption(f"📝 **Text Similarity:** {results['text_scores'][i]:.3f}")
                
                # Display graph score if available
                if results['model_trained'] and results['graph_scores'] is not None:
                    try:
                        graph_score = float(results['graph_scores'][i])
                        st.caption(f"🌐 **Graph Score:** {graph_score:.3f}")
                    except Exception as e:
                        st.sidebar.error(f"Error formatting graph score: {e}")
                        st.caption("🌐 **Graph Score:** Error")
                else:
                    st.caption("🌐 **Graph Score:** Not available")
                
                # Add more drug info here if available
                if 'description' in drug_info[drug_id]:
                    st.write(drug_info[drug_id]['description'])
    
    with col2:
        st.subheader("Evidence & Rationales")
        for i in results['indices']:
            drug_id = drug_ids[i]
            drug_name = drug_info[drug_id]['name']
            
            # Find known indications
            known_diseases = splits['train'][
                splits['train']['drug_id'] == drug_id
            ]['disease_name'].unique().tolist()
            
            with st.container():
                st.markdown(f"**{drug_name}**")
                if known_diseases:
                    st.caption(f"Known indications: {', '.join(known_diseases[:3])}")
                    if len(known_diseases) > 3:
                        st.caption(f"...and {len(known_diseases) - 3} more")
                else:
                    st.caption("No known indications in training set (novel prediction)")
                st.markdown("---")

def show_about_page() -> None:
    """Display the About page with app information."""
    st.title("About ReForgeMed")
    
    st.markdown("""
    ## 🚀 Overview
    ReForgeMed is an AI-powered drug repurposing platform that leverages advanced machine learning 
    and knowledge graphs to identify potential new uses for existing drugs. Our platform helps 
    researchers discover novel therapeutic applications through zero-shot learning and graph-based analysis.
    """)
    
    st.markdown("## 🧠 Technologies Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Technologies
        - **Python**: Primary programming language
        - **PyTorch**: Deep learning framework
        - **Streamlit**: Web application framework
        - **Hugging Face Transformers**: NLP models
        - **NetworkX**: Graph processing
        - **Pandas & NumPy**: Data manipulation
        """)
    
    with col2:
        st.markdown("""
        ### AI/ML Components
        - Graph Neural Networks (GNNs)
        - Sentence Transformers
        - Knowledge Graph Embeddings
        - Zero-Shot Learning
        - Similarity Search
        """)
    
    st.markdown("## 🏆 The Team")
    
    team_members = [
        {"name": "Darshana V Zujam", "role": "Developer", "bio": "MSc Bioinformatics"},
        {"name": "Riyasingh Thakur", "role": "Developer", "bio": "MSc Bioinformatics"},
        
    ]
    
    for member in team_members:
        with st.expander(f"👤 {member['name']} - {member['role']}"):
            st.write(member['bio'])
    
    st.markdown("## 📚 Resources")
    st.markdown("""
    - [GitHub Repository](https://github.com/yourusername/reforgemed)
    - [Documentation](#) (coming soon)
    
    """)
    
    st.markdown("## 📬 Contact Us")
    st.markdown("""
    Have questions or feedback? We'd love to hear from you!
    - Email: darshanavzujam@gmail.com
    
    """)

def main() -> None:
    """Main entry point for the Streamlit application."""
    try:
        # Setup page (this will initialize session state if needed)
        setup_page()
        
        # Get current page from session state
        current_page = st.session_state.page
        
        # Display appropriate page based on navigation
        if current_page == 'about':
            show_about_page()
        else:  # Default to home page
            settings = display_sidebar()
            display_main_content(settings)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the logs for more details.")
        
        # Try to show the main content even if there's an error
        try:
            settings = display_sidebar()
            display_main_content(settings)
        except Exception as e:
            logger.error(f"Error in error handling: {str(e)}")
        
        # Add footer
        st.markdown("---")
        st.caption("""
        **Disclaimer**: This tool is for research purposes only. 
        The predictions provided by this application are not medical advice 
        and should not be used as a substitute for professional medical 
        diagnosis or treatment.
        """)
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Error in main")

if __name__ == "__main__":
    main()
