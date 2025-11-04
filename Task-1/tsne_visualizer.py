import streamlit as st
import torch
import pickle
import glob
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from collections import Counter

def plot_embedding_tsne_streamlit(model, itos, word_freq, top_n=200, perplexity=30, seed=42):
    most_common = sorted(word_freq.items(), key=lambda x: -x[1])[:top_n]
    words_to_plot = [w for w, f in most_common]
    idxs = [i for i, w in itos.items() if w in words_to_plot]

    emb_weights = model.embedding.weight.detach().cpu().numpy()
    selected = emb_weights[idxs]
    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(idxs)//3)), random_state=seed)
    coords = tsne.fit_transform(selected)
    df = {
        'x': coords[:,0],
        'y': coords[:,1],
        'word': [itos[i] for i in idxs],
        'freq': [word_freq[itos[i]] for i in idxs]
    }
    fig = px.scatter(
        df, x='x', y='y', hover_name='word', text='word',
        color='freq', color_continuous_scale='Viridis',
        title=f"t-SNE of Top {top_n} Frequent Word Embeddings",
        width=900, height=700,
        template='plotly_white'
    )
    # Update marker and text properties
    fig.update_traces(
        marker=dict(size=12, opacity=0.7),
        selector=dict(mode='markers+text'),
        textfont=dict(size=12, color='black'),
        textposition='top center'
    )
    # Update layout with improved readability
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='t-SNE Dimension 1',
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='t-SNE Dimension 2',
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        title=dict(
            font=dict(size=16)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def tsne_tab():
    st.header("t-SNE Embedding Visualization")

    category = st.selectbox("Category", ["Natural", "Code"])
    # Find matching checkpoint/model files
    model_files = sorted(glob.glob(f"models/*_{category.lower()}.pth"))
    model_file = st.selectbox("Select model checkpoint", model_files)

    # Load vocab and words
    stoi_file = f"models/stoi_{category.lower()}.pkl"
    itos_file = f"models/itos_{category.lower()}.pkl"
    with open(stoi_file, 'rb') as f:
        stoi = pickle.load(f)
    with open(itos_file, 'rb') as f:
        itos = pickle.load(f)

    words_file = f"models/words_{category.lower()}.pkl"
    try:
        with open(words_file, 'rb') as f:
            words = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Word list file '{words_file}' does not exist. Please create it from your original corpus!")
        st.stop()
    word_freq = Counter(words)

    checkpoint = torch.load(model_file, map_location='cpu')
    if 'hyperparams' in checkpoint:
        hp = checkpoint['hyperparams']
    else:
        hp = {
            'emb_dim': checkpoint.get('emb_dim'),
            'hidden_dim': checkpoint.get('hidden_dims', [1024])[0],
            'num_hidden_layers': checkpoint.get('num_layers'),
            'activation': checkpoint.get('activation'),
            'context_length': checkpoint.get('context_length', 3)
        }
    from model_utils import NextWordMLP
    model = NextWordMLP(
        vocab_size=len(stoi),
        block_size=hp['context_length'],
        embedding_dim=hp['emb_dim'],
        hidden_dim=hp['hidden_dim'],
        num_hidden_layers=hp['num_hidden_layers'],
        activation=hp['activation']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    top_n = st.slider("Number of frequent words (Top-N)", 50, 500, 200, step=10)
    perplexity = st.slider("t-SNE Perplexity", 5, 50, 30, step=1)
    seed = st.slider("Random Seed", 1, 100, 42, step=1)

    if st.button("Plot t-SNE Embeddings"):
        plot_embedding_tsne_streamlit(model, itos, word_freq, top_n, perplexity, seed)
