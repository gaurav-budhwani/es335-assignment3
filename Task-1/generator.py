import streamlit as st
import pickle
import glob
import re
from model_utils import (
    load_checkpoint, NextWordMLP, prepare_context, generate_next_k_words, scan_option_vals
)

def generator_tab():
    category = st.sidebar.selectbox('Category', ['Natural', 'Code'])

    model_files = sorted(glob.glob(f"models/*_{category.lower()}.pth"))
    embdims = scan_option_vals(model_files, r'emb(\d+)')
    acts = scan_option_vals(model_files, r'_(relu|tanh)')
    numlayers = scan_option_vals(model_files, r'layers(\d+)')

    emb_dim = int(st.sidebar.selectbox('Embedding size', embdims))
    activation = st.sidebar.selectbox('Activation function', acts)
    num_hidden_layers = int(st.sidebar.selectbox('No. of hidden layers', numlayers))
    temperature = st.sidebar.slider('Temperature', 0.1, 2.0, 1.0, step=0.05)
    seed = st.sidebar.slider("Random Seed", 1, 999, 42, 1)
    handle_oov = st.sidebar.radio(
        "How to handle unknown words?",
        ["Skip word", "Mask as <UNK>", "Find closest (embedding similarity)"]
    )

    pattern = fr'emb{emb_dim}_hidden\d+_layers{num_hidden_layers}_{activation}_{category.lower()}\.pth'
    model_file = next((f for f in model_files if re.search(pattern, f)), None)
    if not model_file:
        st.warning('Model with chosen hyperparameters not found!')
        return

    # Load vocab
    stoi_file = f"models/stoi_{category.lower()}.pkl"
    itos_file = f"models/itos_{category.lower()}.pkl"
    with open(stoi_file, 'rb') as f:
        stoi = pickle.load(f)
    with open(itos_file, 'rb') as f:
        itos = pickle.load(f)

    # Load model/checkpoint
    checkpoint = load_checkpoint(model_file)
    if 'hyperparams' in checkpoint:
        hp = checkpoint['hyperparams']
    else:
        hp = {
            'emb_dim': checkpoint.get('emb_dim', emb_dim),
            'hidden_dim': checkpoint.get('hidden_dims', [1024])[0],
            'num_hidden_layers': checkpoint.get('num_layers', num_hidden_layers),
            'activation': checkpoint.get('activation', activation),
            'context_length': checkpoint.get('context_length', 3)
        }
    model = NextWordMLP(
        vocab_size=len(stoi),
        block_size=hp['context_length'],
        embedding_dim=hp['emb_dim'],
        hidden_dim=hp['hidden_dim'],
        num_hidden_layers=hp['num_hidden_layers'],
        activation=hp['activation']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    context_input = st.text_input('Enter input context:', '')
    context_words = context_input.strip().split()
    k = st.slider("How many next words?", 1, 10, 5)

    if st.button("Generate Next k Words"):
        clean_context, affected = prepare_context(context_words, handle_oov, stoi, model, itos)
        if affected:
            st.warning(f"Words handled specially: {', '.join(affected)}")
        generated_seq = generate_next_k_words(
            model, clean_context, stoi, itos,
            hp['context_length'], k, temperature, seed=seed
        )
        st.write("Generated sequence:")
        st.write(" ".join(generated_seq))
