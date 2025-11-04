import streamlit as st
from generator import generator_tab
from loss_plots import loss_plot_tab
from tsne_visualizer import tsne_tab

st.set_page_config(page_title="Next Word Project", layout="wide")
st.title("Next-Word Deep Learning Playground")

tab1, tab2, tab3 = st.tabs(["Generator", "Loss Curves", "TSNE Embeddings"])

with tab1:
    generator_tab()
with tab2:
    loss_plot_tab()
with tab3:
    tsne_tab()
