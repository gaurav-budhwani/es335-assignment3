import streamlit as st
from plot_utils import plot_loss_curves, available_model_files

def loss_plot_tab():
    st.header("Loss/Accuracy Curves")

    modeldir = "models"
    files = available_model_files(modeldir)
    selected = st.multiselect("Select model checkpoint(s) to plot", files, default=files[:1])
    show_train = st.checkbox("Show Train Loss", value=True)
    show_val = st.checkbox("Show Validation Loss", value=True)
    show_acc = st.checkbox("Show Validation Accuracy", value=True)

    if selected:
        plot_loss_curves(selected, show_train, show_val, show_acc)
    else:
        st.info("Select at least one model.")
