import matplotlib.pyplot as plt
import torch
import numpy as np
import streamlit as st
import os

def plot_loss_curves(model_files, show_train=True, show_val=True, show_acc=True, title_prefix=""):
    """
    Plots loss/accuracy curves (multiple model files allowed).
    Args:
        model_files: list of checkpoint .pth paths
        show_train: show train loss
        show_val: show val loss
        show_acc: show val accuracy
        title_prefix: string or "" for plot titles
    """
    plt.figure(figsize=(10, 4))

    for model_file in model_files:
        checkpoint = torch.load(model_file, map_location='cpu')
        label = os.path.basename(model_file).replace('.pth','')
        if 'train_losses' in checkpoint:
            if show_train:
                plt.plot(checkpoint['train_losses'], label=f"{label} Train Loss")
            if show_val:
                plt.plot(checkpoint['val_losses'], label=f"{label} Val Loss")
            if show_acc and 'val_accuracies' in checkpoint:
                plt.plot(checkpoint['val_accuracies'], label=f"{label} Val Acc")

    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.title(f"{title_prefix} Loss/Accuracy Curves")
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_individual_loss_curves(model_file):
    """
    Utility for interactive per-model file.
    """
    checkpoint = torch.load(model_file, map_location='cpu')
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if 'train_losses' in checkpoint:
        ax.plot(checkpoint['train_losses'], label='Train Loss')
    if 'val_losses' in checkpoint:
        ax.plot(checkpoint['val_losses'], label='Validation Loss')
    if 'val_accuracies' in checkpoint:
        ax.plot(checkpoint['val_accuracies'], label='Validation Accuracy')
    ax.set_xlabel("Epoch")
    ax.set_title(os.path.basename(model_file))
    ax.legend(), ax.grid(True)
    st.pyplot(fig)

def available_model_files(modeldir, pattern="*.pth"):
    import glob
    return sorted(glob.glob(os.path.join(modeldir, pattern)))
