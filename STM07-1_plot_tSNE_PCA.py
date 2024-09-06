#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:22:06 2024

"""

import pandas as pd
import numpy as np
from joblib import load
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
speech_corp_df2 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv',index_col=0)
music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv',index_col=0)
df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv',index_col=0)

all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)

# %% plot PCA

# pipeline = load('model/allSTM_pca-pipeline_2024-04-22_08-43.joblib')
# pca = pipeline['incrementalpca']


# plt.plot(list(range(1,len(pca.explained_variance_ratio_[:5000])+1)), np.cumsum(pca.explained_variance_ratio_[:5000]))
# plt.show()

# %% load t-SNE and preprocessing

tsne_folder = 'model/STM/tsne/perplexity120_2024-08-15_17-55/'


tsne = load(tsne_folder+'tsne_model.joblib')
df_tsne = pd.DataFrame()
df_tsne['tSNE-1'] = tsne.embedding_[:,0]
df_tsne['tSNE-2'] = tsne.embedding_[:,1]
df_tsne['category'] = pd.concat([all_corp_df['corpus_type'], 
                                 pd.Series(['env_aug'] * (len(tsne.embedding_) - len(all_corp_df)))], ignore_index=True)

# %% scatter plot
with plt.style.context('seaborn-v0_8-notebook'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    
    kde = sns.scatterplot(
        x="tSNE-1", 
        y="tSNE-2",
        data=df_tsne,
        ax=ax,
        hue="category", 
        alpha=0.1
    )
    ax.set_xlim(-75, 75)
    ax.set_ylim(-75, 75)
    plt.show()

# %% plot kernal density
df_tsne_filtered = df_tsne[(df_tsne['tSNE-1'].between(-70, 70)) & (df_tsne['tSNE-2'].between(-70,70))]

with plt.style.context('seaborn-v0_8-poster'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(1, 1, 1)
    
    kde = sns.kdeplot(
        x="tSNE-1", 
        y="tSNE-2",
        data=df_tsne_filtered,
        ax=ax,
        hue="category", 
        fill=False,
        levels=10,
        alpha=0.5,
        threshold=0,
        common_norm=False,
        legend=True, 
        # bw_adjust=0.1,
    )
    kde.legend_.set_title(None)

    # ax.set_xlim(-75, 75)
    # ax.set_ylim(-75, 75)
    
    # ax.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # # Get categories from hue variable
    # categories = df_tsne_filtered["category"].unique()

    # # Create legend handles (empty for kdeplot)
    # handles = [matplotlib.patches.Patch(color='none')] * len(categories)
    # if any(handles):  # Check if any element in handles is True
    #     ax.legend(handles=handles, labels=categories, title="Category")

    # # Set labels manually
    # ax.legend(handles=handles, labels=categories, title="Category")

    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    # # Get categories from hue variable
    # categories = df_tsne_filtered["category"].unique()
    
    # # Set labels manually
    # ax.legend(labels=categories, title="")
    
    plt.tight_layout()
    plt.show()
    fig.savefig(tsne_folder+'kdeplot_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'.png')
    