import streamlit as st

def show():
    dot = r'''
    digraph G {
      rankdir=TB;
      splines=true;
      nodesep=0.35; ranksep=0.5;

      node [shape=box, style="rounded,filled", fillcolor="#f8f9fa", color="#9aa0a6", fontname="Arial", fontsize=12];
      edge [color="#cfd2d6"];

      root [label="Materials and Methods", fillcolor="#e8f0fe"];

      root -> ml;
      root -> opt;
      root -> dev;

      ml  [label="Machine Learning\nModel Selection"];
      opt [label="Optimization Using\nTPE Method"];
      dev [label="Predictive Model\nDevelopment"];

      ml  -> reg;
      ml  -> treealg;
      ml  -> nn;

      opt -> bo;
      opt -> tpe;

      dev -> train;
      dev -> metrics;

      reg     [label="Regression Algorithms"];
      treealg [label="Tree-Based Algorithms"];
      nn      [label="Neural Network Algorithms"];

      bo  [label="Bayesian Optimization"];
      tpe [label="Tree-Structured Parzen\nEstimator (TPE)"];

      train   [label="Training & Hyperparameter Tuning"];
      metrics [label="Performance Evaluation Metrics"];
    }
    '''
    st.graphviz_chart(dot)
