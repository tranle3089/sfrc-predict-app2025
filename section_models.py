import streamlit as st

def show():
    dot = r'''
    digraph ML_Selection_Styled {
      rankdir=TB;
      splines=true;
      nodesep=0.35; 
      ranksep=0.7;

      // ===== Global styles =====
      graph [fontname="Inter,Arial", fontsize=12, color="white", bgcolor="white", pad=0.2];
      node  [shape=box, style="rounded,filled", fontname="Inter,Arial", fontsize=12, margin="0.12,0.07"];
      edge  [color="#94a3b8", arrowsize=0.7, penwidth=1.6];

      // ===== Cluster 1: Regression based algorithms (warm) =====
      subgraph cluster_reg {
        label="  Regression based algorithms  ";
        labelloc=t; fontsize=12; fontcolor="#7a5b00";
        style="rounded,filled,dashed"; color="#e0b857"; penwidth=1.2;
        fillcolor="#fff7e6:#fffdf6"; gradientangle=90;

        reg1 [label="Adaboost",      fillcolor="#fff1d6", color="#e0b857"];
        reg2 [label="Catboost",      fillcolor="#fff1d6", color="#e0b857"];
        reg3 [label="XGboost",       fillcolor="#fff1d6", color="#e0b857"];

        reg1 -> reg2 -> reg3 [color="#e0b857", penwidth=1.1, arrowsize=0.6];
      }

      // ===== Cluster 2: Tree based algorithms (green) =====
      subgraph cluster_tree {
        label="  Tree based algorithms  ";
        labelloc=t; fontsize=12; fontcolor="#10563f";
        style="rounded,filled,dashed"; color="#2bb673"; penwidth=1.2;
        fillcolor="#e8fff5:#f7fffb"; gradientangle=90;

        tree1 [label="Extra tree",    fillcolor="#e9fff4", color="#2bb673"];
        tree2 [label="Random forest", fillcolor="#e9fff4", color="#2bb673"];

        tree1 -> tree2 [color="#2bb673", penwidth=1.1, arrowsize=0.6];
      }

      // ===== Cluster 3: Neural based algorithm (violet) =====
      subgraph cluster_nn {
        label="  Neural based algorithm  ";
        labelloc=t; fontsize=12; fontcolor="#4b3fbf";
        style="rounded,filled,dashed"; color="#7b61ff"; penwidth=1.2;
        fillcolor="#f3f0ff:#faf8ff"; gradientangle=90;

        nn1  [label="Extra tree", fillcolor="#f3f0ff", color="#7b61ff"];
      }

      // ===== Flow between groups (subtle arrows) =====
      reg3 -> tree1  [color="#94a3b8", penwidth=2.0];
      tree2 -> nn1   [color="#94a3b8", penwidth=2.0];

      // Keep vertical ordering of clusters
      {rank=same; }
    }
    '''
    st.graphviz_chart(dot)
