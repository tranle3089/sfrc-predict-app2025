import streamlit as st

def show():
    dot = r'''
    digraph ML_3Columns {
      rankdir=LR;
      nodesep=0.8; ranksep=0.9;
      splines=false;

      // Kiểu chung, tối giản
      graph [bgcolor="white"];
      node  [shape=box, style="rounded", color="#333333", fontname="Arial", fontsize=12];
      edge  [color="#999999", arrowsize=0.6];

      // ===== Regression based algorithms =====
      subgraph cluster_reg {
        label="Regression based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        reg1 [label="Adaboost"];
        reg2 [label="Catboost"];
        reg3 [label="XGboost"];
        reg1 -> reg2 -> reg3 [style=invis]; // sắp dọc
      }

      // ===== Tree based algorithms =====
      subgraph cluster_tree {
        label="Tree based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        tree1 [label="Extra tree"];
        tree2 [label="Random forest"];
        tree1 -> tree2 [style=invis];
      }

      // ===== Neural based algorithm =====
      subgraph cluster_nn {
        label="Neural based algorithm";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        nn1 [label="Extra tree"];
      }

      // ===== Sắp xếp ba cột =====
      {rank=same; cluster_reg; cluster_tree; cluster_nn;}
    }
    '''
    st.graphviz_chart(dot)
