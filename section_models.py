import streamlit as st

def show():
    dot = r'''
    digraph ML_3Columns {
      rankdir=LR;
      nodesep=0.8; ranksep=1.0;
      splines=false;

      // ===== Cài đặt chung =====
      graph [bgcolor="white", label="Machine Learning Model Selection", labelloc=t, fontsize=16, fontname="Arial Bold"];
      node  [shape=box, style="rounded", color="#333333", fontname="Arial", fontsize=12];
      edge  [color="#aaaaaa", arrowsize=0.6];

      // ===== 1️⃣ Regression based algorithms (nhiều nhất) =====
      subgraph cluster_reg {
        label="Regression based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        
        reg1 [label="Adaboost"];
        reg2 [label="Catboost"];
        reg3 [label="XGboost"];

        reg1 -> reg2 -> reg3 [style=invis];
      }

      // ===== 2️⃣ Tree based algorithms (trung bình) =====
      subgraph cluster_tree {
        label="Tree based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        
        tree1 [label="Extra tree"];
        tree2 [label="Random forest"];

        tree1 -> tree2 [style=invis];
      }

      // ===== 3️⃣ Neural based algorithm (ít nhất) =====
      subgraph cluster_nn {
        label="Neural based algorithm";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        
        nn1 [label="Extra tree"];
      }

      // ===== Bố cục trái → phải (nhiều → ít) =====
      {rank=same; cluster_reg; cluster_tree; cluster_nn;}

      // ===== Mũi tên luồng giữa các nhóm =====
      reg3 -> tree1 [color="#999999", penwidth=1.3];
      tree2 -> nn1 [color="#999999", penwidth=1.3];
    }
    '''
    st.graphviz_chart(dot)
