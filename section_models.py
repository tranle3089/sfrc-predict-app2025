import streamlit as st

def show():
    dot = r'''
    digraph ML_3Columns {
      rankdir=LR;  // từ trái sang phải
      nodesep=0.8; ranksep=1.0;
      splines=false;

      // ===== Cài đặt chung =====
      graph [bgcolor="white"];
      node  [shape=box, style="rounded", color="#333333", fontname="Arial", fontsize=12];
      edge  [color="#aaaaaa", arrowsize=0.6];

      // ===== Tiêu đề chính có khung màu =====
      title [label="Machine Learning Model Selection",
             shape=box, style="rounded,filled", fillcolor="#e6f0ff",
             color="#4a7bd1", fontname="Arial Bold", fontsize=14];

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

      // ===== Bố cục và liên kết =====
      title -> reg1 [color="#4a7bd1", penwidth=1.3];
      title -> tree1 [color="#4a7bd1", penwidth=1.3];
      title -> nn1 [color="#4a7bd1", penwidth=1.3];

      // Ba cột song song
      {rank=same; reg1; tree1; nn1;}
    }
    '''
    st.graphviz_chart(dot)
