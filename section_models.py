import streamlit as st

def show():
    dot = r'''
    digraph ML_Selection_Custom {
      rankdir=TB;
      nodesep=0.35; ranksep=0.6;
      splines=true;

      // Default node + edge styles
      node [shape=box, style="rounded,filled", fillcolor="white", color="#000000",
            fontname="Arial", fontsize=12];
      edge [style=invis];  // chỉ dùng để sắp xếp dọc, không vẽ mũi tên

      // ===== Cluster 1: Regression based algorithms =====
      subgraph cluster_reg {
        label="Regression based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#7a7a7a";
        bgcolor="#eef6ea"; // nền xanh nhạt như hình

        reg1 [label="Adaboost"];
        reg2 [label="Catboost"];
        reg3 [label="XGboost"];

        // sắp xếp theo cột
        reg1 -> reg2 -> reg3;
      }

      // ===== Cluster 2: Tree based algorithms =====
      subgraph cluster_tree {
        label="Tree based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#7a7a7a";
        bgcolor="#eef6ea";

        tree1 [label="Extra tree"];
        tree2 [label="Random forest"];

        tree1 -> tree2;
      }

      // ===== Cluster 3: Neural based algorithm =====
      subgraph cluster_nn {
        label="Neural based algorithm";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#7a7a7a";
        bgcolor="#eef6ea";

        nn1 [label="Extra tree"];
      }

      // giữ thứ tự từ trên xuống dưới giữa các cụm
      {rank=same; } // placeholder
    }
    '''
    st.graphviz_chart(dot)
