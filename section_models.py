import streamlit as st

def show():
    dot = r'''
    digraph ML_3Columns {
      rankdir=TB; // sơ đồ cây từ trên xuống
      nodesep=0.5; ranksep=0.8;
      splines=false;

      // ===== Cài đặt chung =====
      graph [bgcolor="white"];
      node  [shape=box, style="rounded", color="#333333", fontname="Arial", fontsize=12];
      edge  [color="#999999", arrowsize=0.6];

      // ===== Tiêu đề chính trong khung có màu =====
      title [label="Machine Learning Model Selection", shape=box, style="rounded,filled", 
             fillcolor="#e6f0ff", color="#4a7bd1", fontname="Arial Bold", fontsize=14];

      // ===== Cụm 1: Regression based algorithms =====
      subgraph cluster_reg {
        label="Regression based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        
        reg1 [label="Adaboost"];
        reg2 [label="Catboost"];
        reg3 [label="XGboost"];

        reg1 -> reg2 -> reg3 [style=invis];
      }

      // ===== Cụm 2: Tree based algorithms =====
      subgraph cluster_tree {
        label="Tree based algorithms";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        
        tree1 [label="Extra tree"];
        tree2 [label="Random forest"];

        tree1 -> tree2 [style=invis];
      }

      // ===== Cụm 3: Neural based algorithm =====
      subgraph cluster_nn {
        label="Neural based algorithm";
        labelloc=t; fontsize=12; fontname="Arial";
        style="rounded,dashed"; color="#777777";
        
        nn1 [label="Extra tree"];
      }

      // ===== Liên kết sơ đồ cây: từ tiêu đề xuống 3 cụm =====
      title -> reg1 [color="#4a7bd1", penwidth=1.2];
      title -> tree1 [color="#4a7bd1", penwidth=1.2];
      title -> nn1 [color="#4a7bd1", penwidth=1.2];

      // ===== Căn hàng trái =====
      {rank=same; reg1; tree1; nn1;}
    }
    '''
    st.graphviz_chart(dot)
