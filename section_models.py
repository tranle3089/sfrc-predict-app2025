import streamlit as st
def show():
    dot = r'''
    digraph ML_3Columns {
      rankdir=LR;  // Bố cục từ trái sang phải
      nodesep=0.8; 
      ranksep=1.0;
      splines=false;

      // ===== CÀI ĐẶT CHUNG =====
      graph [
        bgcolor="lightblue",
        label="Machine Learning Model Selection",
        labelloc=t,
        fontsize=18,
        fontname="Times New Roman Bold"
      ];

      node [
        shape=box,
        style="rounded",
        color="#333333",
        fontname="Times New Roman",
        fontsize=12
      ];

      edge [
        color="#aaaaaa",
        arrowsize=0.6
      ];

      // ===== (1) Regression based algorithms =====
      subgraph cluster_reg {
        label="Regression based algorithms";
        labelloc=t;
        fontsize=12;
        fontname="Times New Roman";
        style="rounded,dashed";
        color="#777777";

        reg1 [label="Adaboost"];
        reg2 [label="Catboost"];
        reg3 [label="XGboost"];

        reg1 -> reg2 -> reg3 [style=invis];
      }

      // ===== (2) Tree based algorithms =====
      subgraph cluster_tree {
        label="Tree based algorithms";
        labelloc=t;
        fontsize=12;
        fontname="Times New Roman";
        style="rounded,dashed";
        color="#777777";

        tree1 [label="Extra tree"];
        tree2 [label="Random forest"];

        tree1 -> tree2 [style=invis];
      }

      // ===== (3) Neural based algorithm =====
      subgraph cluster_nn {
        label="Neural based algorithm";
        labelloc=t;
        fontsize=12;
        fontname="Times New Roman";
        style="rounded,dashed";
        color="#777777";

        nn1 [label="Extra tree"];
      }
    }
    '''
    st.graphviz_chart(dot)


