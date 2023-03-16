import matplotlib.pyplot as plt
import networkx as nx
from example_pipelines.healthcare import custom_monkeypatching
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components

from mlwhatif import PipelineAnalyzer
from mlwhatif.visualisation._visualisation import get_original_simple_dag


def analyze_pipeline(pipeline_filename, *_what_if_analyses, add_monkey_patching=False):
    builder = PipelineAnalyzer.on_pipeline_from_py_file(pipeline_filename)

    for analysis in _what_if_analyses:
        builder = builder.add_what_if_analysis(analysis)

    builder = builder.set_intermediate_dag_storing(True)
    builder = builder.add_custom_monkey_patching_modules([custom_monkeypatching])

    if add_monkey_patching:
        # TODO: add monkey patching only when necessary?
        pass

    analysis_result = builder.execute()

    return analysis_result


def get_report(result, what_if_analysis):
    report = result.analysis_to_result_reports[what_if_analysis]
    return report


def render_graph1(graph: nx.classes.digraph.DiGraph):
    G = get_original_simple_dag(graph)
    fig, _ = plt.subplots()
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True)
    st.pyplot(fig)


def render_graph2(graph: nx.classes.digraph.DiGraph):
    G = get_original_simple_dag(graph)

    # nt = Network("500px", "500px", notebook=True, heading="")
    nt = Network()
    nt.from_nx(G)
    nt.show("graph.html")

    with open("graph.html", "r", encoding="utf-8") as html_file:
        source_code = html_file.read() 
    # components.html(source_code, height=1200, width=1000)
    components.html(source_code, height=600)
