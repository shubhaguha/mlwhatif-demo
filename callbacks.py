from inspect import cleandoc

import matplotlib.pyplot as plt
import networkx as nx
from example_pipelines.healthcare import custom_monkeypatching
from networkx.drawing.nx_agraph import to_agraph, graphviz_layout
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components

from mlwhatif import PipelineAnalyzer
from mlwhatif.visualisation._visualisation import get_original_simple_dag
from streamlit_cytoscapejs import st_cytoscapejs


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

def render_graph3(graph: nx.classes.digraph.DiGraph):
    G = get_original_simple_dag(graph)

    # def get_new_node_label(node):
    #     label = cleandoc(f"{node}: {nx.get_node_attributes(G, 'operator_name')[node]}")
    #     return label
    #
    # # noinspection PyTypeChecker
    # G = nx.relabel_nodes(G, get_new_node_label)
    cytoscape_data = nx.cytoscape_data(G)["elements"]

    stylesheet = [{
        'selector': 'node',
        'css': {
            'content': 'data(operator_name)',
            'text-valign': 'center',
            'color': 'white',
            'text-outline-width': 2,
            'text-outline-color': 'data(fontcolor)',
            'background-color': 'data(fillcolor)'
        }
    },
        {
            'selector': ':selected',
            'css': {
                'background-color': 'black',
                'line-color': 'black',
                'target-arrow-color': 'black',
                'source-arrow-color': 'black',
                'text-outline-color': 'black'
            }
        },
        {
            "selector": "edge",
            "style": {
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle'
            }
        },
    ]
    elements = cytoscape_data["nodes"] + cytoscape_data["edges"]
    print(elements)
    # cytoscapeobj = ipycytoscape.CytoscapeWidget()
    # cytoscapeobj.graph.add_graph_from_networkx(plan, directed=True)

    # klay
    # Z

    clicked_elements = st_cytoscapejs(elements=elements, stylesheet=stylesheet)
    # nt = Network()
    # nt.from_nx(G)
    # nt.show("graph.html")
    #
    # with open("graph.html", "r", encoding="utf-8") as html_file:
    #     source_code = html_file.read()
    #     # components.html(source_code, height=1200, width=1000)
    # components.html(source_code, height=600)


    # fig, _ = plt.subplots()
    # nx.draw(G, pos, with_labels=True)
    # st.pyplot(fig)

    # agraph = to_agraph(G)
    # agraph.layout(prog='dot')
    # agraph.draw("graph.dot")
    #
    # with open("graph.dot", "r", encoding="utf-8") as dot_file:
    #     dot_content = dot_file.read()
    # st.graphviz_chart(dot_content, use_container_width=True)

    # nt = Network()
    # nt.from_DOT(agraph)
    # nt.show("graph.html")
    #
    # with open("graph.html", "r", encoding="utf-8") as html_file:
    #     source_code = html_file.read()
    #     # components.html(source_code, height=1200, width=1000)
    # components.html(source_code, height=600)

    # # Now we need to somehow visualize this
    # agraph.draw("graph.html")
    # with open("graph.html", "r", encoding="utf-8") as html_file:
    #     source_code = html_file.read()
    # components.html(source_code, height=600)
