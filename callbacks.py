import matplotlib.pyplot as plt
import networkx as nx
import pandas
import streamlit as st
import streamlit.components.v1 as components
from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
from mlwhatif.execution._patches import AppendNodeAfterOperator, DataProjection, OperatorReplacement, DataTransformer, \
    DataFiltering
from mlwhatif.visualisation._visualisation import get_original_simple_dag, get_colored_simple_dags, \
    get_final_optimized_combined_colored_simple_dag
from pyvis.network import Network
from st_cytoscape import cytoscape


def analyze_pipeline(dag_extraction_result, *_what_if_analyses, add_monkey_patching=False):
    builder = PipelineAnalyzer.on_previously_extracted_pipeline(dag_extraction_result)

    for analysis in _what_if_analyses:
        builder = builder.add_what_if_analysis(analysis)

    builder = builder.set_intermediate_dag_storing(True)
    builder = builder.add_custom_monkey_patching_modules([custom_monkeypatching])

    if add_monkey_patching:
        builder = builder.add_custom_monkey_patching_module(custom_monkeypatching)

    analysis_result = builder.execute()

    return analysis_result


def estimate_pipeline_analysis(dag_extraction_result, *_what_if_analyses, add_monkey_patching=False):
    builder = PipelineAnalyzer.on_previously_extracted_pipeline(dag_extraction_result)

    for analysis in _what_if_analyses:
        builder = builder.add_what_if_analysis(analysis)

    builder = builder.set_intermediate_dag_storing(True)
    builder = builder.add_custom_monkey_patching_modules([custom_monkeypatching])

    if add_monkey_patching:
        builder = builder.add_custom_monkey_patching_module(custom_monkeypatching)

    estimation_result = builder.estimate()

    return estimation_result


def scan_pipeline(pipeline_source_code, add_monkey_patching=False):
    builder = PipelineAnalyzer.on_pipeline_from_string(pipeline_source_code)

    builder = builder.set_intermediate_dag_storing(True)
    builder = builder.add_custom_monkey_patching_modules([custom_monkeypatching])

    if add_monkey_patching:
        builder = builder.add_custom_monkey_patching_module(custom_monkeypatching)

    result = builder.execute()
    runtime_orig = result.runtime_info.original_pipeline_estimated
    dag_extraction_info = result.dag_extraction_info

    return runtime_orig, dag_extraction_info


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

    # def get_new_node_label(node):
    #     label = cleandoc(f"{node}: {nx.get_node_attributes(G, 'operator_name')[node]}")
    #     return label
    #
    # # noinspection PyTypeChecker
    # G = nx.relabel_nodes(G, get_new_node_label)
    cytoscape_data = nx.cytoscape_data(graph)["elements"]

    stylesheet = [{
        'selector': 'node',
        'css': {
            'content': 'data(operator_name)',
            'text-valign': 'center',
            'color': 'white',
            'text-outline-width': 2,
            'text-outline-color': 'black',
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
    # elements = cytoscape_data["nodes"] + cytoscape_data["edges"]
    # print(elements)
    # cytoscapeobj = ipycytoscape.CytoscapeWidget()
    # cytoscapeobj.graph.add_graph_from_networkx(plan, directed=True)

    return cytoscape_data, stylesheet
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

main_description = {
    'Original': "The plan extracted from the original pipeline.",
    'Variants': "The what-if analyses use patches to create each variant the users wants to test.",
    'Shared': "There is already some work that could be shared between the variants.",
    'FRP': "Filter Removal Push-Up, the first optimization rule. If we have filters that get removed in some, but"
           " not all variants, pushing the filter up in all variants where they are present might be beneficial.",
    'PP': "Projection Push-Up, the second optimization rule.",
    'FP': "Filter Push-Up, the third optimization rule.",
    'UDF': "UDF Split-Reuse, the fourth optimization rule. If we apply expensive UDFs repeatedly to large fractions of "
           "the input data, for example to test the "
           "robustness against data corruptions, applying it just once to all data and then sampling from the result"
           " might be beneficial.",
    'Merging': "Finally, we merge all variants into one combined execution plan, using common subexpression "
               "elimination.",
    'Done': "The combined plan is ready for execution now.",
}


def get_dags(name):
    if name == "Original":
        if st.session_state.DAG_EXTRACTION_RESULT:
            return [get_original_simple_dag(st.session_state.DAG_EXTRACTION_RESULT.original_dag)]
    elif name == "Variants":
        if st.session_state.ANALYSIS_RESULT:
            return get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"],
                with_reuse_coloring=False)
    elif name == "Shared":
        if st.session_state.ANALYSIS_RESULT:
            return get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"],
                with_reuse_coloring=True)
    elif name == "FRP":
        if st.session_state.ANALYSIS_RESULT:
            return get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['1-optimize_dag_2_OperatorDeletionFilterPushUp'],
                with_reuse_coloring=True)
    elif name == "PP":
        if st.session_state.ANALYSIS_RESULT:
            return get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['2-optimize_patches_0_SimpleProjectionPushUp'],
                with_reuse_coloring=True)
    elif name == "FP":
        if st.session_state.ANALYSIS_RESULT:
            return get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['3-optimize_patches_1_SimpleFilterAdditionPushUp'],
                with_reuse_coloring=True)
    elif name == "UDF":
        if st.session_state.ANALYSIS_RESULT:
            return get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['4-optimize_patches_3_UdfSplitAndReuse'],
                with_reuse_coloring=True)
    elif name in {"Merging", "Done"}:
        if st.session_state.ANALYSIS_RESULT:
            return [get_final_optimized_combined_colored_simple_dag(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["4-optimize_patches_3_UdfSplitAndReuse"])]


def render_cytoscape(dag, key, height):
    if dag:
        cytoscape_data, stylesheet = render_graph3(dag)
        return cytoscape(cytoscape_data, stylesheet, layout={"name": "dagre"}, key=key, height=height)


def render_dag_slot(name, dag, key,  height='300px'):
    if name == "Original":
        if st.session_state.DAG_EXTRACTION_RESULT:
            render_cytoscape(dag, key, height)
            st.write("Here is a description of what the original DAG is")
    elif name == "Merging":
        if st.session_state.ANALYSIS_RESULT:
            render_cytoscape(dag, key, height)
        st.write("Here is a description of what the merged DAG is")
    else:
        if st.session_state.ANALYSIS_RESULT:
            render_cytoscape(dag, key, height)
        st.write(f"Here is a description of what the {name} is")


def render_full_size_dag(stage_name):
    st.markdown(main_description[stage_name])
    if st.session_state.DAG_EXTRACTION_RESULT:
        dag = get_dags(stage_name)[0]
        render_dag_slot(stage_name, dag, f"full-size-{stage_name}", height='800px')


def render_dag_comparison(before, after):
    st.markdown(main_description[after])
    dags_before = get_dags(before)
    dags_after = get_dags(after)
    if len(dags_before) == 1:
        dags_before = [orig_dag for orig_dag in dags_before for _ in range(len(dags_after))]
    if len(dags_after) == 1:
        dags_after = [merged_dag for merged_dag in dags_after for _ in range(len(dags_before))]

    patches = st.session_state.ANALYSIS_RESULT.what_if_patches
    for variant_left, variant_right in zip(range(0, len(patches), 2), range(1, len(patches) + 1, 2)):
        left, right = st.columns(2)
        with left:
            st.markdown(f"### Variant {variant_left}")
            render_patches(patches[variant_left])
        if variant_right < len(patches):
            with right:
                st.markdown(f"### Variant {variant_right}")
                render_patches(patches[variant_right])
        columns = st.columns(4)
        with st.container():
            render_variant_slot(after, before, dags_after, dags_before, variant_left, columns[0:2])
        if variant_right < len(patches):
            with st.container():
                render_variant_slot(after, before, dags_after, dags_before, variant_right, columns[2:4])
        st.markdown("""---""")


def render_variant_slot(after, before, dags_after, dags_before, variant_index, columns):
    with st.container():
        left, right = columns
        with left:
            with st.container():
                st.write("#### before")
                render_dag_slot(before, dags_before[variant_index], f"before-{before}-{variant_index}")

        with right:
            with st.container():
                st.write("#### after")
                render_dag_slot(after, dags_after[variant_index], f"after-{after}-{variant_index}")


def render_patches(variant_patches, key=None):
    patch_names = []
    patch_analyses = []
    patch_descriptions = []
    for patch in variant_patches:
        if type(patch) != AppendNodeAfterOperator:
            patch_analyses.append(type(patch.analysis).__name__)
            if type(patch) == DataProjection:
                patch_names.append(type(patch).__name__)
                patch_descriptions.append(patch.projection_operator.details.description)
            elif type(patch) == OperatorReplacement:
                patch_names.append(type(patch).__name__)
                description = f"Replace '{patch.operator_to_replace.details.description}' with " \
                              f"'{patch.replacement_operator.details.description}'"
                patch_descriptions.append(description)
            elif type(patch) == DataTransformer:
                patch_names.append("DataEstimator")
                description = f"{patch.fit_transform_operator.details.description}"
                patch_descriptions.append(description)
            elif type(patch) == DataFiltering:
                patch_names.append("DataFilter")
                description = f"{patch.filter_operator.details.description}"
                if patch.train_not_test:
                    description += ' on train side'
                else:
                    description += ' on test side'
                patch_descriptions.append(description)
            # TODO: Model patches
            else:
                patch_names.append(type(patch).__name__)
                patch_descriptions.append("")
    variant_df = pandas.DataFrame({'Patch Type': patch_names, 'Analysis': patch_analyses,
                                   'Description': patch_descriptions})
    st.table(variant_df)
