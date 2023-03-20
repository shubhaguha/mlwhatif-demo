import matplotlib.pyplot as plt
import networkx
import networkx as nx
import pandas
import streamlit as st
import streamlit.components.v1 as components
from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
from mlwhatif.execution._patches import AppendNodeAfterOperator, DataProjection, OperatorReplacement, DataTransformer, \
    DataFiltering, ModelPatch
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

    # TODO: Figure out how to make font wide to make it more readable
    stylesheet = [{
        'selector': 'node',
        'css': {
            'content': 'data(operator_name)',
            'text-valign': 'center',
            'color': 'white',
            'text-outline-width': '2',
            'text-outline-color': 'black',
            'background-color': 'data(fillcolor)',
            'border-color': 'black',
            'border-width': '2px',
        }
    },
        {
            'selector': ':selected',
            'css': {
                'background-color': 'black',
                'line-color': 'black',
                'target-arrow-color': 'black',
                'source-arrow-color': 'black',
                'text-outline-color': 'black',
                'border-color': 'black',
                'border-width': '2px',
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

    return cytoscape_data, stylesheet


main_description = {
    'Original': "The plan extracted from the original pipeline. You can click on nodes to learn more about them.",
    'Variants': "The what-if analyses use patches to create each variant the users wants to test.",
    'Shared': "There is already some work that could be shared between the variants. This shared work is highlighted "
              "in black in the plans on the right.",
    'FRP': "Filter Removal Push-Up, the first optimization rule. If we have filters that get removed in some, but"
           " not all variants, pushing the filter up in all variants where they are present might be beneficial. "
           "Work shared between multiple variants is marked in black. This optimization is specifically for Filter "
           "Removal Patches. If the optimization is worth it, according to our cost-based heuristics, then filters"
           "that are removed in some variants can be pushed up in all variants, to have more shared work across all "
           "variants. This is visible my more shared work, a.k.a. more black nodes, in the respective variants.  ",
    'PP': "Projection Push-Up, the second optimization rule. Work shared between multiple variants is marked in black. "
          "By pushing up DataProjections and DataEstimators, we can increase the amount of sharable work between the "
          "variants.",
    'FP': "Filter Push-Up, the third optimization rule. Work shared between multiple variants is marked in black. By "
          "pushing up Data Filter Patches, we can increase the amount of sharable work between the variants.",
    'UDF': "UDF Split-Reuse, the fourth optimization rule. If we apply expensive UDFs repeatedly to large fractions of "
           "the input data, for example to test the "
           "robustness against data corruptions, applying it just once to all data and then sampling from the result"
           " might be beneficial, if the UDF is applied to large enough fractions of the data. Then, the optimisation "
           "results in `corrupt x%' nodes being split up into one corrupt everything node and a second one that samples "
           "corrupted data from the first node. Work shared between multiple variants is marked in black.",
    'Merging': "Finally, we merge all variants into one combined execution plan, using common subexpression "
               "elimination. Work shared between multiple variants is marked in black.",
    'Done': "The combined plan is ready for execution now. Work shared between multiple variants is marked in black.",
}


def get_dags(name):
    if name == "Original":
        if st.session_state.DAG_EXTRACTION_RESULT:
            return ([get_original_simple_dag(st.session_state.DAG_EXTRACTION_RESULT.original_dag)],
                    [st.session_state.DAG_EXTRACTION_RESULT.original_dag])
    elif name == "Variants":
        if st.session_state.ANALYSIS_RESULT:
            return (get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"],
                with_reuse_coloring=False),
                    st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"])
    elif name == "Shared":
        if st.session_state.ANALYSIS_RESULT:
            return (get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"],
                with_reuse_coloring=True),
                    st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"])
    elif name == "FRP":
        if st.session_state.ANALYSIS_RESULT:
            return (get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['1-optimize_dag_2_OperatorDeletionFilterPushUp'],
                with_reuse_coloring=True),
                    st.session_state.ANALYSIS_RESULT.intermediate_stages[
                        '1-optimize_dag_2_OperatorDeletionFilterPushUp'])
    elif name == "PP":
        if st.session_state.ANALYSIS_RESULT:
            return (get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['2-optimize_patches_0_SimpleProjectionPushUp'],
                with_reuse_coloring=True),
                    st.session_state.ANALYSIS_RESULT.intermediate_stages['2-optimize_patches_0_SimpleProjectionPushUp'])
    elif name == "FP":
        if st.session_state.ANALYSIS_RESULT:
            return (get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['3-optimize_patches_1_SimpleFilterAdditionPushUp'],
                with_reuse_coloring=True),
                    st.session_state.ANALYSIS_RESULT.intermediate_stages[
                        '3-optimize_patches_1_SimpleFilterAdditionPushUp'])
    elif name == "UDF":
        if st.session_state.ANALYSIS_RESULT:
            return (get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages['4-optimize_patches_3_UdfSplitAndReuse'],
                with_reuse_coloring=True),
                    st.session_state.ANALYSIS_RESULT.intermediate_stages['4-optimize_patches_3_UdfSplitAndReuse'])
    elif name in {"Merging", "Done"}:
        if st.session_state.ANALYSIS_RESULT:
            return ([get_final_optimized_combined_colored_simple_dag(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["4-optimize_patches_3_UdfSplitAndReuse"])],
                    [networkx.compose_all(st.session_state.ANALYSIS_RESULT.intermediate_stages[
                                              "4-optimize_patches_3_UdfSplitAndReuse"])])


def render_cytoscape(dag, key, height):
    if dag:
        cytoscape_data, stylesheet = render_graph3(dag)
        return cytoscape(cytoscape_data, stylesheet, layout={"name": "dagre"}, key=key, height=height,
                         selection_type='single')


def render_dag_slot(name, dag, key, height='300px', description=True):
    selected = {'nodes': []}
    if name == "Original":
        if st.session_state.DAG_EXTRACTION_RESULT:
            selected = render_cytoscape(dag, key, height)
            if description is True:
                st.write("The plan extracted from the original pipeline for comparison.")
    elif name == "Merging":
        if st.session_state.ANALYSIS_RESULT:
            selected = render_cytoscape(dag, key, height)
        if description is True:
            st.write("The final combined plan for comparison.")
    else:
        if st.session_state.ANALYSIS_RESULT:
            selected = render_cytoscape(dag, key, height)
        if description is True:
            st.write(f"Here is a description of what the {name} is")
    return selected


def render_full_size_dag(stage_name):
    st.markdown(main_description[stage_name])
    if st.session_state.DAG_EXTRACTION_RESULT:
        visualization_dag, internal_dag = get_dags(stage_name)
        visualization_dag = visualization_dag[0]
        internal_dag = internal_dag[0]
        selected = render_dag_slot(stage_name, visualization_dag, f"full-size-{stage_name}", height='800px',
                                   description=False)
        if len(selected['nodes']) != 0:
            render_dag_node_details(internal_dag, selected, width=1500)
        else:
            st.markdown("Select a DAG Node for details")


def render_dag_node_details(internal_dag, selected, width=300):
    with st.container():
        selected_id = int(selected['nodes'][0])
        selected_node = [dag_node for dag_node in list(internal_dag.nodes())
                         if dag_node.node_id == selected_id]
        assert len(selected_node) == 1
        selected_node = selected_node[0]
        attribute_names = ['Operator Type', "Source Code", "Description", "Runtime", "Shape",
                           "Line Number", "Column Offset", "End Line Number", "End Column Offset",
                           ]
        if selected_node.optional_code_info is not None:
            source_code = selected_node.optional_code_info.source_code
            lineno = str(selected_node.optional_code_info.code_reference.lineno)
            col_offset = str(selected_node.optional_code_info.code_reference.col_offset)
            end_lineno = str(selected_node.optional_code_info.code_reference.end_lineno)
            end_col_offset = str(selected_node.optional_code_info.code_reference.end_col_offset)
        else:
            source_code = None
            lineno = None
            col_offset = None
            end_lineno = None
            end_col_offset = None
        attribute_values = [selected_node.operator_info.operator.value,
                            source_code,
                            selected_node.details.description,
                            f"{selected_node.details.optimizer_info.runtime:.2f} ms",
                            f"{selected_node.details.optimizer_info.shape}",
                            lineno,
                            col_offset,
                            end_lineno,
                            end_col_offset,
                            ]
        info_df = pandas.DataFrame({'Attribute': attribute_names, 'Value': attribute_values})
        st.dataframe(info_df, width=width)


def render_dag_comparison(before, after):
    st.markdown(main_description[after])
    dags_before, internal_dags_before = get_dags(before)
    dags_after, internal_dags_after = get_dags(after)
    if len(dags_before) == 1:
        dags_before = [orig_dag for orig_dag in dags_before for _ in range(len(dags_after))]
        internal_dags_before = [orig_dag for orig_dag in internal_dags_before for _ in range(len(dags_after))]
    if len(dags_after) == 1:
        dags_after = [merged_dag for merged_dag in dags_after for _ in range(len(dags_before))]
        internal_dags_after = [merged_dag for merged_dag in internal_dags_after for _ in range(len(dags_before))]

    # TODO: It seems like patches and what-if dags might not be in the same order, investigate tomorrow
    patches = st.session_state.ANALYSIS_RESULT.what_if_patches

    # Pagination
    elements_per_page = 4
    if f"{after}-page_number" not in st.session_state or st.session_state[f"{after}-page_number"] >= len(patches):
        st.session_state[f"{after}-page_number"] = 0
    last_page = len(patches) // elements_per_page
    prev, _, next = st.columns([1, 10, 1])
    if next.button("Next"):
        if st.session_state[f"{after}-page_number"] + 1 > last_page:
            st.session_state[f"{after}-page_number"] = 0
        else:
            st.session_state[f"{after}-page_number"] += 1
    if prev.button("Previous"):
        if st.session_state[f"{after}-page_number"] - 1 < 0:
            st.session_state[f"{after}-page_number"] = last_page
        else:
            st.session_state[f"{after}-page_number"] -= 1
    # Get start and end indices of the next page of the dataframe
    start_idx = st.session_state[f"{after}-page_number"] * elements_per_page
    end_idx = (1 + st.session_state[f"{after}-page_number"]) * elements_per_page

    for variant_left, variant_right in zip(range(start_idx, end_idx, 2), range(start_idx + 1, end_idx, 2)):
        left, right = st.columns(2)
        if variant_left < len(patches):
            with left:
                st.markdown(f"### Variant {variant_left}")
                render_patches(patches[variant_left])
        if variant_right < len(patches):
            with right:
                st.markdown(f"### Variant {variant_right}")
                render_patches(patches[variant_right])
        columns = st.columns(4)
        if variant_left < len(patches):
            with st.container():
                render_variant_slot(before, after, dags_before, dags_after, internal_dags_before, internal_dags_after,
                                    variant_left, columns[0:2])
        if variant_right < len(patches):
            with st.container():
                render_variant_slot(before, after, dags_before, dags_after, internal_dags_before, internal_dags_after,
                                    variant_right, columns[2:4])
        if variant_left < len(patches):
            st.markdown("""---""")


def render_variant_slot(before, after, dags_before, dags_after, internal_dags_before, internal_dags_after,
                        variant_index, columns):
    with st.container():
        left, right = columns
        with left:
            with st.container():
                st.write("#### before")
                selected_before = render_dag_slot(before, dags_before[variant_index], f"before-{before}-{variant_index}")
                if len(selected_before['nodes']) != 0:
                    render_dag_node_details(internal_dags_before[variant_index], selected_before)

        with right:
            with st.container():
                st.write("#### after")
                selected_after = render_dag_slot(after, dags_after[variant_index], f"after-{after}-{variant_index}")
                if len(selected_after['nodes']) != 0:
                    render_dag_node_details(internal_dags_after[variant_index], selected_after)


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
            elif type(patch) == ModelPatch:
                patch_names.append("Model")
                description = f"{patch.replace_with_node.details.description}"
                patch_descriptions.append(description)
            else:
                patch_names.append(type(patch).__name__)
                patch_descriptions.append("")
    variant_df = pandas.DataFrame({'Patch Type': patch_names, 'Analysis': patch_analyses,
                                   'Description': patch_descriptions})
    st.table(variant_df)
