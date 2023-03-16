import streamlit as st
from fairlearn.metrics import MetricFrame
from mlwhatif.visualisation._visualisation import get_final_optimized_combined_colored_simple_dag, \
    get_original_simple_dag, get_colored_simple_dags
from st_cytoscape import cytoscape
from streamlit_ace import st_ace

from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType
from mlwhatif.analysis._permutation_feature_importance import PermutationFeatureImportance
from mlwhatif.analysis._operator_impact import OperatorImpact
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType

from callbacks import analyze_pipeline, get_report, render_graph1, render_graph2, render_graph3
from constants import PIPELINE_CONFIG


if 'ANALYSIS_RESULT' not in st.session_state:
    st.session_state['ANALYSIS_RESULT'] = None

if 'analyses' not in st.session_state:
    st.session_state['analyses'] = {}

st.set_page_config(page_title="mlwhatif", page_icon="🧐", layout="wide")
st.title("`mlwhatif` demo")
# with st.echo():
#     st.__version__


### === SIDEBAR / CONFIGURATION ===
st.sidebar.title("Configuration")

# Pipeline
pipeline = st.sidebar.selectbox("Choose a pipeline", list(PIPELINE_CONFIG.keys()))
pipeline_filename = PIPELINE_CONFIG[pipeline]["filename"]
pipeline_columns = PIPELINE_CONFIG[pipeline]["columns"]
if pipeline_filename:
    with open(pipeline_filename) as f:
        pipeline_code = f.read()
    pipeline_num_lines = len(pipeline_code.splitlines())

# What-if Analyses

if st.sidebar.checkbox("Data Corruption"):  # a.k.a. robustness
    # column_to_corruption: List[Tuple[str, Union[FunctionType, CorruptionType]]],
    column_to_corruption = {}
    selected_columns = st.sidebar.multiselect("Columns to corrupt", pipeline_columns)
    for column in selected_columns:
        column_to_corruption[column] = st.sidebar.selectbox(
            column, CorruptionType.__members__.values(), format_func=lambda m: m.value)

    # corruption_percentages: Iterable[Union[float, Callable]] or None = None,
    corruption_percentages = []
    num = st.sidebar.number_input(
        "Corruption percentage", min_value=0.0, max_value=1.0, step=0.01, key=0)
    while num and num > 0:
        corruption_percentages.append(num)
        num = st.sidebar.number_input("Corruption percentage", min_value=0.0,
                                      max_value=1.0, step=0.01, key=len(corruption_percentages))

    # also_corrupt_train: bool = False):
    also_corrupt_train = st.sidebar.checkbox("Also corrupt train")

    # __init__
    robustness = DataCorruption(column_to_corruption=list(column_to_corruption.items()),
                                corruption_percentages=corruption_percentages,
                                also_corrupt_train=also_corrupt_train)
    st.session_state.analyses["robustness"] = robustness

# if st.sidebar.checkbox("Permutation Importance"):
#     # restrict_to_columns: Iterable[str] or None = None
#     restrict_to_columns = st.sidebar.multiselect("Restrict to columns", pipeline_columns)
#
#     # __init__
#     importance = PermutationFeatureImportance(restrict_to_columns=restrict_to_columns)
#     analyses["importance"] = importance

if st.sidebar.checkbox("Operator Impact"):
    # test_transformers=True
    test_transformers = st.sidebar.checkbox("Test selections", value=True)

    # test_selections=False
    test_selections = st.sidebar.checkbox("Test selections")

    # restrict_to_linenos: List[int] or None = None
    # line_numbers = []
    # num = st.sidebar.number_input(
    #     "Line number", min_value=1, max_value=pipeline_num_lines, step=1, key=0)
    # while num and num > 0:
    #     line_numbers.append(num)
    #     num = st.sidebar.number_input("Line number", min_value=1, max_value=pipeline_num_lines,
    #                                   step=1, key=len(line_numbers))

    # __init__
    preproc = OperatorImpact(test_transformers=test_transformers, test_selections=test_selections)
    st.session_state.analyses["preproc"] = preproc

if st.sidebar.checkbox("Data Cleaning"):
    # columns_with_error: dict[str or None, ErrorType] or List[Tuple[str, ErrorType]]
    columns_with_error = {}
    selected_columns = st.sidebar.multiselect("Columns with errors", pipeline_columns + ["_TARGET_"])
    for column in selected_columns:
        columns_with_error[column] = st.sidebar.selectbox(
            column, ErrorType.__members__.values(), format_func=lambda m: m.value)

    # __init__
    cleanlearn = DataCleaning(columns_with_error=columns_with_error)
    st.session_state.analyses["cleanlearn"] = cleanlearn

# Actions
scan_button = st.sidebar.button("Scan Pipeline")
estimate_button = st.sidebar.button(
    "Estimate Execution Time", disabled=not scan_button)  # estimate execution time
run_button = st.sidebar.button("Run Analyses")


### === LAYOUT ===
left, right = st.columns(2)

with left:
    pipeline_code_container = st.expander("Pipeline Code", expanded=True)
    original_dag_container = st.expander("Original DAG")
    intermediate_container_0 = st.expander("Generated Variants")
    intermediate_container_1 = st.expander("Common Subexpression Elimination")
    intermediate_container_2 = st.expander("Filter Removal Push-Up")
    intermediate_container_3 = st.expander("Projection Removal Push-Up")
    intermediate_container_4 = st.expander("Filter Removal Push-Up")
    intermediate_container_5 = st.expander("UDF Split-Reuse")
with right:
    analysis_results_container = st.expander("Analysis Results", expanded=True)
    optimized_dag_container = st.expander("Optimized DAG")

### === MAIN CONTENT ===
with left:
    with pipeline_code_container:
        # st.code(pipeline_code)
        # Check out more themes: https://github.com/okld/streamlit-ace/blob/main/streamlit_ace/__init__.py#L36-L43
        final_pipeline_code = st_ace(value=pipeline_code, language="python", theme="katzenmilch")

with right:
    if run_button:
        with analysis_results_container:
            with st.spinner("Analyzing pipeline..."):
                st.session_state.ANALYSIS_RESULT = \
                    analyze_pipeline(pipeline_filename, *st.session_state.analyses.values())
            st.balloons()

            for analysis in st.session_state.analyses.values():
                report = get_report(st.session_state.ANALYSIS_RESULT, analysis)

                metrics_frame_columns = report.select_dtypes('object')
                for column in metrics_frame_columns:
                    if len(report) != 0 and isinstance(report[column].iloc[0], MetricFrame):
                        # TODO: Better visualisation or remove MetricFrame from healthcare pipeline
                        report[column] = report.apply(lambda row: str(row[column].by_group), axis=1)

                st.subheader(analysis.__class__.__name__)
                st.table(report)

if st.session_state.ANALYSIS_RESULT:
    with optimized_dag_container:
        combined_plan = get_final_optimized_combined_colored_simple_dag(
            st.session_state.ANALYSIS_RESULT.intermediate_stages["4-optimize_patches_3_UdfSplitAndReuse"])
        cytoscape_data, stylesheet = render_graph3(combined_plan)
        selected = cytoscape(cytoscape_data, stylesheet, key="optimized-plan", layout={"name": "dagre"})

        st.markdown("**Selected nodes**: %s" % (", ".join(selected["nodes"])))
        st.markdown("**Selected edges**: %s" % (", ".join(selected["edges"])))

with left:
    if st.session_state.ANALYSIS_RESULT:
        with original_dag_container:
            original_plan = get_original_simple_dag(st.session_state.ANALYSIS_RESULT.original_dag)
            cytoscape_data, stylesheet = render_graph3(original_plan)
            selected = cytoscape(cytoscape_data, stylesheet, key="original-plan", layout={"name": "dagre"})

            st.markdown("**Selected nodes**: %s" % (", ".join(selected["nodes"])))
            st.markdown("**Selected edges**: %s" % (", ".join(selected["edges"])))
        with intermediate_container_0:
            # with_reuse_coloring=False here is important
            colored_simple_dags = get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"],
                with_reuse_coloring=False)
            for dag_index, what_if_dag in enumerate(colored_simple_dags):
                st.markdown(f"Variant {dag_index}")
                cytoscape_data, stylesheet = render_graph3(what_if_dag)
                selected = cytoscape(cytoscape_data, stylesheet, key=f"plan-0-unopt-variants-{dag_index}",
                                     layout={"name": "dagre"})
        with intermediate_container_1:
            colored_simple_dags = get_colored_simple_dags(
                st.session_state.ANALYSIS_RESULT.intermediate_stages["0-unoptimized_variants"],
                with_reuse_coloring=True)
            for dag_index, what_if_dag in enumerate(colored_simple_dags):
                st.markdown(f"Variant {dag_index}")
                cytoscape_data, stylesheet = render_graph3(what_if_dag)
                selected = cytoscape(cytoscape_data, stylesheet, key=f"plan-1-cse-{dag_index}",
                                     layout={"name": "dagre"})
        # FIXME: The optimize dag filter removal is not handled correctly
        print(st.session_state.ANALYSIS_RESULT.intermediate_stages.keys())
        with intermediate_container_2:
            colored_simple_dags = get_colored_simple_dags(st.session_state.ANALYSIS_RESULT.intermediate_stages["3-optimize_patches_2_OperatorDeletionFilterPushUp"],
                                                          with_reuse_coloring=True)
            for dag_index, what_if_dag in enumerate(colored_simple_dags):
                st.markdown(f"Variant {dag_index}")
                cytoscape_data, stylesheet = render_graph3(what_if_dag)
                selected = cytoscape(cytoscape_data, stylesheet, key=f"plan-2-filter-remove-{dag_index}",
                                     layout={"name": "dagre"})
        with intermediate_container_3:
            colored_simple_dags = get_colored_simple_dags(st.session_state.ANALYSIS_RESULT.intermediate_stages["1-optimize_patches_0_SimpleProjectionPushUp"],
                                                          with_reuse_coloring=True)
            for dag_index, what_if_dag in enumerate(colored_simple_dags):
                st.markdown(f"Variant {dag_index}")
                cytoscape_data, stylesheet = render_graph3(what_if_dag)
                selected = cytoscape(cytoscape_data, stylesheet, key=f"plan-3-proj-add-{dag_index}",
                                     layout={"name": "dagre"})
        with intermediate_container_4:
            colored_simple_dags = get_colored_simple_dags(st.session_state.ANALYSIS_RESULT.intermediate_stages["2-optimize_patches_1_SimpleFilterAdditionPushUp"],
                                                          with_reuse_coloring=True)
            for dag_index, what_if_dag in enumerate(colored_simple_dags):
                st.markdown(f"Variant {dag_index}")
                cytoscape_data, stylesheet = render_graph3(what_if_dag)
                selected = cytoscape(cytoscape_data, stylesheet, key=f"plan-4-filter-add-{dag_index}",
                                     layout={"name": "dagre"})
        with intermediate_container_5:
            colored_simple_dags = get_colored_simple_dags(st.session_state.ANALYSIS_RESULT.intermediate_stages["4-optimize_patches_3_UdfSplitAndReuse"],
                                                          with_reuse_coloring=True)
            for dag_index, what_if_dag in enumerate(colored_simple_dags):
                st.markdown(f"Variant {dag_index}")
                cytoscape_data, stylesheet = render_graph3(what_if_dag)
                selected = cytoscape(cytoscape_data, stylesheet, key=f"plan-5-udf-add-{dag_index}",
                                     layout={"name": "dagre"})
