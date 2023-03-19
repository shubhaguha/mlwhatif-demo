from copy import copy

import streamlit as st
from fairlearn.metrics import MetricFrame
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType
from mlwhatif.analysis._operator_impact import OperatorImpact
from mlwhatif.visualisation._visualisation import get_final_optimized_combined_colored_simple_dag, \
    get_original_simple_dag, get_colored_simple_dags
from st_cytoscape import cytoscape
from streamlit_ace import st_ace
from threadpoolctl import threadpool_limits

from callbacks import analyze_pipeline, get_report, render_graph3, scan_pipeline, \
    estimate_pipeline_analysis, render_dag_comparison, render_patches, render_full_size_dag
from constants import PIPELINE_CONFIG

if 'PIPELINE_SOURCE_CODE_PREV_RUN' not in st.session_state:
    st.session_state['PIPELINE_SOURCE_CODE_PREV_RUN'] = None

if 'PIPELINE_SOURCE_CODE' not in st.session_state:
    st.session_state['PIPELINE_SOURCE_CODE'] = ''

if 'ANALYSIS_RESULT' not in st.session_state:
    st.session_state['ANALYSIS_RESULT'] = None

if 'DAG_EXTRACTION_RESULT' not in st.session_state:
    st.session_state['DAG_EXTRACTION_RESULT'] = None

if 'RUNTIME_ORIG' not in st.session_state:
    st.session_state['RUNTIME_ORIG'] = None

if 'ESTIMATION_RESULT' not in st.session_state:
    st.session_state['ESTIMATION_RESULT'] = None

if 'analyses' not in st.session_state:
    st.session_state['analyses'] = {}

# The two lines below are for trying to fix some refresh bugs, it probably is not necessary to store this in the state
if 'dag_mapping' not in st.session_state:
    st.session_state['dag_mapping'] = {
        "Original": lambda: render_full_size_dag("Original"),
        "Variants": lambda: render_dag_comparison("Original", "Variants"),
        "Shared": lambda: render_dag_comparison("Variants", "Shared"),
        "FRP": lambda: render_dag_comparison("Shared", "FRP"),
        "PP": lambda: render_dag_comparison("FRP", "PP"),
        "FP": lambda: render_dag_comparison("PP", "FP"),
        "UDF": lambda: render_dag_comparison("FP", "UDF"),
        "Merging": lambda: render_dag_comparison("UDF", "Merging"),
        "Done": lambda: render_full_size_dag("Done"),
    }

if 'optimization_steps' not in st.session_state:
    st.session_state['optimization_steps'] = list(st.session_state['dag_mapping'].keys())

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
if st.sidebar.button("Load source code"):
    with open(pipeline_filename) as f:
        st.session_state['PIPELINE_SOURCE_CODE'] = f.read()
    st.session_state['ANALYSIS_RESULT'] = None
    st.session_state['DAG_EXTRACTION_RESULT'] = None
    st.session_state['RUNTIME_ORIG'] = None
    st.session_state['ESTIMATION_RESULT'] = None
# pipeline_num_lines = len(st.session_state['PIPELINE_SOURCE_CODE'].splitlines())
code_has_changed = st.session_state.PIPELINE_SOURCE_CODE != st.session_state.PIPELINE_SOURCE_CODE_PREV_RUN
scan_button = st.sidebar.button("Run and Scan Pipeline", disabled=not code_has_changed)

# What-if Analyses
if st.sidebar.checkbox("Data Corruption"):  # a.k.a. robustness
    # column_to_corruption: List[Tuple[str, Union[FunctionType, CorruptionType]]],
    column_to_corruption = {}
    selected_columns = st.sidebar.multiselect("Columns to corrupt", pipeline_columns)
    for column in selected_columns:
        column_to_corruption[column] = st.sidebar.selectbox(
            column, CorruptionType.__members__.values(), format_func=lambda m: m.value)

    # corruption_percentages: Iterable[Union[float, Callable]] or None = None,
    corruption_percentages = st.sidebar.multiselect("Corruption percentages", list(range(0, 101, 10)),
                                                    format_func=lambda i: f"{i}%")
    # corruption_percentages = []
    # num = st.sidebar.number_input(
    #     "Corruption percentage", min_value=0.0, max_value=1.0, step=0.01, key=0)
    # while num and num > 0:
    #     corruption_percentages.append(num)
    #     num = st.sidebar.number_input("Corruption percentage", min_value=0.0,
    #                                   max_value=1.0, step=0.01, key=len(corruption_percentages))

    # also_corrupt_train: bool = False):
    also_corrupt_train = st.sidebar.checkbox("Also corrupt train")

    # __init__
    robustness = DataCorruption(column_to_corruption=list(column_to_corruption.items()),
                                corruption_percentages=[p / 100. for p in corruption_percentages],
                                also_corrupt_train=also_corrupt_train)
    st.session_state.analyses["robustness"] = robustness

if st.sidebar.checkbox("Operator Impact"):
    # test_transformers=True
    test_transformers = st.sidebar.checkbox("Test transformers", value=True)

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

if st.sidebar.checkbox("Data Cleaning", key="data_cleaning"):
    # columns_with_error: dict[str or None, ErrorType] or List[Tuple[str, ErrorType]]
    labels_ui_col = "LABELS"
    columns_with_error = {}
    selected_columns = st.sidebar.multiselect("Columns with errors", pipeline_columns + [labels_ui_col],
                                              key="data_cleaning_columns")
    for column in selected_columns:
        columns_with_error[column] = st.sidebar.selectbox(
            column, ErrorType.__members__.values(), format_func=lambda m: m.value,
            key=f"data_cleaning_columns_{column}")

    # __init__
    columns_with_error_with_label_formatting = {}
    for column_name, error_name in columns_with_error.items():
        if column_name == labels_ui_col:
            columns_with_error_with_label_formatting[None] = copy(error_name)
        else:
            columns_with_error_with_label_formatting[column_name] = error_name
    cleanlearn = DataCleaning(columns_with_error=copy(columns_with_error_with_label_formatting),
                              parallelism=False)
    st.session_state.analyses["cleanlearn"] = cleanlearn

### === LAYOUT ===
left, right = st.columns(2)

with left:
    pipeline_code_container = st.expander("Pipeline Code", expanded=True)
with right:
    results_container = st.expander("What-If Analysis", expanded=True)
    with results_container:
        st.empty()

### === ACTIONS ===
estimate_button = st.sidebar.button("Estimate Execution Time", disabled=code_has_changed)
run_button = st.sidebar.button("Run Analyses", disabled=code_has_changed)

if scan_button:
    with results_container:
        with st.spinner("Running and scanning the pipeline..."):
            runtime_orig, dag_extraction_result = scan_pipeline(st.session_state.PIPELINE_SOURCE_CODE)
        st.session_state.PIPELINE_SOURCE_CODE_PREV_RUN = st.session_state.PIPELINE_SOURCE_CODE
        st.session_state.RUNTIME_ORIG = runtime_orig
        st.session_state.DAG_EXTRACTION_RESULT = dag_extraction_result
        st.session_state.ESTIMATION_RESULT = None
        st.session_state.ANALYSIS_RESULT = None
        st.experimental_rerun()

if estimate_button:
    with results_container:
        with st.spinner("Estimating analysis cost..."):
            st.session_state.ESTIMATION_RESULT = estimate_pipeline_analysis(st.session_state.DAG_EXTRACTION_RESULT,
                                                                            *st.session_state.analyses.values())
        st.session_state.ANALYSIS_RESULT = None

if run_button:
    with results_container:
        with st.spinner("Estimating analysis cost..."):
            st.session_state.ESTIMATION_RESULT = estimate_pipeline_analysis(st.session_state.DAG_EXTRACTION_RESULT,
                                                                            *st.session_state.analyses.values())
        with st.spinner("Analyzing pipeline..."):
            analysis_result = \
                analyze_pipeline(st.session_state.DAG_EXTRACTION_RESULT, *st.session_state.analyses.values())
        st.session_state.ANALYSIS_RESULT = analysis_result
        st.balloons()
        print("end")

### === MAIN CONTENT ===
with pipeline_code_container:
    # st.code(pipeline_code)
    # Check out more themes: https://github.com/okld/streamlit-ace/blob/main/streamlit_ace/__init__.py#L36-L43
    st.session_state['PIPELINE_SOURCE_CODE'] = st_ace(value=st.session_state['PIPELINE_SOURCE_CODE'],
                                                      language="python",
                                                      theme="katzenmilch",
                                                      auto_update=True)
    if st.session_state.DAG_EXTRACTION_RESULT:
        st.code(st.session_state.DAG_EXTRACTION_RESULT.captured_orig_pipeline_stdout)

with results_container:
    if st.session_state.RUNTIME_ORIG:
        runtime_orig = st.session_state.RUNTIME_ORIG
        # TODO: Should we use humanize here?
        #  from humanize import naturalsize
        #  naturaldelta or something like that
        st.write(f"Measured runtime of original pipeline is {runtime_orig:.2f} ms.")

    if st.session_state.ESTIMATION_RESULT:
        estimate = st.session_state.ESTIMATION_RESULT
        # TODO: Should we use humanize here?
        #  from humanize import naturalsize
        #  naturaldelta or something like that
        st.write(f"Estimated total runtime is {estimate.runtime_info.what_if_optimized_estimated:.2f} ms.")
        st.write("Estimated time saved with our multi-query optimization is "
                 f"{estimate.runtime_info.what_if_optimization_saving_estimated:.2f} ms.")

    if st.session_state.ANALYSIS_RESULT:
        for analysis in st.session_state.analyses.values():
            actual_runtime = st.session_state.ANALYSIS_RESULT.runtime_info.what_if_execution
            st.write(f"Measured runtime of what-if analyses is {actual_runtime:.2f} ms.")

            report = get_report(st.session_state.ANALYSIS_RESULT, analysis)

            metrics_frame_columns = report.select_dtypes('object')
            for column in metrics_frame_columns:
                if len(report) != 0 and isinstance(report[column].iloc[0], MetricFrame):
                    # TODO: Better visualisation or remove MetricFrame from healthcare pipeline
                    # Try `config.dataFrameSerialization = "arrow"` in case pyarrow's df serialization is better
                    report[column] = report.apply(lambda row: str(row[column].by_group), axis=1)
            # TODO: Map mislabel cleaning None back to LABELS?

            st.subheader(analysis.__class__.__name__)
            st.dataframe(report)

st.markdown("""---""")
st.markdown("## How it works")

# TODO: Expander or not?
# with st.expander("How it works"):
# TODO: for me, there are random refreshes sometimes. Then an expanser makes the user experience even worse.
#  However, if others have the same refresh issues, we should try to fix them
dag_choice = st.radio("", st.session_state['optimization_steps'], horizontal=True, key="refresh-bugfix-key")

st.session_state['dag_mapping'][dag_choice]()
