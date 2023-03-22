from copy import copy

import pandas
import streamlit as st
from fairlearn.metrics import MetricFrame
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType
from mlwhatif.analysis._operator_impact import OperatorImpact
from streamlit_ace import st_ace

from callbacks import analyze_pipeline, get_report, scan_pipeline, \
    estimate_pipeline_analysis, render_dag_comparison, render_full_size_dag
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
    dag_mapping = {
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
    optimization_steps = list(dag_mapping.keys())
    st.session_state['dag_mapping'] = dag_mapping
    st.session_state['optimization_steps'] = optimization_steps

st.set_page_config(page_title="mlwhatif", page_icon="🧐", layout="wide")
st.title("`mlwhatif` demo")
# with st.echo():
#     st.__version__


### === SIDEBAR / CONFIGURATION ===
st.sidebar.title("Configuration")

# Pipeline
if 'pipeline_file_name_index' not in st.session_state:
    st.session_state['pipeline_file_name_index'] = 0
pipeline = st.sidebar.selectbox("Choose a pipeline", list(PIPELINE_CONFIG.keys()), key="pipeline-selection",
                                index=st.session_state['pipeline_file_name_index'])
st.session_state['pipeline_file_name_index'] = list(PIPELINE_CONFIG.keys()).index(pipeline)

pipeline_filename = PIPELINE_CONFIG[pipeline]["filename"]
pipeline_columns = PIPELINE_CONFIG[pipeline]["columns"]
if st.sidebar.button("Load source code", key="source-code-loading"):
    with open(pipeline_filename) as f:
        st.session_state['PIPELINE_SOURCE_CODE'] = f.read()
    st.session_state['ANALYSIS_RESULT'] = None
    st.session_state['DAG_EXTRACTION_RESULT'] = None
    st.session_state['RUNTIME_ORIG'] = None
    st.session_state['ESTIMATION_RESULT'] = None
# pipeline_num_lines = len(st.session_state['PIPELINE_SOURCE_CODE'].splitlines())
code_has_changed = st.session_state.PIPELINE_SOURCE_CODE != st.session_state.PIPELINE_SOURCE_CODE_PREV_RUN
scan_button = st.sidebar.button("Run and Scan Pipeline", disabled=not code_has_changed, key="scan-button")

# What-if Analyses
with st.sidebar.expander("Robustness"):  # a.k.a. Data Corruption
    data_corruption_active = st.checkbox("Enable analysis", key="enable_corruptions",
                                         value=st.session_state.get("enable_corruptions", False))

    # column_to_corruption: List[Tuple[str, Union[FunctionType, CorruptionType]]],
    if '_data_corruption_columns' not in st.session_state:
        st.session_state['_data_corruption_columns'] = []
    column_to_corruption = {}
    selected_columns = st.multiselect("Columns to corrupt", pipeline_columns, key="corruption-columns",
                                      default=st.session_state['_data_corruption_columns'])
    st.session_state['_data_corruption_columns'] = selected_columns
    corruption_types = list(CorruptionType.__members__.values())
    for column in selected_columns:
        if f'_data_corruption_type_idx__{column}' not in st.session_state:
            st.session_state[f'_data_corruption_type_idx__{column}'] = 0
        column_to_corruption[column] = st.selectbox(
            column, corruption_types, format_func=lambda m: m.value,
            key=f"corruption-columns-{column}", index=st.session_state[f'_data_corruption_type_idx__{column}'])
        st.session_state[f'_data_corruption_type_idx__{column}'] = corruption_types.index(column_to_corruption[column])

    # corruption_percentages: Iterable[Union[float, Callable]] or None = None,
    if '_data_corruption_percentages' not in st.session_state:
        st.session_state['_data_corruption_percentages'] = [40, 70, 100]
    corruption_percentages = st.multiselect("Corruption percentages", list(range(0, 101, 10)),
                                            default=st.session_state['_data_corruption_percentages'],
                                            format_func=lambda i: f"{i}%",
                                            key="corruption-percentages")
    st.session_state['_data_corruption_percentages'] = corruption_percentages
    # corruption_percentages = []
    # num = st.sidebar.number_input(
    #     "Corruption percentage", min_value=0.0, max_value=1.0, step=0.01, key=0)
    # while num and num > 0:
    #     corruption_percentages.append(num)
    #     num = st.sidebar.number_input("Corruption percentage", min_value=0.0,
    #                                   max_value=1.0, step=0.01, key=len(corruption_percentages))

    # also_corrupt_train: bool = False):

    if '_also_corrupt_train' not in st.session_state:
        st.session_state['_also_corrupt_train'] = False
    also_corrupt_train = st.checkbox("Also corrupt train", key="corruption-train",
                                     value=st.session_state['_also_corrupt_train'])
    st.session_state['_also_corrupt_train'] = also_corrupt_train

    # __init__
    if data_corruption_active:
        robustness = DataCorruption(column_to_corruption=list(column_to_corruption.items()),
                                    corruption_percentages=[p / 100. for p in corruption_percentages],
                                    also_corrupt_train=also_corrupt_train)
        st.session_state.analyses["robustness"] = robustness
    else:
        st.session_state.analyses.pop("robustness", None)

with st.sidebar.expander("Operator Impact"):
    if '_operator_impact_active' not in st.session_state:
        st.session_state['_operator_impact_active'] = False
    operator_impact_active = st.checkbox("Enable analysis", key="operator-impact",
                                         value=st.session_state['_operator_impact_active'])
    st.session_state['_operator_impact_active'] = operator_impact_active

    # test_transformers=True
    if '_test_transformers' not in st.session_state:
        st.session_state['_test_transformers'] = True
    test_transformers = st.checkbox("Test transformers", key="operator-impact-transformers",
                                    value=st.session_state['_test_transformers'])
    st.session_state['_test_transformers'] = test_transformers

    # test_selections=False
    if '_test_selections' not in st.session_state:
        st.session_state['_test_selections'] = False
    test_selections = st.checkbox("Test selections", key="operator-impact-selections",
                                  value=st.session_state['_test_selections'])
    st.session_state['_test_selections'] = test_selections

    # restrict_to_linenos: List[int] or None = None
    # line_numbers = []
    # num = st.sidebar.number_input(
    #     "Line number", min_value=1, max_value=pipeline_num_lines, step=1, key=0)
    # while num and num > 0:
    #     line_numbers.append(num)
    #     num = st.sidebar.number_input("Line number", min_value=1, max_value=pipeline_num_lines,
    #                                   step=1, key=len(line_numbers))

    # __init__
    if operator_impact_active:
        preproc = OperatorImpact(test_transformers=test_transformers, test_selections=test_selections)
        st.session_state.analyses["preproc"] = preproc
    else:
        st.session_state.analyses.pop("preproc", None)

# columns_with_error: dict[str or None, ErrorType] or List[Tuple[str, ErrorType]]
with st.sidebar.expander("Data Cleaning"):
    if '_data_cleaning_active' not in st.session_state:
        st.session_state['_data_cleaning_active'] = False
    data_cleaning_active = st.checkbox("Enable analysis", key="data_cleaning",
                                       value=st.session_state['_data_cleaning_active'])
    st.session_state['_data_cleaning_active'] = data_cleaning_active

    labels_ui_col = "LABELS"
    columns_with_error = {}
    selected_columns = st.multiselect("Columns with errors", pipeline_columns + [labels_ui_col],
                                      key="data_cleaning_columns",
                                      default=st.session_state.get('data_cleaning_columns', []))
    error_types = list(ErrorType.__members__.values())
    for column in selected_columns:
        if f'_data_cleaning_error_type_idx__{column}' not in st.session_state:
            st.session_state[f'_data_cleaning_error_type_idx__{column}'] = 0
        columns_with_error[column] = st.selectbox(
            column, error_types, format_func=lambda m: m.value,
            key=f"data_cleaning_columns_{column}",
            index=st.session_state[f'_data_cleaning_error_type_idx__{column}'])
        st.session_state[f'_data_cleaning_error_type_idx__{column}'] = error_types.index(columns_with_error[column])

    # __init__
    columns_with_error_with_label_formatting = {}
    for column_name, error_name in columns_with_error.items():
        if column_name == labels_ui_col:
            columns_with_error_with_label_formatting[None] = copy(error_name)
        else:
            columns_with_error_with_label_formatting[column_name] = error_name
    if data_cleaning_active:
        cleanlearn = DataCleaning(columns_with_error=copy(columns_with_error_with_label_formatting),
                                  parallelism=False)
        st.session_state.analyses["cleanlearn"] = cleanlearn
    else:
        st.session_state.analyses.pop("cleanlearn", None)

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
            waiting_message = f"Estimated total runtime is " \
                              f"{st.session_state.ESTIMATION_RESULT.runtime_info.what_if_optimized_estimated:.2f} ms."
        with st.spinner(f"Analyzing pipeline... {waiting_message}"):
            analysis_result = \
                analyze_pipeline(st.session_state.DAG_EXTRACTION_RESULT, *st.session_state.analyses.values())
        st.session_state.ANALYSIS_RESULT = analysis_result
        st.balloons()

with st.sidebar:
    st.markdown("")
    st.markdown("")

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
    runtime_messages = ""
    if st.session_state.RUNTIME_ORIG:
        runtime_orig = st.session_state.RUNTIME_ORIG
        # TODO: Should we use humanize here?
        #  from humanize import naturalsize
        #  naturaldelta or something like that
        runtime_messages += f"The runtime of the original pipeline is `{runtime_orig:.2f} ms`.  \n  \n  "

    if st.session_state.ESTIMATION_RESULT:
        estimate = st.session_state.ESTIMATION_RESULT
        # TODO: Should we use humanize here?
        #  from humanize import naturalsize
        #  naturaldelta or something like that
        runtime_messages += (f"The estimated runtime of the configured what-if analyses is "
                             f"`{estimate.runtime_info.what_if_optimized_estimated:.2f} ms.`  \n "
                             f"The estimated runtime saved because of our multi-query optimizer is "
                             f"`{estimate.runtime_info.what_if_optimization_saving_estimated:.2f} ms.`  \n  \n  ")

    if st.session_state.ANALYSIS_RESULT:
        actual_runtime = st.session_state.ANALYSIS_RESULT.runtime_info.what_if_execution
        runtime_messages += f"The actual runtime of the configured what-if analyses is `{actual_runtime:.2f} ms`." \
                            f"  \n "
        st.markdown(runtime_messages)
        for analysis in st.session_state.analyses.values():
            if analysis in st.session_state.ANALYSIS_RESULT.analysis_to_result_reports:
                report = get_report(st.session_state.ANALYSIS_RESULT, analysis)

                metrics_frame_columns = report.select_dtypes('object')
                for column in metrics_frame_columns:
                    if len(report) != 0 and isinstance(report[column].iloc[0], MetricFrame):
                        def format_metric_frame(row):
                            pandas_df = row[column].by_group.reset_index(drop=False)
                            pandas_groups = pandas_df.iloc[:, 0].tolist()
                            pandas_values = pandas_df.iloc[:, 1].tolist()
                            results = []
                            for group, value in zip(pandas_groups, pandas_values):
                                results.append(f"'{group}': {value:.3f}")
                            return ", ".join(results)
                        report[column] = report.apply(format_metric_frame, axis=1)
                for column in list(report.columns):
                    if "percentage" in column or "lineno" in column:
                        def format_percentage_column(row):
                            number = row[column]
                            if type(number) == float and not pandas.isna(number):
                                if "percentage" in column:
                                    number = number * 100
                                result = str(int(number))
                                if "percentage" in column:
                                    result += "%"
                            elif type(number) == str:
                                result = number
                            else:
                                result = "<NA>"
                            return result

                        report[column] = report.apply(format_percentage_column, axis=1)
                # TODO: Map mislabel cleaning None back to LABELS

                if type(analysis) == DataCorruption:
                    header = "Robustness"
                elif type(analysis) == OperatorImpact:
                    header = "Operator Impact"
                else:
                    header = "Data Cleaning"
                    report.loc[report["error"] == "mislabel", "corrupted_column"] = "LABELS"
                st.subheader(header)
                st.dataframe(report)
    else:
        st.markdown(runtime_messages)

st.markdown("""---""")
st.markdown("## How it works")

# TODO: Expander or not?
# with st.expander("How it works"):
# TODO: for me, there are random refreshes sometimes. Then an expanser makes the user experience even worse.
#  However, if others have the same refresh issues, we should try to fix them
# The below code prevents the reloads from being too noticeable
if 'optimization_step_selection_index' not in st.session_state:
    st.session_state['optimization_step_selection_index'] = 0
dag_choice = st.radio("", st.session_state['optimization_steps'], horizontal=True, key="refresh-bugfix-key",
                      index=st.session_state['optimization_step_selection_index'])
st.session_state['optimization_step_selection_index'] = st.session_state['optimization_steps'].index(dag_choice)

st.session_state['dag_mapping'][dag_choice]()
