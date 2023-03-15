from mlwhatif import PipelineAnalyzer


def analyze_pipeline(pipeline_filename, *_what_if_analyses):
    builder = PipelineAnalyzer.on_pipeline_from_py_file(pipeline_filename)

    for analysis in _what_if_analyses:
        builder = builder.add_what_if_analysis(analysis)

    analysis_result = builder.execute()

    return analysis_result


def get_report(result, what_if_analysis):
    report = result.analysis_to_result_reports[what_if_analysis]
    return report
