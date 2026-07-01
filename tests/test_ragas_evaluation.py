import json
import sys
import types
from pathlib import Path

from evaluation.ragas_cases import load_ragas_cases
from evaluation.ragas_eval import (
    RagasCollectedCase,
    install_ragas_vertexai_import_shim,
    ragas_dataset_rows,
    run_ragas_scores,
    summarize_results,
)


def test_load_ragas_cases_reads_seed_dataset() -> None:
    cases = load_ragas_cases()

    assert len(cases) == 12
    assert cases[0].case_id == "assignment-unauthorized-consequence"
    assert cases[0].critical is True
    assert cases[-1].expected_clause_type is None
    assert cases[-1].messages[-1].content == (
        "Is either party required to indemnify the other?"
    )


def test_ragas_dataset_rows_use_expected_columns() -> None:
    row = RagasCollectedCase(
        case_id="case",
        critical=False,
        user_input="Question?",
        response="Answer.",
        reference="Reference.",
        retrieved_contexts=["Context."],
        expected_clause_type="Audit Rights",
        resolved_clause_type="Audit Rights",
        abstained=False,
    )

    assert ragas_dataset_rows([row]) == [
        {
            "user_input": "Question?",
            "response": "Answer.",
            "reference": "Reference.",
            "retrieved_contexts": ["Context."],
        }
    ]


def test_summarize_results_reports_threshold_failures_and_json_shape(
    tmp_path: Path,
) -> None:
    rows = [
        RagasCollectedCase(
            case_id="critical-case",
            critical=True,
            user_input="Question?",
            response="Answer.",
            reference="Reference.",
            retrieved_contexts=["Context."],
            expected_clause_type="Anti-Assignment",
            resolved_clause_type="Anti-Assignment",
            abstained=False,
        )
    ]

    report = summarize_results(
        rows=rows,
        scores=[
            {
                "faithfulness": 0.70,
                "answer_relevancy": 0.82,
                "context_precision": 0.90,
                "context_recall": 0.80,
            }
        ],
        judge_model="mock-judge",
    )

    assert report["passed"] is False
    assert report["judge_model"] == "mock-judge"
    assert report["aggregate_means"]["faithfulness"] == 0.7
    assert report["cases"][0]["scores"]["answer_relevancy"] == 0.82
    assert any("faithfulness mean" in failure for failure in report["failures"])
    assert any("critical faithfulness" in failure for failure in report["failures"])

    output = tmp_path / "ragas_eval_results.json"
    output.write_text(json.dumps(report), encoding="utf-8")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["thresholds"]["critical_faithfulness_min"] == 0.75


def test_install_ragas_vertexai_import_shim_provides_legacy_module(monkeypatch) -> None:
    module_name = "langchain_community.chat_models.vertexai"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    install_ragas_vertexai_import_shim()

    module = sys.modules[module_name]
    assert hasattr(module, "ChatVertexAI")


def test_run_ragas_scores_passes_openai_client_to_llm_factory(monkeypatch) -> None:
    captured = {}

    class EvaluationDataset:
        @classmethod
        def from_list(cls, rows):
            captured["dataset_rows"] = rows
            return "dataset"

    class Result:
        scores = [{"faithfulness": 1.0}]

    def evaluate(**kwargs):
        captured["evaluate"] = kwargs
        return Result()

    def llm_factory(model, *, client):
        captured["model"] = model
        captured["client"] = client
        return "llm"

    def embedding_factory(provider, *, interface):
        captured["embedding_provider"] = provider
        captured["embedding_interface"] = interface
        return "embeddings"

    class Metric:
        def __init__(self, **kwargs):
            captured.setdefault("metrics", []).append((self.__class__.__name__, kwargs))

    class RunConfig:
        def __init__(self, **kwargs):
            captured["run_config"] = kwargs

    ragas_module = types.ModuleType("ragas")
    ragas_module.EvaluationDataset = EvaluationDataset
    ragas_module.evaluate = evaluate

    ragas_llms_module = types.ModuleType("ragas.llms")
    ragas_llms_module.llm_factory = llm_factory

    ragas_embeddings_module = types.ModuleType("ragas.embeddings.base")
    ragas_embeddings_module.embedding_factory = embedding_factory

    ragas_metrics_module = types.ModuleType("ragas.metrics")
    ragas_metrics_module.Faithfulness = Metric
    ragas_metrics_module.LLMContextPrecisionWithReference = Metric
    ragas_metrics_module.LLMContextRecall = Metric
    ragas_metrics_module.ResponseRelevancy = Metric

    ragas_run_config_module = types.ModuleType("ragas.run_config")
    ragas_run_config_module.RunConfig = RunConfig

    class OpenAI:
        pass

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = OpenAI

    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms_module)
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", ragas_embeddings_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.run_config", ragas_run_config_module)
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    scores = run_ragas_scores(
        [{"user_input": "Question?", "response": "Answer."}],
        judge_model="judge-model",
    )

    assert scores == [{"faithfulness": 1.0}]
    assert captured["model"] == "judge-model"
    assert isinstance(captured["client"], OpenAI)
    assert captured["evaluate"]["llm"] == "llm"
    assert captured["embedding_provider"] == "text-embedding-3-small"
    assert captured["embedding_interface"] == "legacy"
    assert captured["evaluate"]["embeddings"] == "embeddings"
    assert captured["evaluate"]["run_config"].__class__ is RunConfig
    assert captured["run_config"] == {
        "timeout": 60,
        "max_retries": 2,
        "max_wait": 10,
        "max_workers": 4,
    }
    assert ("Metric", {"strictness": 1}) in captured["metrics"]
