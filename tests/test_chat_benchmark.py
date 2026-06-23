from evaluation.chat_benchmark import ChatBenchmarkRow, percentile, result_rows


def test_percentile_interpolates_expected_value() -> None:
    assert percentile([100.0, 200.0, 300.0], 0.5) == 200.0


def test_result_rows_rounds_latency_fields() -> None:
    rows = result_rows(
        [
            ChatBenchmarkRow(
                question="What is termination for convenience?",
                expected_clause_type="Termination For Convenience",
                category="demo",
                reranking_enabled=True,
                limit=5,
                result_count=1,
                resolved_clause_type="Termination For Convenience",
                abstained=False,
                wall_clock_latency_ms=123.4567,
                reported_total_latency_ms=234.5678,
                first_token_latency_ms=45.6789,
                rewrite_latency_ms=12.3456,
                contextualization_latency_ms=1.2345,
                retrieval_latency_ms=23.4567,
                embedding_latency_ms=8.1234,
                vector_search_latency_ms=9.2345,
                reranker_loading_latency_ms=10.3456,
                reranking_latency_ms=34.5678,
                answer_latency_ms=56.789,
                generation_first_token_latency_ms=22.3456,
                prompt_chars=2400,
                evidence_chars=1600,
                estimated_input_tokens=600,
            )
        ]
    )

    assert rows[0]["wall_clock_latency_ms"] == 123.457
    assert rows[0]["first_token_latency_ms"] == 45.679
    assert rows[0]["reranking_enabled"] is True
