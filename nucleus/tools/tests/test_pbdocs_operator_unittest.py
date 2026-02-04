from pbdocs_operator import AuditReport, DocCoverage, PBDocsOperator


def test_build_coverage_report_maps_categories():
    report = AuditReport(report_id="test", timestamp="2025-01-01T00:00:00Z")
    report.coverage = [
        DocCoverage(category="Python Tools", total=10, documented=7, missing=["tool_a", "tool_b"]),
        DocCoverage(category="Systemd Services", total=4, documented=1, missing=["svc_a", "svc_b", "svc_c"]),
        DocCoverage(category="Specifications", total=5, documented=5, missing=[]),
        DocCoverage(category="MCP Servers", total=2, documented=1, missing=["mcp_a"]),
    ]

    operator = PBDocsOperator(emit_bus_events=False)
    payload = operator.build_coverage_report(report)

    assert payload["total_components"] == 21
    assert payload["documented_components"] == 14
    assert payload["coverage_percent"] == 66.7

    by_category = payload["by_category"]
    assert by_category["tools"] == 66.7
    assert by_category["services"] == 25.0
    assert by_category["protocols"] == 100.0
    assert by_category["agents"] == 0.0
    assert by_category["dashboard"] == 0.0

    undocumented = payload["undocumented"]
    assert "Python Tools:tool_a" in undocumented
    assert "Systemd Services:svc_a" in undocumented
    assert "MCP Servers:mcp_a" in undocumented
