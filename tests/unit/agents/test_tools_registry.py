from src.agents.tools import DISPATCH, TOOLS, call_tool


# every tool submodule's schemas are aggregated into one list Claude sees
def test_tools_aggregates_every_submodule():
    assert {t["name"] for t in TOOLS} == {
        "get_factor_scores", "compare_tickers", "search_memos", "get_earnings_insight",
        "get_qoq_changes", "search_earnings", "get_analyst_report",
    }


# ... and DISPATCH covers exactly the same names, with no collisions silently dropping one
def test_dispatch_has_no_name_collisions_and_matches_tools():
    assert set(DISPATCH.keys()) == {t["name"] for t in TOOLS}


def test_call_tool_raises_for_unknown_name():
    try:
        call_tool(db=None, name="not_a_real_tool", tool_input={})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_a_real_tool" in str(exc)
