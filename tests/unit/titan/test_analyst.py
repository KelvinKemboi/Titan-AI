from titan.analyst import RoboAnalyst

_MEMO_SNAPSHOT_NO_EARNINGS = (
    "\n        ####Rating: :green[STRONG BUY] (Score: 90)\n\n"
    "        **Investment Thesis:**\n"
    "        AAPL is currently trading at **$190.50**.\n"
    "        The model has flagged this asset based on a **PEG of 1.50** and a **Bullish** long-term trend profile.\n\n"
    "        **Key Drivers:**\n"
    "        * **Momentum:** RSI is 65.0.  Healthy buying pressure.\n"
    "        * **Quality:** Net Margins of 25.0% suggest heavy competitive moat.\n"
    "        * **Risk:** Beta of 0.90 indicates low volatility.\n"
    "        "
)


def _make_analyst(ticker="AAPL"):
    analyst = RoboAnalyst(ticker)
    analyst.score = 90.75
    analyst.metrics = {
        "Price": 190.5, "RSI": 65.0, "Trend": "Bullish",
        "Val_Metric": 1.5, "Val_Type": "PEG",
        "Margin": 0.25, "Debt": 40.0, "Beta": 0.9,
        "Scores": [75.0, 100.0, 100.0, 80.0, 90.0],
    }
    return analyst


# "Tickers without earnings data render exactly as today (no regression to existing memo output)"
def test_no_earnings_insight_is_byte_identical_to_the_pre_change_snapshot():
    analyst = _make_analyst()
    analyst.generate_memo()

    assert analyst.memo == _MEMO_SNAPSHOT_NO_EARNINGS


def test_empty_dict_earnings_insight_also_skips_the_section():
    analyst = _make_analyst()
    analyst.generate_memo({})

    assert analyst.memo == _MEMO_SNAPSHOT_NO_EARNINGS


def test_generate_memo_returns_the_memo_text():
    analyst = _make_analyst()
    result = analyst.generate_memo()

    assert result == analyst.memo


# "Tickers with earnings data get an added section (summary, guidance direction, top risk) in the memo"
def test_earnings_insight_adds_a_recent_earnings_section():
    analyst = _make_analyst()
    analyst.generate_memo({
        "summary": "Record revenue driven by services growth.",
        "guidance_direction": "raised",
        "risks": [
            {"risk": "Supply chain constraints in China", "quote": "we saw some constraints"},
            {"risk": "FX headwinds", "quote": "the dollar strengthened"},
        ],
    })

    assert analyst.memo.startswith(_MEMO_SNAPSHOT_NO_EARNINGS)
    assert "**Recent Earnings:**" in analyst.memo
    assert "Record revenue driven by services growth." in analyst.memo
    assert "**Guidance:** Raised" in analyst.memo
    # only the top (first) risk is shown, not the whole list
    assert "Supply chain constraints in China" in analyst.memo
    assert "FX headwinds" not in analyst.memo


def test_guidance_direction_is_rendered_title_case_with_underscores_as_spaces():
    analyst = _make_analyst()
    analyst.generate_memo({"summary": "s", "guidance_direction": "none_given", "risks": []})

    assert "**Guidance:** None Given" in analyst.memo


def test_no_risks_falls_back_to_a_plain_message():
    analyst = _make_analyst()
    analyst.generate_memo({"summary": "s", "guidance_direction": "maintained", "risks": []})

    assert "**Top Risk:** No specific risks flagged this quarter." in analyst.memo


def test_missing_summary_and_guidance_degrade_gracefully_instead_of_crashing():
    analyst = _make_analyst()
    analyst.generate_memo({"summary": None, "guidance_direction": None, "risks": None})

    assert "**Recent Earnings:**\n        No summary available." in analyst.memo
    assert "**Guidance:** Unclear" in analyst.memo
    assert "**Top Risk:** No specific risks flagged this quarter." in analyst.memo
