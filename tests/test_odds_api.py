import pytest

from nba_ladders.sources import odds_api as oa

EVENT = {
    "id": "ev1", "sport_key": "basketball_nba", "commence_time": "2026-10-31T23:00:00Z",
    "home_team": "Washington Wizards", "away_team": "Charlotte Hornets",
    "bookmakers": [{
        "key": "draftkings", "title": "DraftKings", "last_update": "2026-10-31T20:00:00Z",
        "markets": [
            {"key": "player_rebounds_alternate", "last_update": "2026-10-31T20:00:00Z",
             "outcomes": [{"name": "Over", "description": "Moussa Diabaté", "price": 1.95, "point": 9.5},
                          {"name": "Over", "description": "Moussa Diabaté", "price": 2.6, "point": 11.5}]},
            {"key": "player_rebounds", "last_update": "2026-10-31T20:00:00Z",
             "outcomes": [{"name": "Over", "description": "Moussa Diabaté", "price": 1.87, "point": 9.5},
                          {"name": "Under", "description": "Moussa Diabaté", "price": 1.93, "point": 9.5}]},
        ]}],
}


class FakeResp:
    def __init__(self, payload, cost, remaining, status=200):
        self._p, self.status_code, self.text = payload, status, "err"
        self.headers = {"x-requests-last": str(cost), "x-requests-remaining": str(remaining)}

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, 0

    def get(self, url, params, timeout):
        assert "apiKey" in params
        self.calls += 1
        return self.resp


@pytest.fixture(autouse=True)
def key(monkeypatch, tmp_path):
    monkeypatch.setenv("ODDS_API_KEY", "t" * 32)
    monkeypatch.setattr(oa, "RAW_DIR", tmp_path)
    monkeypatch.setattr(oa.time, "sleep", lambda s: None)


def test_parse_milestone_lines_to_thresholds():
    df = oa.parse_event_odds(EVENT, "snap")
    alt = df.filter(df["alternate"])
    assert alt["threshold"].to_list() == [10, 12]  # Over 9.5 = 10+, Over 11.5 = 12+
    assert set(df["stat"]) == {"reb"}
    under = df.filter(df["side"] == "Under")
    assert under["threshold"].to_list() == [None]


def test_parse_historical_wrapper():
    df = oa.parse_event_odds({"timestamp": "t", "data": EVENT}, "snap")
    assert df.height == 4 and df["event_id"].unique().to_list() == ["ev1"]


def test_budget_refuses_before_calling():
    session = FakeSession(FakeResp(EVENT, cost=4, remaining=400))
    budget = oa.Budget(max_credits=3)
    with pytest.raises(oa.BudgetExceeded):
        oa.event_odds("ev1", oa.DEFAULT_MARKETS, "us", budget, session=session)  # ~4 credits
    assert session.calls == 0


def test_budget_tracks_spend_and_protects_reserve():
    session = FakeSession(FakeResp(EVENT, cost=2, remaining=oa.RESERVE + 3))
    budget = oa.Budget(max_credits=100)
    oa.event_odds("ev1", ["player_rebounds", "player_rebounds_alternate"], "us", budget, session=session)
    assert budget.spent == 2 and budget.remaining == oa.RESERVE + 3
    with pytest.raises(oa.BudgetExceeded):  # another 4 would dip under the reserve
        oa.event_odds("ev1", oa.DEFAULT_MARKETS, "us", budget, session=session)


def test_historical_cost_estimate_is_ten_x():
    budget = oa.Budget(max_credits=39)
    with pytest.raises(oa.BudgetExceeded):
        oa.historical_event_odds("ev1", "2026-03-15T23:00:00Z", ["player_points_alternate"] * 4, "us",
                                 budget, session=FakeSession(FakeResp(EVENT, 40, 1000)))


def test_responses_cached_once(tmp_path):
    session = FakeSession(FakeResp(EVENT, cost=2, remaining=400))
    oa.event_odds("ev1", ["player_rebounds"], "us", oa.Budget(100), session=session)
    assert len(list((tmp_path / "live").glob("ev1_*.json"))) == 1


def test_placeholder_key_is_rejected_before_any_call(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "your_key_here")
    with pytest.raises(SystemExit, match="placeholder"):
        oa.api_key()


def test_bookmakers_param_counts_as_one_region_per_ten():
    params, n = oa._book_params(None, ["sportsbet", "fanduel", "draftkings", "fanatics"])
    assert params == {"bookmakers": "sportsbet,fanduel,draftkings,fanatics"} and n == 1
    assert oa._book_params(None, [f"b{i}" for i in range(11)])[1] == 2
    assert oa._book_params("us,au", None) == ({"regions": "us,au"}, 2)
