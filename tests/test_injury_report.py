from nba_ladders.sources.injury_report import parse_words, player_key


def w(x0, top, text):
    return {"x0": x0, "top": top, "text": text}


HEADER = [w(23, 108, "GameDate"), w(425, 108, "PlayerName"), w(586, 108, "CurrentStatus")]


def test_player_key_matches_report_and_box_score_styles():
    assert player_key("Udeh,Jr.,Ernest") == player_key("Ernest Udeh Jr.")
    assert player_key("Vučević,Nikola") == player_key("Nikola Vucevic")
    assert player_key("Nae'Qwan Tomlin") == player_key("Tomlin,Nae'Qwan")


def test_parse_words_carries_blocks_and_attaches_wrapped_reasons():
    words = HEADER + [
        w(24, 129, "10/24/2026"), w(121, 129, "07:30(ET)"), w(201, 129, "BKN@CLE"),
        w(265, 129, "BrooklynNets"), w(426, 129, "Bilodeau,Tyler"), w(587, 129, "Out"),
        w(667, 129, "Shoulder"),
        w(667, 140, "Injury/Illness-LeftKneeInjury"),  # wrapped line above Diop
        w(265, 151, "ClevelandCavaliers"), w(426, 151, "Diop,Khalifa"), w(587, 151, "Out"),
        w(667, 162, "Management"),  # ...and below
        w(265, 185, "SacramentoKings"), w(667, 185, "NOTYETSUBMITTED"),
        w(289, 46, "Injury"), w(356, 46, "Report:"),  # page title must be ignored
    ]
    rows = parse_words(words)
    assert [r["player"] for r in rows] == ["Bilodeau,Tyler", "Diop,Khalifa", None]
    diop = rows[1]
    assert diop["team_name"] == "ClevelandCavaliers" and diop["matchup"] == "BKN@CLE"
    assert diop["status"] == "Out"
    assert diop["reason"] == "Injury/Illness-LeftKneeInjury Management"
    assert rows[2]["team_name"] == "SacramentoKings" and rows[2]["status"] == "Not submitted"


def test_later_pages_without_header_ignore_title():
    rows = parse_words([w(289, 46, "Injury"), w(356, 46, "Report:"), w(438, 46, "10/24/26"),
                        w(426, 90, "White,Coby"), w(587, 90, "Out")])
    assert [r["player"] for r in rows] == ["White,Coby"]
