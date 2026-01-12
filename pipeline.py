import re
import unicodedata
from difflib import get_close_matches

import pandas as pd
import requests
from understatapi import UnderstatClient


def norm_text(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def per90(series, minutes):
    mins = minutes.replace(0, pd.NA)
    return (series / mins) * 90


def coalesce(primary, secondary):
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    return primary.combine_first(secondary)


def fetch_fpl_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def build_fpl_df(data):
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])[["id", "name", "short_name"]].rename(columns={"id": "team_id"})
    pos = pd.DataFrame(data["element_types"])[["id", "singular_name_short"]].rename(columns={"id": "pos_id"})
    fpl = players.merge(teams, left_on="team", right_on="team_id", how="left").merge(pos, left_on="element_type", right_on="pos_id", how="left")

    keep = [
        "id",
        "first_name",
        "second_name",
        "web_name",
        "name",
        "short_name",
        "singular_name_short",
        "minutes",
        "starts",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "yellow_cards",
        "red_cards",
        "influence",
        "creativity",
        "threat",
        "ict_index",
    ]
    fpl = fpl[[c for c in keep if c in fpl.columns]].copy()

    to_num(
        fpl,
        [
            "minutes",
            "starts",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "saves",
            "yellow_cards",
            "red_cards",
            "influence",
            "creativity",
            "threat",
            "ict_index",
        ],
    )

    fpl["player_name"] = (fpl["first_name"].fillna("") + " " + fpl["second_name"].fillna("")).str.strip()
    fpl["player_key"] = fpl["player_name"].map(norm_text)
    fpl["team_key"] = fpl["name"].map(norm_text)

    rename_map = {c: f"FPL_{c}" for c in fpl.columns if c not in ["player_key", "team_key"]}
    fpl = fpl.rename(columns=rename_map)
    return fpl


def fetch_understat_league_players(season):
    with UnderstatClient() as understat:
        return understat.league(league="EPL").get_player_data(season=str(season))


def build_understat_df(players):
    us = pd.DataFrame(players)
    to_num(
        us,
        [
            "id",
            "games",
            "time",
            "goals",
            "xG",
            "assists",
            "xA",
            "shots",
            "key_passes",
            "yellow_cards",
            "red_cards",
            "npg",
            "npxG",
            "xGChain",
            "xGBuildup",
        ],
    )
    us["player_key"] = us["player_name"].map(norm_text)
    us["team_key_raw"] = us["team_title"].map(norm_text)
    return us


def map_understat_teams_to_fpl(us, fpl):
    fpl_team_keys = sorted(fpl["team_key"].dropna().unique().tolist())
    us_team_keys = sorted(us["team_key_raw"].dropna().unique().tolist())
    team_map = {}
    for utk in us_team_keys:
        match = get_close_matches(utk, fpl_team_keys, n=1, cutoff=0.60)
        team_map[utk] = match[0] if match else utk
    us = us.copy()
    us["team_key"] = us["team_key_raw"].map(lambda x: team_map.get(x, x))
    map_df = pd.DataFrame({"understat_team_key": list(team_map.keys()), "mapped_fpl_team_key": list(team_map.values())})
    return us, map_df


def build_master(
    season,
    prefer_fpl_for=None,
    master_col_order=None,
):
    if prefer_fpl_for is None:
        prefer_fpl_for = {"minutes", "goals", "assists", "yellow_cards", "red_cards"}

    if master_col_order is None:
        master_col_order = [
            "player_name",
            "team",
            "team_short",
            "position",
            "fpl_id",
            "understat_id",
            "merge_status",
            "minutes",
            "starts",
            "games",
            "goals",
            "assists",
            "goals_p90",
            "assists_p90",
            "xG",
            "xA",
            "xG_p90",
            "xA_p90",
            "npg",
            "npxG",
            "npg_p90",
            "npxG_p90",
            "shots",
            "shots_p90",
            "key_passes",
            "key_passes_p90",
            "yellow_cards",
            "red_cards",
            "yellow_cards_p90",
            "red_cards_p90",
            "clean_sheets",
            "clean_sheets_p90",
            "goals_conceded",
            "saves",
            "saves_p90",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "xGChain",
            "xGBuildup",
        ]

    fpl_data = fetch_fpl_bootstrap()
    fpl = build_fpl_df(fpl_data)

    us_players = fetch_understat_league_players(season)
    us = build_understat_df(us_players)
    us, team_map_df = map_understat_teams_to_fpl(us, fpl)

    us = us.rename(columns={c: f"US_{c}" for c in us.columns if c not in ["player_key", "team_key"]})
    raw = fpl.merge(us, on=["player_key", "team_key"], how="left", indicator=True)
    raw["merge_status"] = raw["_merge"].map({"both": "matched", "left_only": "no_understat_match", "right_only": "no_fpl_match"})
    raw = raw.drop(columns=["_merge"])

    understat_only = us.merge(fpl[["player_key", "team_key"]], on=["player_key", "team_key"], how="left", indicator=True)
    understat_only = understat_only[understat_only["_merge"] == "left_only"].drop(columns=["_merge"])

    master = pd.DataFrame()

    master["player_name"] = coalesce(raw.get("FPL_player_name"), raw.get("US_player_name"))
    master["team"] = coalesce(raw.get("FPL_name"), raw.get("US_team_title"))
    master["team_short"] = raw.get("FPL_short_name")
    master["position"] = coalesce(raw.get("FPL_singular_name_short"), raw.get("US_position"))
    master["fpl_id"] = raw.get("FPL_id")
    master["understat_id"] = raw.get("US_id")
    master["merge_status"] = raw.get("merge_status")

    fpl_minutes = raw.get("FPL_minutes")
    us_minutes = raw.get("US_time")
    master["minutes"] = coalesce(fpl_minutes, us_minutes) if "minutes" in prefer_fpl_for else coalesce(us_minutes, fpl_minutes)
    master["starts"] = raw.get("FPL_starts")
    master["games"] = raw.get("US_games")

    def pick_box(stat_name, fpl_col, us_col):
        f = raw.get(fpl_col)
        u = raw.get(us_col)
        if stat_name in prefer_fpl_for:
            return coalesce(f, u)
        return coalesce(u, f)

    master["goals"] = pick_box("goals", "FPL_goals_scored", "US_goals")
    master["assists"] = pick_box("assists", "FPL_assists", "US_assists")
    master["yellow_cards"] = pick_box("yellow_cards", "FPL_yellow_cards", "US_yellow_cards")
    master["red_cards"] = pick_box("red_cards", "FPL_red_cards", "US_red_cards")

    master["clean_sheets"] = raw.get("FPL_clean_sheets")
    master["goals_conceded"] = raw.get("FPL_goals_conceded")
    master["saves"] = raw.get("FPL_saves")
    master["influence"] = raw.get("FPL_influence")
    master["creativity"] = raw.get("FPL_creativity")
    master["threat"] = raw.get("FPL_threat")
    master["ict_index"] = raw.get("FPL_ict_index")

    master["xG"] = raw.get("US_xG")
    master["xA"] = raw.get("US_xA")
    master["npg"] = raw.get("US_npg")
    master["npxG"] = raw.get("US_npxG")
    master["shots"] = raw.get("US_shots")
    master["key_passes"] = raw.get("US_key_passes")
    master["xGChain"] = raw.get("US_xGChain")
    master["xGBuildup"] = raw.get("US_xGBuildup")

    to_num(
        master,
        [
            "minutes",
            "starts",
            "games",
            "goals",
            "assists",
            "yellow_cards",
            "red_cards",
            "clean_sheets",
            "goals_conceded",
            "saves",
            "xG",
            "xA",
            "npg",
            "npxG",
            "shots",
            "key_passes",
            "xGChain",
            "xGBuildup",
            "influence",
            "creativity",
            "threat",
            "ict_index",
        ],
    )

    for col in ["goals", "assists", "yellow_cards", "red_cards", "clean_sheets", "saves", "shots", "key_passes", "xG", "xA", "npg", "npxG"]:
        if col in master.columns:
            master[f"{col}_p90"] = per90(master[col], master["minutes"])

    ordered = [c for c in master_col_order if c in master.columns]
    extras = [c for c in master.columns if c not in ordered]
    master = master[ordered + extras].copy()

    sort_cols = [c for c in ["team_short", "position", "player_name"] if c in master.columns]
    if sort_cols:
        master = master.sort_values(sort_cols, na_position="last")

    unmatched_fpl = raw[raw["merge_status"] != "matched"].copy()
    return {
        "master": master,
        "raw": raw,
        "unmatched_fpl": unmatched_fpl,
        "understat_only": understat_only,
        "team_mapping": team_map_df,
    }


def write_excel(frames, path):
    path = str(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frames["master"].to_excel(writer, sheet_name="MASTER", index=False)
        frames["raw"].to_excel(writer, sheet_name="RAW_MERGED", index=False)
        frames["unmatched_fpl"].to_excel(writer, sheet_name="UNMATCHED_FPL", index=False)
        frames["understat_only"].to_excel(writer, sheet_name="UNDERSTAT_ONLY", index=False)
        frames["team_mapping"].to_excel(writer, sheet_name="TEAM_MAPPING", index=False)
    return path
