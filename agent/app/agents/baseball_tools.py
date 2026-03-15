import json
from datetime import datetime
from typing import Optional

from langchain.tools import tool
from pybaseball import (
    batting_stats,
    pitching_stats,
    schedule_and_record,
    statcast_batter,
    statcast_pitcher,
    playerid_lookup,
)

# 주요 팀 약어 매핑
TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}


def _lookup_player_id(player_name: str) -> Optional[int]:
    """선수 이름으로 MLBAM ID를 조회합니다."""
    parts = player_name.strip().split()
    if len(parts) < 2:
        return None
    first = parts[0]
    last = " ".join(parts[1:])
    try:
        result = playerid_lookup(last, first, fuzzy=True)
        if result.empty:
            return None
        # 가장 최근에 활동한 선수를 우선 선택
        result = result.sort_values("mlb_played_last", ascending=False)
        return int(result.iloc[0]["key_mlbam"])
    except Exception:
        return None


def _resolve_team_abbr(team_name: str) -> str:
    """팀 이름을 약어로 변환합니다. 이미 약어면 그대로 반환."""
    upper = team_name.upper().strip()
    # 이미 약어인 경우
    if len(upper) <= 4 and upper in TEAM_ABBR.values():
        return upper
    # 풀네임에서 찾기
    for full, abbr in TEAM_ABBR.items():
        if full.lower() in team_name.lower():
            return abbr
    # 부분 매칭 (예: "Dodgers" → "LAD")
    for full, abbr in TEAM_ABBR.items():
        if team_name.lower() in full.lower() or full.split()[-1].lower() == team_name.lower():
            return abbr
    return upper


@tool
def search_player_stats(
    player_name: str,
    season: Optional[int] = None,
) -> str:
    """MLB 선수의 시즌 타격 또는 투수 스탯을 조회합니다.
    타자: AVG, OBP, SLG, OPS, HR, RBI, WAR 등
    투수: ERA, WHIP, K/9, W, L, SV, WAR 등

    Args:
        player_name: 선수 영문 이름 (예: "Shohei Ohtani")
        season: 조회할 시즌 연도 (예: 2024). 미입력 시 최근 시즌
    """
    if season is None:
        season = datetime.now().year

    try:
        # 타격 스탯 조회
        bat_df = batting_stats(season, season, qual=0)
        bat_match = bat_df[bat_df["Name"].str.contains(player_name, case=False, na=False)]

        # 투수 스탯 조회
        pitch_df = pitching_stats(season, season, qual=0)
        pitch_match = pitch_df[pitch_df["Name"].str.contains(player_name, case=False, na=False)]

        results = {}

        if not bat_match.empty:
            row = bat_match.iloc[0]
            results["batting"] = {
                "Name": row.get("Name", ""),
                "Team": row.get("Team", ""),
                "G": int(row.get("G", 0)),
                "PA": int(row.get("PA", 0)),
                "AB": int(row.get("AB", 0)),
                "H": int(row.get("H", 0)),
                "HR": int(row.get("HR", 0)),
                "RBI": int(row.get("RBI", 0)),
                "R": int(row.get("R", 0)),
                "SB": int(row.get("SB", 0)),
                "BB": int(row.get("BB", 0)),
                "SO": int(row.get("SO", 0)),
                "AVG": round(float(row.get("AVG", 0)), 3),
                "OBP": round(float(row.get("OBP", 0)), 3),
                "SLG": round(float(row.get("SLG", 0)), 3),
                "OPS": round(float(row.get("OPS", 0)), 3),
                "WAR": round(float(row.get("WAR", 0)), 1),
            }

        if not pitch_match.empty:
            row = pitch_match.iloc[0]
            results["pitching"] = {
                "Name": row.get("Name", ""),
                "Team": row.get("Team", ""),
                "W": int(row.get("W", 0)),
                "L": int(row.get("L", 0)),
                "ERA": round(float(row.get("ERA", 0)), 2),
                "G": int(row.get("G", 0)),
                "GS": int(row.get("GS", 0)),
                "IP": round(float(row.get("IP", 0)), 1),
                "SO": int(row.get("SO", 0)),
                "BB": int(row.get("BB", 0)),
                "WHIP": round(float(row.get("WHIP", 0)), 2),
                "K/9": round(float(row.get("K/9", 0)), 1),
                "FIP": round(float(row.get("FIP", 0)), 2),
                "WAR": round(float(row.get("WAR", 0)), 1),
            }

        if not results:
            return json.dumps({"error": f"'{player_name}'에 해당하는 선수를 {season}시즌에서 찾을 수 없습니다."}, ensure_ascii=False)

        results["season"] = season
        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"스탯 조회 중 오류 발생: {str(e)}"}, ensure_ascii=False)


@tool
def compare_players(
    player_name_1: str,
    player_name_2: str,
    season: Optional[int] = None,
) -> str:
    """두 MLB 선수의 스탯을 비교합니다.

    Args:
        player_name_1: 첫 번째 선수 영문 이름 (예: "Shohei Ohtani")
        player_name_2: 두 번째 선수 영문 이름 (예: "Aaron Judge")
        season: 비교할 시즌 연도. 미입력 시 최근 시즌
    """
    if season is None:
        season = datetime.now().year

    try:
        bat_df = batting_stats(season, season, qual=0)
        pitch_df = pitching_stats(season, season, qual=0)

        comparison = {"season": season, "players": {}}

        for name in [player_name_1, player_name_2]:
            player_data = {}

            bat_match = bat_df[bat_df["Name"].str.contains(name, case=False, na=False)]
            if not bat_match.empty:
                row = bat_match.iloc[0]
                player_data["batting"] = {
                    "Name": row.get("Name", ""),
                    "Team": row.get("Team", ""),
                    "G": int(row.get("G", 0)),
                    "PA": int(row.get("PA", 0)),
                    "HR": int(row.get("HR", 0)),
                    "RBI": int(row.get("RBI", 0)),
                    "AVG": round(float(row.get("AVG", 0)), 3),
                    "OBP": round(float(row.get("OBP", 0)), 3),
                    "SLG": round(float(row.get("SLG", 0)), 3),
                    "OPS": round(float(row.get("OPS", 0)), 3),
                    "WAR": round(float(row.get("WAR", 0)), 1),
                }

            pitch_match = pitch_df[pitch_df["Name"].str.contains(name, case=False, na=False)]
            if not pitch_match.empty:
                row = pitch_match.iloc[0]
                player_data["pitching"] = {
                    "Name": row.get("Name", ""),
                    "Team": row.get("Team", ""),
                    "W": int(row.get("W", 0)),
                    "L": int(row.get("L", 0)),
                    "ERA": round(float(row.get("ERA", 0)), 2),
                    "IP": round(float(row.get("IP", 0)), 1),
                    "SO": int(row.get("SO", 0)),
                    "WHIP": round(float(row.get("WHIP", 0)), 2),
                    "WAR": round(float(row.get("WAR", 0)), 1),
                }

            if not player_data:
                player_data = {"error": f"'{name}' 선수를 {season}시즌에서 찾을 수 없습니다."}

            comparison["players"][name] = player_data

        return json.dumps(comparison, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"비교 조회 중 오류 발생: {str(e)}"}, ensure_ascii=False)


@tool
def search_game_results(
    team: str,
    season: Optional[int] = None,
) -> str:
    """MLB 팀의 경기 결과를 조회합니다.

    Args:
        team: 팀 이름 또는 약어 (예: "LAD", "Dodgers", "Los Angeles Dodgers")
        season: 조회할 시즌 연도. 미입력 시 현재 시즌
    """
    if season is None:
        season = datetime.now().year

    try:
        team_abbr = _resolve_team_abbr(team)
        df = schedule_and_record(season, team_abbr)

        if df.empty:
            return json.dumps({"error": f"'{team}'의 {season}시즌 경기 기록을 찾을 수 없습니다."}, ensure_ascii=False)

        # 최근 10경기만 반환 (전체는 너무 길 수 있음)
        recent = df.tail(10)
        games = []
        for _, row in recent.iterrows():
            games.append({
                "Date": str(row.get("Date", "")),
                "Opponent": str(row.get("Opp", "")),
                "Result": str(row.get("W/L", "")),
                "Runs_Scored": str(row.get("R", "")),
                "Runs_Allowed": str(row.get("RA", "")),
                "Record": str(row.get("W-L", "")),
                "Winning_Pitcher": str(row.get("Win", "")),
                "Losing_Pitcher": str(row.get("Loss", "")),
            })

        total_record = str(df.iloc[-1].get("W-L", "")) if not df.empty else ""

        result = {
            "team": team_abbr,
            "season": season,
            "total_record": total_record,
            "recent_games": games,
            "note": "최근 10경기 결과입니다.",
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"경기 결과 조회 중 오류 발생: {str(e)}"}, ensure_ascii=False)


@tool
def search_statcast(
    player_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """MLB 선수의 Statcast 상세 데이터를 조회합니다.
    구속, 회전수, 타구속도, 발사각도 등 심층 분석 데이터를 제공합니다.

    Args:
        player_name: 선수 영문 이름 (예: "Shohei Ohtani")
        start_date: 조회 시작일 (YYYY-MM-DD). 미입력 시 해당 시즌 시작일
        end_date: 조회 종료일 (YYYY-MM-DD). 미입력 시 해당 시즌 종료일
    """
    try:
        player_id = _lookup_player_id(player_name)
        if player_id is None:
            return json.dumps({"error": f"'{player_name}' 선수의 ID를 찾을 수 없습니다."}, ensure_ascii=False)

        now = datetime.now()
        if start_date is None:
            start_date = f"{now.year}-03-01"
        if end_date is None:
            end_date = f"{now.year}-11-30"

        # 타자 데이터 시도
        batter_df = statcast_batter(start_date, end_date, player_id)
        # 투수 데이터 시도
        pitcher_df = statcast_pitcher(start_date, end_date, player_id)

        result = {"player": player_name, "period": f"{start_date} ~ {end_date}"}

        if batter_df is not None and not batter_df.empty:
            result["batter_statcast"] = {
                "total_pitches_faced": len(batter_df),
                "avg_launch_speed": round(float(batter_df["launch_speed"].dropna().mean()), 1) if "launch_speed" in batter_df.columns and not batter_df["launch_speed"].dropna().empty else None,
                "avg_launch_angle": round(float(batter_df["launch_angle"].dropna().mean()), 1) if "launch_angle" in batter_df.columns and not batter_df["launch_angle"].dropna().empty else None,
                "max_launch_speed": round(float(batter_df["launch_speed"].dropna().max()), 1) if "launch_speed" in batter_df.columns and not batter_df["launch_speed"].dropna().empty else None,
                "avg_hit_distance": round(float(batter_df["hit_distance_sc"].dropna().mean()), 1) if "hit_distance_sc" in batter_df.columns and not batter_df["hit_distance_sc"].dropna().empty else None,
            }
            # 구종별 피안타 분석
            if "pitch_type" in batter_df.columns:
                pitch_counts = batter_df["pitch_type"].value_counts().head(5)
                result["batter_statcast"]["pitch_type_faced"] = {str(k): int(v) for k, v in pitch_counts.items()}

        if pitcher_df is not None and not pitcher_df.empty:
            result["pitcher_statcast"] = {
                "total_pitches_thrown": len(pitcher_df),
                "avg_release_speed": round(float(pitcher_df["release_speed"].dropna().mean()), 1) if "release_speed" in pitcher_df.columns and not pitcher_df["release_speed"].dropna().empty else None,
                "max_release_speed": round(float(pitcher_df["release_speed"].dropna().max()), 1) if "release_speed" in pitcher_df.columns and not pitcher_df["release_speed"].dropna().empty else None,
                "avg_spin_rate": round(float(pitcher_df["release_spin_rate"].dropna().mean()), 0) if "release_spin_rate" in pitcher_df.columns and not pitcher_df["release_spin_rate"].dropna().empty else None,
            }
            # 구종별 분석
            if "pitch_type" in pitcher_df.columns:
                pitch_breakdown = {}
                for pt in pitcher_df["pitch_type"].dropna().unique():
                    subset = pitcher_df[pitcher_df["pitch_type"] == pt]
                    pitch_breakdown[str(pt)] = {
                        "count": len(subset),
                        "avg_speed": round(float(subset["release_speed"].dropna().mean()), 1) if not subset["release_speed"].dropna().empty else None,
                        "avg_spin": round(float(subset["release_spin_rate"].dropna().mean()), 0) if "release_spin_rate" in subset.columns and not subset["release_spin_rate"].dropna().empty else None,
                    }
                result["pitcher_statcast"]["pitch_breakdown"] = pitch_breakdown

        if "batter_statcast" not in result and "pitcher_statcast" not in result:
            return json.dumps({"error": f"'{player_name}'의 Statcast 데이터를 찾을 수 없습니다."}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Statcast 조회 중 오류 발생: {str(e)}"}, ensure_ascii=False)


# 에이전트에 등록할 도구 목록
baseball_tools = [
    search_player_stats,
    compare_players,
    search_game_results,
    search_statcast,
]
