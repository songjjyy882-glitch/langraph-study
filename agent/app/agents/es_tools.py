import json
from typing import Optional

from elasticsearch import Elasticsearch
from langchain.tools import tool

from app.core.config import settings


def _get_es_client() -> Elasticsearch:
    """Elasticsearch 클라이언트를 반환합니다."""
    if settings.ES is None:
        raise ValueError("Elasticsearch 설정이 없습니다. .env에 ES__URL, ES__USERNAME, ES__PASSWORD를 설정하세요.")
    return Elasticsearch(
        settings.ES.URL,
        basic_auth=(settings.ES.USERNAME, settings.ES.PASSWORD),
    )


@tool
def search_historical_batting(
    player_name: Optional[str] = None,
    team: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_hr: Optional[int] = None,
    sort_by: Optional[str] = "HR",
    limit: int = 10,
) -> str:
    """MLB 역대 타격 통계를 검색합니다 (1871~2021). 선수명, 팀, 연도, 홈런 수 등 조건으로 검색할 수 있습니다.

    Args:
        player_name: 선수 영문 이름 (예: "Babe Ruth", "Shohei Ohtani")
        team: 팀 약어 (예: "NYA"=양키스, "LAN"=다저스, "BOS"=보스턴)
        year_from: 검색 시작 연도 (예: 2000)
        year_to: 검색 종료 연도 (예: 2021)
        min_hr: 최소 홈런 수 필터 (예: 40)
        sort_by: 정렬 기준 필드 (HR, H, RBI, R, BB, SB 등). 기본값: HR
        limit: 반환할 최대 결과 수. 기본값: 10
    """
    try:
        es = _get_es_client()
        must_clauses = []

        if player_name:
            parts = player_name.strip().split()
            if len(parts) >= 2:
                must_clauses.append({"match": {"nameFirst": parts[0]}})
                must_clauses.append({"match": {"nameLast": " ".join(parts[1:])}})
            else:
                must_clauses.append({
                    "multi_match": {
                        "query": player_name,
                        "fields": ["nameFirst", "nameLast"],
                    }
                })

        if team:
            must_clauses.append({"term": {"teamID": team.upper()}})

        if year_from or year_to:
            range_filter = {}
            if year_from:
                range_filter["gte"] = year_from
            if year_to:
                range_filter["lte"] = year_to
            must_clauses.append({"range": {"yearID": range_filter}})

        if min_hr:
            must_clauses.append({"range": {"HR": {"gte": min_hr}}})

        query = {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}

        res = es.search(
            index="mlb-batting",
            query=query,
            sort=[{sort_by: {"order": "desc"}}],
            size=limit,
        )

        hits = []
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            hits.append({
                "player": f"{s.get('nameFirst', '')} {s.get('nameLast', '')}",
                "year": s.get("yearID"),
                "team": s.get("teamID"),
                "G": s.get("G"),
                "AB": s.get("AB"),
                "H": s.get("H"),
                "HR": s.get("HR"),
                "RBI": s.get("RBI"),
                "R": s.get("R"),
                "BB": s.get("BB"),
                "SO": s.get("SO"),
                "SB": s.get("SB"),
            })

        if not hits:
            return json.dumps({"error": "조건에 맞는 타격 기록을 찾을 수 없습니다."}, ensure_ascii=False)

        return json.dumps({
            "total_found": res["hits"]["total"]["value"],
            "results": hits,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"ES 타격 검색 중 오류: {str(e)}"}, ensure_ascii=False)


@tool
def search_historical_pitching(
    player_name: Optional[str] = None,
    team: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    max_era: Optional[float] = None,
    sort_by: Optional[str] = "SO",
    limit: int = 10,
) -> str:
    """MLB 역대 투구 통계를 검색합니다 (1871~2021). 선수명, 팀, 연도, ERA 등 조건으로 검색할 수 있습니다.

    Args:
        player_name: 선수 영문 이름 (예: "Clayton Kershaw", "Nolan Ryan")
        team: 팀 약어 (예: "NYA"=양키스, "LAN"=다저스, "BOS"=보스턴)
        year_from: 검색 시작 연도 (예: 2000)
        year_to: 검색 종료 연도 (예: 2021)
        max_era: 최대 ERA 필터 (예: 3.0 이하만 검색)
        sort_by: 정렬 기준 필드 (SO, W, SV, ERA 등). 기본값: SO
        limit: 반환할 최대 결과 수. 기본값: 10
    """
    try:
        es = _get_es_client()
        must_clauses = []

        if player_name:
            parts = player_name.strip().split()
            if len(parts) >= 2:
                must_clauses.append({"match": {"nameFirst": parts[0]}})
                must_clauses.append({"match": {"nameLast": " ".join(parts[1:])}})
            else:
                must_clauses.append({
                    "multi_match": {
                        "query": player_name,
                        "fields": ["nameFirst", "nameLast"],
                    }
                })

        if team:
            must_clauses.append({"term": {"teamID": team.upper()}})

        if year_from or year_to:
            range_filter = {}
            if year_from:
                range_filter["gte"] = year_from
            if year_to:
                range_filter["lte"] = year_to
            must_clauses.append({"range": {"yearID": range_filter}})

        if max_era:
            must_clauses.append({"range": {"ERA": {"lte": max_era}}})

        query = {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}

        # ERA 정렬은 오름차순 (낮을수록 좋음)
        sort_order = "asc" if sort_by == "ERA" else "desc"

        res = es.search(
            index="mlb-pitching",
            query=query,
            sort=[{sort_by: {"order": sort_order}}],
            size=limit,
        )

        hits = []
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            hits.append({
                "player": f"{s.get('nameFirst', '')} {s.get('nameLast', '')}",
                "year": s.get("yearID"),
                "team": s.get("teamID"),
                "W": s.get("W"),
                "L": s.get("L"),
                "ERA": s.get("ERA"),
                "G": s.get("G"),
                "GS": s.get("GS"),
                "SV": s.get("SV"),
                "SO": s.get("SO"),
                "BB": s.get("BB"),
                "H": s.get("H"),
            })

        if not hits:
            return json.dumps({"error": "조건에 맞는 투구 기록을 찾을 수 없습니다."}, ensure_ascii=False)

        return json.dumps({
            "total_found": res["hits"]["total"]["value"],
            "results": hits,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"ES 투구 검색 중 오류: {str(e)}"}, ensure_ascii=False)


# 에이전트에 등록할 ES 도구 목록
es_tools = [search_historical_batting, search_historical_pitching]
