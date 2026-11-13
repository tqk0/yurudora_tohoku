"""
ゆるドラ東北 - Python Reference Implementation
================================================
1. Gemini Structured Output (google-genai + Pydantic)
2. Google Maps Routes API v2 による実移動時間・道路検証（ハルシネーション防止）
3. 東北地方スポット特化型 RAG（安全ハードフィルタ + ベクトル類似検索）

必要パッケージ:
  pip install google-genai pydantic requests numpy
"""
#web公開用
pip install streamlit
#web公開用

from __future__ import annotations

import math
import os
from typing import Literal, Optional

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# 1. Pydantic Structured Output スキーマ定義
# ============================================================

class RouteLeg(BaseModel):
    from_spot_id: str
    to_spot_id: str
    distance_km: float
    duration_minutes: int
    road_guidance: str = Field(description="片側2車線バイパス中心など安心運転のポイント")
    verification_source: Literal["google_maps_routes_api", "haversine_fallback"]


class TimelineStop(BaseModel):
    spot_id: str
    spot_name: str
    arrival_time: str = Field(description="HH:MM形式")
    departure_time: str = Field(description="HH:MM形式")
    stay_minutes: int
    parking_reassurance: str = Field(description="駐車場の広さ・停めやすさの安心コメント")
    ai_driver_tip: str = Field(description="運転が苦手な方に寄り添う具体的なアドバイス")
    estimated_cost_yen: int
    leg_from_previous: Optional[RouteLeg] = None


class TieredDrivePlan(BaseModel):
    tier: Literal["matsu", "take", "ume"]
    tier_title: str = Field(description="松（最適）・竹（運転最簡単）・梅（最安）")
    plan_title: str
    total_budget_yen: int
    total_drive_minutes: int
    summary: str
    stops: list[TimelineStop]


class TohokuTripPlansResponse(BaseModel):
    safety_premise_summary: str
    excluded_spot_names: list[str]
    plans: list[TieredDrivePlan] = Field(min_length=3, max_length=3)


# ============================================================
# 2. 人の設計による安全ハードフィルタ + RAG知識ベース
# ============================================================

class TohokuSpotDoc(BaseModel):
    id: str
    name: str
    prefecture: str
    lat: float
    lng: float
    min_road_width_meters: float
    has_mountain_pass: bool
    parking_capacity: int
    parking_difficulty: Literal["easy", "moderate", "hard"]
    average_cost_yen: int
    estimated_stay_minutes: int
    description: str


def filter_safe_spots(
    docs: list[TohokuSpotDoc],
    require_easy_parking: bool = True,
) -> tuple[list[TohokuSpotDoc], list[str]]:
    """
    【人の役割：安全境界の定義】
    LLMに候補を渡す前に、狭い道・急な山道・停めにくい駐車場を100%除外する。
    """
    safe: list[TohokuSpotDoc] = []
    excluded: list[str] = []

    for doc in docs:
        is_narrow = doc.min_road_width_meters < 5.5
        is_mountain = doc.has_mountain_pass
        is_hard_parking = doc.parking_difficulty == "hard" or (
            require_easy_parking and doc.parking_capacity < 40
        )

        if is_narrow or is_mountain or is_hard_parking:
            excluded.append(doc.name)
        else:
            safe.append(doc)

    return safe, excluded


def retrieve_top_k_rag_spots(
    safe_docs: list[TohokuSpotDoc],
    query_text: str,
    client: genai.Client,
    top_k: int = 8,
) -> list[TohokuSpotDoc]:
    """
    【RAG拡張点】
    gemini-embedding-2 を用いて東北特化型スポットDBからクエリに最適な候補を検索する。
    """
    if len(safe_docs) <= top_k:
        return safe_docs

    texts = [query_text] + [
        f"{d.prefecture} {d.name} {d.description} 駐車{d.parking_capacity}台"
        for d in safe_docs
    ]
    embed_res = client.models.embed_content(
        model="gemini-embedding-2",
        contents=texts,
    )
    vectors = [e.values for e in embed_res.embeddings]
    q_vec = vectors[0]

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    scored = sorted(
        zip(safe_docs, vectors[1:]),
        key=lambda pair: cosine(q_vec, pair[1]),
        reverse=True,
    )
    return [doc for doc, _ in scored[:top_k]]


# ============================================================
# 3. Google Maps Routes API による移動時間の実測検証
# ============================================================

def fetch_verified_drive_leg(
    origin: TohokuSpotDoc,
    destination: TohokuSpotDoc,
    maps_api_key: Optional[str],
) -> RouteLeg:
    """
    生成AIの移動時間ハルシネーションを防ぐため、Google Maps Routes API v2 で
    実測の所要時間と距離を取得して上書きする。
    """
    if maps_api_key:
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": maps_api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
        }
        payload = {
            "origin": {"location": {"latLng": {"latitude": origin.lat, "longitude": origin.lng}}},
            "destination": {"location": {"latLng": {"latitude": destination.lat, "longitude": destination.lng}}},
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            "routeModifiers": {"avoidTolls": False, "avoidHighways": False},
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.ok and resp.json().get("routes"):
            route = resp.json()["routes"][0]
            duration_sec = int(route["duration"].rstrip("s"))
            distance_km = round(route["distanceMeters"] / 1000.0, 1)
            return RouteLeg(
                from_spot_id=origin.id,
                to_spot_id=destination.id,
                distance_km=distance_km,
                duration_minutes=max(5, round(duration_sec / 60)),
                road_guidance=f"実測検証済みルート（最小道路幅 {min(origin.min_road_width_meters, destination.min_road_width_meters)}m）",
                verification_source="google_maps_routes_api",
            )

    # フォールバック計算（Haversine + 東北バイパス平均速度42km/h）
    lat1, lon1, lat2, lon2 = map(math.radians, [origin.lat, origin.lng, destination.lat, destination.lng])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    km = round(6371 * 2 * math.asin(math.sqrt(a)) * 1.25, 1)
    minutes = max(10, round((km / 42.0) * 60 / 5) * 5)
    return RouteLeg(
        from_spot_id=origin.id,
        to_spot_id=destination.id,
        distance_km=km,
        duration_minutes=minutes,
        road_guidance="広いバイパス・平坦幹線中心ルート",
        verification_source="haversine_fallback",
    )


# ============================================================
# 4. Gemini Structured Output による松竹梅プラン生成オーケストレーション
# ============================================================

def generate_yurudora_plans(
    diagnosis_summary: str,
    spot_database: list[TohokuSpotDoc],
) -> TohokuTripPlansResponse:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    maps_key = os.environ.get("GOOGLE_MAPS_API_KEY")

    # Step 1: 狭い道・難所を100%除外
    safe_spots, excluded_names = filter_safe_spots(spot_database, require_easy_parking=True)

    # Step 2: RAGで診断内容に合う東北スポットを抽出
    rag_spots = retrieve_top_k_rag_spots(safe_spots, diagnosis_summary, client, top_k=8)
    spot_by_id = {s.id: s for s in rag_spots}

    prompt = f"""
あなたは運転が苦手な方に寄り添う東北ドライブ旅行プランナーです。
以下の「安全検証済み東北スポット候補」のみを使用して、松・竹・梅の3プランをJSONで生成してください。

【診断結果】
{diagnosis_summary}

【安全検証済み候補スポット（これ以外は使用禁止）】
{[s.model_dump() for s in rag_spots]}

【松竹梅の定義】
- matsu（松）: 診断の希望（食事/体験/癒やし・予算）に最も合う最適プラン
- take（竹）: 運転距離と右左折が最も少なく、大型駐車場のみを使う超安心プラン
- ume（梅）: 交通費・入場料を抑えた高コスパプラン
"""

    response = client.models.generate_content(
        model="gemini-4-hydrogen",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TohokuTripPlansResponse,
        ),
    )

    plans_obj = TohokuTripPlansResponse.model_validate_json(response.text)
    plans_obj.excluded_spot_names = excluded_names

    # Step 3: 移動時間を Google Maps Routes API で実測検証して上書き（ハルシネーション防止）
    for plan in plans_obj.plans:
        total_drive = 0
        for idx, stop in enumerate(plan.stops):
            if idx > 0:
                prev_spot = spot_by_id.get(plan.stops[idx - 1].spot_id)
                curr_spot = spot_by_id.get(stop.spot_id)
                if prev_spot and curr_spot:
                    verified_leg = fetch_verified_drive_leg(prev_spot, curr_spot, maps_key)
                    stop.leg_from_previous = verified_leg
                    total_drive += verified_leg.duration_minutes
        plan.total_drive_minutes = total_drive

    return plans_obj
