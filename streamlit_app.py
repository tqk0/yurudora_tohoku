import os
import streamlit as st
from yurudora_tohoku_rag import TohokuSpotDoc, generate_yurudora_plans

st.set_page_config(
    page_title="ゆるドラ東北 - AI東北ドライブ診断",
    page_icon="🚗",
    layout="wide",
)

# Streamlit Cloud の Secrets から環境変数へ読み込み（ローカル実行時はそのまま os.environ を使用）
for key in ("GEMINI_API_KEY", "GOOGLE_MAPS_API_KEY"):
    if key in st.secrets and not os.environ.get(key):
        os.environ[key] = str(st.secrets[key])

# 東北スポット知識ベース（実際はJSONやDB/RAGストアから読み込み）
SPOT_DATABASE = [
    TohokuSpotDoc(
        id="sendai-east-hub",
        name="仙台駅東口 レンタカーエリア",
        prefecture="宮城県",
        lat=38.2601,
        lng=140.8856,
        min_road_width_meters=14.0,
        has_mountain_pass=False,
        parking_capacity=80,
        parking_difficulty="easy",
        average_cost_yen=5200,
        estimated_stay_minutes=20,
        description="交通量の多い西口を避け、広い東口大通りから出発できる拠点。",
    ),
    TohokuSpotDoc(
        id="michinoeki-murata",
        name="道の駅 村田",
        prefecture="宮城県",
        lat=38.1189,
        lng=140.7244,
        min_road_width_meters=9.5,
        has_mountain_pass=False,
        parking_capacity=155,
        parking_difficulty="easy",
        average_cost_yen=900,
        estimated_stay_minutes=45,
        description="平地155台の大型無料駐車場を備え、初心者でも休憩しやすい道の駅。",
    ),
    TohokuSpotDoc(
        id="kawamachi-yuriage",
        name="かわまちてらす閖上",
        prefecture="宮城県",
        lat=38.1762,
        lng=140.9531,
        min_road_width_meters=11.0,
        has_mountain_pass=False,
        parking_capacity=240,
        parking_difficulty="easy",
        average_cost_yen=1800,
        estimated_stay_minutes=70,
        description="240台の広大な無料駐車場を備えた水辺のグルメスポット。",
    ),
    TohokuSpotDoc(
        id="zao-farm-kitchen",
        name="蔵王山麓 広々ファームレストラン",
        prefecture="宮城県",
        lat=38.1058,
        lng=140.6352,
        min_road_width_meters=8.5,
        has_mountain_pass=False,
        parking_capacity=120,
        parking_difficulty="easy",
        average_cost_yen=2600,
        estimated_stay_minutes=75,
        description="急な山道手前の平坦エリアにあり大型駐車場完備のご当地グルメ施設。",
    ),
    TohokuSpotDoc(
        id="akiu-onsen-valet",
        name="秋保温泉 バレー駐車対応宿",
        prefecture="宮城県",
        lat=38.2248,
        lng=140.7204,
        min_road_width_meters=8.5,
        has_mountain_pass=False,
        parking_capacity=110,
        parking_difficulty="easy",
        average_cost_yen=15500,
        estimated_stay_minutes=120,
        description="玄関でスタッフに鍵を預けるだけで駐車が不要な温泉宿。",
    ),
    # 安全フィルタの動作確認用（自動除外される例）
    TohokuSpotDoc(
        id="excluded-zao-echoline",
        name="蔵王エコーライン山頂（御釜直通ルート）",
        prefecture="宮城県",
        lat=38.1375,
        lng=140.4496,
        min_road_width_meters=5.0,
        has_mountain_pass=True,
        parking_capacity=150,
        parking_difficulty="moderate",
        average_cost_yen=1000,
        estimated_stay_minutes=60,
        description="急勾配・連続ヘアピンカーブの山岳道路。",
    ),
]

st.title("🚗 ゆるドラ東北")
st.caption("狭い道・急な山道・停めにくい駐車場を100%除外するAI協働ドライブ診断")

with st.form("diagnosis_form"):
    col1, col2 = st.columns(2)
    with col1:
        skill = st.selectbox(
            "運転の自信",
            ["全く自信なし（まっすぐ走るだけで精一杯）", "駐車が特に苦手", "普通だが山道・細い道はNG"],
        )
        duration = st.selectbox("日程", ["日帰り", "1泊2日", "2泊3日"])
    with col2:
        focus = st.selectbox("旅のテーマ", ["食事・ご当地グルメ重視", "体験重視", "絶景・癒やし重視"])
        budget = st.selectbox("ご予算", ["1万〜2万円（コスパ重視）", "2万〜4万円", "4万円以上"])

    notes = st.text_input("自由リクエスト（任意）", placeholder="例：海鮮が食べたい、休憩多め")
    submitted = st.form_submit_button("松・竹・梅プランを生成する", type="primary")

if submitted:
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY が設定されていません。Streamlit Secrets を確認してください。")
    else:
        diagnosis_summary = (
            f"運転自信: {skill} / 日程: {duration} / テーマ: {focus} / 予算: {budget} / メモ: {notes or 'なし'}"
        )
        with st.spinner("安全フィルタとGemini Structured Outputでプランを生成中..."):
            result = generate_yurudora_plans(diagnosis_summary, SPOT_DATABASE)

        st.success(
            f"安全確認済み：難所 {len(result.excluded_spot_names)}件（{', '.join(result.excluded_spot_names)}）を事前除外しました。"
        )

        cols = st.columns(3)
        for col, plan in zip(cols, result.plans):
            with col:
                st.subheader(plan.tier_title)
                st.markdown(f"**{plan.plan_title}**")
                st.metric("予算目安", f"¥{plan.total_budget_yen:,}")
                st.metric("運転合計", f"{plan.total_drive_minutes}分")
                st.write(plan.summary)

                with st.expander("タイムラインを見る"):
                    for idx, stop in enumerate(plan.stops, start=1):
                        if stop.leg_from_previous:
                            st.caption(
                                f"🚗 移動 {stop.leg_from_previous.duration_minutes}分 "
                                f"({stop.leg_from_previous.distance_km}km) · {stop.leg_from_previous.road_guidance}"
                            )
                        st.markdown(f"**{idx}. {stop.spot_name}**（{stop.arrival_time}〜{stop.departure_time}）")
                        st.caption(f"🅿️ {stop.parking_reassurance}")
                        st.info(f"💡 {stop.ai_driver_tip}")
