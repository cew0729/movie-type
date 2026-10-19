
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==================================================
# 1. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 특성을 바탕으로 비슷한 영화끼리 묶고, "
    "군집 수와 군집의 뚜렷한 정도를 분석합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

FEATURES = [
    "스크린 수(로그)",
    "누적 관객(로그)",
    "10위권 일수",
    "롱런 지수"
]

TYPE_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

# ==================================================
# 2. 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd",
        "movieNm",
        "openDt",
        "genre",
        "nation",
        "first_scrn",
        "first_show",
        "first_date",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    return data


try:
    original_df = load_data()
except Exception as error:
    st.error("영화 데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()

total_movies = len(original_df)
df = original_df.copy()

# ==================================================
# 3. 숫자 데이터 정리
# ==================================================
numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

df = df.dropna(
    subset=numeric_columns + ["movieNm"]
).copy()

# 로그 및 나눗셈에 필요한 값 검사
df = df[
    (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["first_week_audi"] > 0)
    & (df["days_in_top10"] >= 0)
].copy()

# ==================================================
# 4. 분석에 사용할 네 가지 속성 생성
# ==================================================

# 스크린 수와 누적 관객: 상용로그
df["스크린 수(로그)"] = np.log10(df["first_scrn"])
df["누적 관객(로그)"] = np.log10(df["total_audi"])

# 10위권 일수: 원래 값
df["10위권 일수"] = df["days_in_top10"]

# 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

# 전체 편수 및 분석 대상 편수
st.markdown(
    f"**전체 영화: {total_movies:,}편　|　"
    f"분석 대상 영화: {len(df):,}편**"
)

st.caption(
    "필요한 값이 없거나 스크린 수, 누적 관객, 첫 주 관객이 "
    "0 이하인 영화는 분석에서 제외했습니다."
)

if len(df) < 8:
    st.error("군집 수를 7개까지 비교하려면 분석 대상 영화가 최소 8편 필요합니다.")
    st.stop()

# ==================================================
# 5. 군집 분석 속성 선택
# ==================================================
st.divider()
st.subheader("1. 영화 유형을 나눌 속성 선택")

selected_features = st.multiselect(
    "군집 분석에 사용할 속성을 두 개 이상 선택하세요.",
    options=FEATURES,
    default=FEATURES
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()

# ==================================================
# 6. 선택한 속성 표준화
# ==================================================
X = df[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================================
# 7. 묶음 수 선택
# ==================================================
st.divider()
st.subheader("2. 묶음 수 선택")

selected_k = st.selectbox(
    "영화를 몇 개의 유형으로 나눌까요?",
    options=[2, 3, 4, 5, 6, 7],
    index=1,
    format_func=lambda k: f"{k}개 묶음",
    key="selected_k"
)

st.write(
    f"현재 선택: **{selected_k}개 묶음**"
)

# ==================================================
# 8. 선택한 묶음 수로 K-평균 군집 분석
# ==================================================
@st.cache_data
def run_kmeans(X_array, k):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_array)
    return labels, float(model.inertia_)


try:
    labels, selected_sse = run_kmeans(
        X_scaled,
        selected_k
    )
except Exception as error:
    st.error("군집 분석 중 오류가 발생했습니다.")
    st.code(str(error))
    st.stop()

df["군집번호"] = labels

# ==================================================
# 9. 군집 기호 배정
# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ...
# ==================================================
cluster_order = (
    df.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_to_type = {
    cluster_id: TYPE_SYMBOLS[position]
    for position, cluster_id in enumerate(cluster_order)
}

df["영화 유형"] = df["군집번호"].map(cluster_to_type)

type_order = [
    cluster_to_type[cluster_id]
    for cluster_id in cluster_order
]

# ==================================================
# 10. 실루엣 점수
# ==================================================
if 2 <= len(np.unique(labels)) < len(df):
    try:
        silhouette = silhouette_score(
            X_scaled,
            labels
        )
        st.markdown(
            f"**현재 {selected_k}개 묶음의 실루엣 점수: "
            f"{silhouette:.4f}**"
        )
        st.caption(
            "실루엣 점수는 -1에서 1 사이입니다. "
            "1에 가까울수록 각 영화가 자기 유형 안에 잘 모여 있고 "
            "다른 유형과 구분되는 경향이 큽니다. "
            "0에 가까우면 유형 사이의 경계가 겹칠 수 있고, "
            "음수이면 일부 영화가 다른 유형에 더 가까울 수 있습니다."
        )
    except Exception as error:
        st.warning(f"실루엣 점수를 계산하지 못했습니다: {error}")
else:
    st.warning(
        "실루엣 점수를 계산하려면 둘 이상의 유형과 "
        "각 유형에 속한 영화가 필요합니다."
    )

# ==================================================
# 11. 2차원 산점도
# ==================================================
st.divider()
st.subheader("3. 2차원 산점도")

st.write(
    "축에 사용할 속성을 선택하세요. "
    "점에 마우스를 올리면 영화 제목을 확인할 수 있습니다."
)

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축(X)",
        options=FEATURES,
        index=0,
        key="scatter_x"
    )

with col_y:
    y_feature = st.selectbox(
        "세로축(Y)",
        options=FEATURES,
        index=1,
        key="scatter_y"
    )

fig2 = px.scatter(
    df,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    category_orders={"영화 유형": type_order},
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True,
        "롱런 지수": ":.2f"
    },
    labels={
        "영화 유형": "영화 유형",
        "스크린 수(로그)": "스크린 수 (상용로그)",
        "누적 관객(로그)": "누적 관객 (상용로그)",
        "10위권 일수": "10위권 일수 (일)",
        "롱런 지수": "롱런 지수"
    },
    title=f"영화 유형별 분포 ({selected_k}개 묶음)"
)

fig2.update_traces(
    marker=dict(size=7, opacity=0.75)
)

fig2.update_layout(
    height=550,
    legend_title_text="영화 유형"
)

st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# 12. 3차원 산점도
# ==================================================
st.divider()
st.subheader("4. 3차원 산점도")

st.write(
    "세 축의 속성을 선택하세요. 마우스로 드래그하면 "
    "그래프를 회전할 수 있습니다."
)

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 군집 분석에 사용할 속성을 "
        "세 개 이상 선택해 주세요."
    )
else:
    col_x3, col_y3, col_z3 = st.columns(3)

    with col_x3:
        x3 = st.selectbox(
            "X축 속성",
            options=selected_features,
            index=0,
            key="x3"
        )

    with col_y3:
        y3 = st.selectbox(
            "Y축 속성",
            options=selected_features,
            index=1,
            key="y3"
        )

    with col_z3:
        z3 = st.selectbox(
            "Z축 속성",
            options=selected_features,
            index=2,
            key="z3"
        )

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        category_orders={"영화 유형": type_order},
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            "first_scrn": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "롱런 지수": ":.2f"
        },
        labels={
            "영화 유형": "영화 유형",
            "스크린 수(로그)": "스크린 수 (상용로그)",
            "누적 관객(로그)": "누적 관객 (상용로그)",
            "10위권 일수": "10위권 일수 (일)",
            "롱런 지수": "롱런 지수"
        },
        title=f"영화 유형별 3차원 분포 ({selected_k}개 묶음)"
    )

    fig3.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig3.update_layout(
        height=650,
        legend_title_text="영화 유형"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ==================================================
# 13. 유형별 편수 및 네 속성의 평균
# ==================================================
st.divider()
st.subheader("5. 영화 유형별 요약")

st.caption(
    "평균은 로그 변환 전 원래 단위로 계산합니다. "
    "유형 기호는 누적 관객 평균이 높은 순서입니다."
)

summary = (
    df.groupby("영화 유형")
    .agg(
        영화_편수=("movieNm", "size"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        십위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("롱런 지수", "mean")
    )
    .reindex(type_order)
    .reset_index()
)

summary = summary.rename(
    columns={
        "영화 유형": "영화 유형",
        "영화_편수": "영화 편수",
        "스크린_수_평균": "스크린 수 평균",
        "누적_관객_평균": "누적 관객 평균",
        "십위권_일수_평균": "10위권 일수 평균",
        "롱런_지수_평균": "롱런 지수 평균"
    }
)

display_summary = summary.copy()

display_summary["스크린 수 평균"] = (
    display_summary["스크린 수 평균"].map(
        lambda value: f"{value:,.1f}"
    )
)

display_summary["누적 관객 평균"] = (
    display_summary["누적 관객 평균"].map(
        lambda value: f"{value:,.0f}"
    )
)

display_summary["10위권 일수 평균"] = (
    display_summary["10위권 일수 평균"].round(1)
)

display_summary["롱런 지수 평균"] = (
    display_summary["롱런 지수 평균"].round(2)
)

st.dataframe(
    display_summary,
    hide_index=True,
    use_container_width=True
)

# ==================================================
# 14. 유형별 누적 관객 상위 5편
# ==================================================
st.divider()
st.subheader("6. 영화 유형별 누적 관객 상위 5편")

st.write(
    "각 유형 안에서 누적 관객이 많은 영화 다섯 편을 보여 줍니다."
)

for label in type_order:
    group_df = (
        df[df["영화 유형"] == label]
        .sort_values(
            by=["total_audi", "movieNm"],
            ascending=[False, True]
        )
        .head(5)
    )

    st.markdown(f"### {label} 유형")

    if group_df.empty:
        st.write("해당 유형에 영화가 없습니다.")
    else:
        top5 = group_df[["movieNm", "total_audi"]].copy()

        top5 = top5.rename(
            columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객"
            }
        )

        top5["누적 관객"] = top5["누적 관객"].map(
            lambda value: f"{value:,.0f}명"
        )

        st.dataframe(
            top5.reset_index(drop=True),
            hide_index=True,
            use_container_width=True
        )

# ==================================================
# 15. 묶음 수에 따른 SSE 계산
# ==================================================
st.divider()
st.subheader("7. 묶음 수에 따른 군집 내 제곱합")

st.write(
    "SSE는 각 영화와 해당 군집 중심 사이의 거리 제곱을 "
    "모두 더한 값입니다. 값이 작을수록 군집 내부의 거리가 "
    "작다는 뜻이지만, 묶음 수가 증가하면 대체로 감소합니다."
)

@st.cache_data
def calculate_sse(X_array):
    results = []

    for k in range(1, 8):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        model.fit(X_array)

        results.append({
            "묶음 수": k,
            "SSE": float(model.inertia_)
        })

    return pd.DataFrame(results)


try:
    sse_df = calculate_sse(X_scaled)

    # 직전 묶음 수의 SSE - 현재 묶음 수의 SSE
    # 첫 번째 값은 비교할 이전 값이 없으므로 결측값
    sse_df["직전 값 대비 감소량"] = (
        sse_df["SSE"].shift(1) - sse_df["SSE"]
    )

    fig_sse = go.Figure()

    fig_sse.add_trace(
        go.Scatter(
            x=sse_df["묶음 수"],
            y=sse_df["SSE"],
            mode="lines+markers",
            name="SSE",
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate=(
                "묶음 수: %{x}<br>"
                "SSE: %{y:,.2f}<extra></extra>"
            )
        )
    )

    # 현재 선택한 묶음 수 위치에 세로선
    fig_sse.add_vline(
        x=selected_k,
        line_dash="dash",
        line_width=2,
        annotation_text=f"현재 선택: {selected_k}개",
        annotation_position="top"
    )

    fig_sse.update_layout(
        title="묶음 수와 SSE의 관계",
        xaxis_title="묶음 수",
        yaxis_title="군집 내 제곱합 (SSE)",
        xaxis=dict(
            tickmode="linear",
            tick0=1,
            dtick=1
        ),
        height=450,
        margin=dict(l=20, r=20, t=70, b=20)
    )

    st.plotly_chart(
        fig_sse,
        use_container_width=True
    )

    st.caption(
        "꺾은선에서 감소 폭이 눈에 띄게 작아지는 지점은 "
        "묶음 수를 늘리는 효과가 줄어들기 시작하는 후보일 수 있습니다. "
        "이 그래프만으로 최적의 묶음 수가 자동 확정되는 것은 아닙니다."
    )

    # 표에 표시할 값 정리
    display_sse = sse_df.copy()

    display_sse["SSE"] = display_sse["SSE"].map(
        lambda value: f"{value:,.2f}"
    )

    display_sse["직전 값 대비 감소량"] = (
        display_sse["직전 값 대비 감소량"].map(
            lambda value: "" if pd.isna(value)
            else f"{value:,.2f}"
        )
    )

    st.dataframe(
        display_sse,
        hide_index=True,
        use_container_width=True
    )

except Exception as error:
    st.error("SSE 계산 중 오류가 발생했습니다.")
    st.code(str(error))

# ==================================================
# 16. 분석 방법 안내
# ==================================================
st.divider()
st.subheader("분석 방법")

st.markdown(
    """
    1. **데이터 정리:** 분석에 필요한 값이 없는 영화와
       첫 주 관객이 0인 영화 등을 제외합니다.
    2. **속성 변환:** 스크린 수와 누적 관객은 상용로그를 적용합니다.
       롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며,
       20을 초과하면 20으로 제한합니다.
    3. **표준화:** 현재 선택한 속성을 평균 0, 표준편차 1로 변환합니다.
    4. **K-평균:** 선택한 속성으로 지정된 묶음 수만큼 군집화합니다.
    5. **기호 지정:** 누적 관객 평균이 높은 군집부터
       ㉮, ㉯, ㉰, ㉱, ㉲, ㉳, ㉴를 붙입니다.
    6. **SSE:** 1~7개 묶음의 군집 내 제곱합을 계산합니다.
    7. **실루엣 점수:** 현재 선택한 묶음의 응집도와 분리 정도를 평가합니다.

    **주의:** 군집은 선택한 속성에 따른 상대적인 유사성을 나타냅니다.
    특정 기호가 모든 면에서 우수하다는 의미는 아닙니다.
    """
)
