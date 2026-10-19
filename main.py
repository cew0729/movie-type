
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    "영화의 여러 특성을 분석해 비슷한 특징을 가진 영화끼리 "
    "세 가지 유형으로 나누는 프로그램입니다."
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

TYPE_ORDER = ["㉮", "㉯", "㉰"]

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

# 필요한 값이 없는 영화 제외
df = df.dropna(
    subset=numeric_columns + ["movieNm"]
).copy()

# 로그와 나눗셈을 위해 양수 조건 확인
df = df[
    (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["first_week_audi"] > 0)
    & (df["days_in_top10"] >= 0)
].copy()

# ==================================================
# 4. 네 가지 분석 속성 만들기
# ==================================================

# 스크린 수와 누적 관객은 상용로그
df["스크린 수(로그)"] = np.log10(df["first_scrn"])
df["누적 관객(로그)"] = np.log10(df["total_audi"])

# 10위권 일수는 원래 값 그대로 사용
df["10위권 일수"] = df["days_in_top10"]

# 롱런 지수 = 누적 관객 / 첫 주 관객
# 20을 넘으면 20으로 제한
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

# 전체 영화 수와 분석 대상 수
st.markdown(
    f"**전체 영화: {total_movies:,}편　|　"
    f"분석 대상 영화: {len(df):,}편**"
)

st.caption(
    "필요한 값이 없거나 스크린 수, 누적 관객, 첫 주 관객이 "
    "0 이하인 영화는 분석에서 제외했습니다."
)

if len(df) < 3:
    st.error("세 유형으로 나누려면 분석 대상 영화가 최소 3편 필요합니다.")
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
# 6. 표준화 및 K-평균 군집 분석
# ==================================================
X = df[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["군집번호"] = kmeans.fit_predict(X_scaled)

# ==================================================
# 7. 누적 관객 평균 순서로 ㉮, ㉯, ㉰ 지정
# ==================================================
cluster_means = (
    df.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
)

cluster_order = cluster_means.index.tolist()

cluster_to_type = {
    cluster_order[0]: "㉮",
    cluster_order[1]: "㉯",
    cluster_order[2]: "㉰"
}

df["영화 유형"] = df["군집번호"].map(cluster_to_type)

# ==================================================
# 8. 2차원 산점도
# ==================================================
st.divider()
st.subheader("2. 2차원 산점도")

st.write(
    "가로축과 세로축에 사용할 속성을 선택하세요. "
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
    category_orders={"영화 유형": TYPE_ORDER},
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
    title="영화 유형별 분포"
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
# 9. 3차원 산점도
# ==================================================
st.divider()
st.subheader("3. 3차원 산점도")

st.write(
    "세 축의 속성을 선택하세요. 그래프를 마우스로 드래그하면 "
    "시점을 회전할 수 있습니다."
)

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 위에서 군집 분석 속성을 "
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
        category_orders={"영화 유형": TYPE_ORDER},
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
        title="영화 유형별 3차원 분포"
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
# 10. 유형별 편수와 네 속성의 평균
# ==================================================
st.divider()
st.subheader("4. 영화 유형별 요약")

st.caption(
    "평균은 로그 변환 전의 원래 단위로 계산합니다. "
    "㉮부터 누적 관객 평균이 높은 순서입니다."
)

summary = (
    df.groupby("영화 유형")
    .agg(
        영화_편수=("movieNm", "size"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        십위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("롱런 지수", "mean"),
    )
    .reindex(TYPE_ORDER)
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

# 표시용 표와 숫자 형식 정리
display_summary = summary.copy()

display_summary["스크린 수 평균"] = (
    display_summary["스크린 수 평균"].round(1)
)
display_summary["누적 관객 평균"] = (
    display_summary["누적 관객 평균"].round(0)
)
display_summary["10위권 일수 평균"] = (
    display_summary["10위권 일수 평균"].round(1)
)
display_summary["롱런 지수 평균"] = (
    display_summary["롱런 지수 평균"].round(2)
)

display_summary["스크린 수 평균"] = (
    display_summary["스크린 수 평균"]
    .map(lambda value: f"{value:,.1f}")
)
display_summary["누적 관객 평균"] = (
    display_summary["누적 관객 평균"]
    .map(lambda value: f"{value:,.0f}")
)

st.dataframe(
    display_summary,
    hide_index=True,
    use_container_width=True
)

# ==================================================
# 11. 유형별 누적 관객 상위 5편
# ==================================================
st.divider()
st.subheader("5. 영화 유형별 누적 관객 상위 5편")

st.write(
    "각 유형에서 누적 관객이 많은 영화 다섯 편을 보여 줍니다."
)

for label in TYPE_ORDER:
    group_df = df[df["영화 유형"] == label].copy()

    group_df = group_df.sort_values(
        by=["total_audi", "movieNm"],
        ascending=[False, True]
    ).head(5)

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
# 12. 분석 방법 설명
# ==================================================
st.divider()
st.subheader("분석 방법")

st.markdown(
    """
    - **데이터 정리:** 필요한 속성에 결측값이 있거나
      첫 주 관객이 0 이하인 영화 등을 제외합니다.
    - **상용로그 변환:** 스크린 수와 누적 관객에 log10을 적용합니다.
    - **롱런 지수:** 누적 관객을 첫 주 관객으로 나누고
      20을 초과하면 20으로 제한합니다.
    - **표준화:** 선택한 속성을 평균 0, 표준편차 1로 변환합니다.
    - **K-평균:** 선택한 속성을 사용해 세 군집으로 나눕니다.
    - **유형 이름:** 누적 관객 평균이 높은 군집부터
      ㉮, ㉯, ㉰로 표시합니다.

    군집은 선택한 속성에 따른 상대적인 유사성을 나타냅니다.
    ㉮ 유형이 모든 면에서 우수하다는 의미는 아닙니다.
    """
)
