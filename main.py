
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.markdown(
    "영화의 스크린 수, 누적 관객, 10위권 일수, 롱런 지수를 "
    "분석해 비슷한 특징을 가진 영화끼리 세 유형으로 나눕니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    return df


try:
    original_df = load_data()
except Exception as e:
    st.error("영화 데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.info("데이터 주소와 CSV 파일 상태를 확인해 주세요.")
    st.stop()

total_movies = len(original_df)

# --------------------------------------------------
# 3. 숫자 데이터 정리 및 분석 속성 만들기
# --------------------------------------------------
df = original_df.copy()

numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in numeric_columns:
    # 쉼표가 포함된 숫자도 처리
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

# 분석에 사용할 원래 속성이 모두 있는 영화만 사용
df = df.dropna(subset=numeric_columns).copy()

# 로그 계산과 롱런 지수 계산에 필요한 값 확인
df = df[
    (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["first_week_audi"] > 0)
    & (df["days_in_top10"] >= 0)
].copy()

# 상용로그(log10) 변환
df["스크린 수(로그)"] = np.log10(df["first_scrn"])
df["누적 관객(로그)"] = np.log10(df["total_audi"])

# 10위권 일수는 원래 값 사용
df["10위권 일수"] = df["days_in_top10"]

# 롱런 지수 = 누적 관객 / 첫 주 관객
# 최대 20으로 제한
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

# 원래 단위의 네 속성
original_features = [
    "first_scrn",
    "total_audi",
    "days_in_top10",
    "롱런 지수"
]

# 화면 표시용 이름
original_feature_names = {
    "first_scrn": "스크린 수",
    "total_audi": "누적 관객",
    "days_in_top10": "10위권 일수",
    "롱런 지수": "롱런 지수"
}

# K-평균 및 산점도에 사용할 네 속성
feature_names = [
    "스크린 수(로그)",
    "누적 관객(로그)",
    "10위권 일수",
    "롱런 지수"
]

# 한 줄로 전체 영화 수와 분석 대상 영화 수 표시
st.markdown(
    f"**전체 영화: {total_movies:,}편　|　"
    f"분석 대상 영화: {len(df):,}편**"
)

st.caption(
    "분석 대상은 네 속성의 계산에 필요한 값이 모두 있고, "
    "첫 주 관객·스크린 수·누적 관객이 양수인 영화입니다."
)

if len(df) < 3:
    st.error("세 유형으로 나누려면 분석 대상 영화가 최소 3편 필요합니다.")
    st.stop()

# --------------------------------------------------
# 4. 군집 분석에 사용할 속성 선택
# --------------------------------------------------
st.subheader("1. 영화 유형을 나눌 속성 선택")

selected_features = st.multiselect(
    "K-평균 분석에 사용할 속성을 선택하세요. 두 개 이상 골라 주세요.",
    options=feature_names,
    default=feature_names
)

if len(selected_features) < 2:
    st.warning("분석 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# 선택한 속성만 사용해 표준화
X = df[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 난수와 초기 중심점 설정을 고정해 결과 재현
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["군집_원번호"] = kmeans.fit_predict(X_scaled)

# --------------------------------------------------
# 5. 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰ 지정
# --------------------------------------------------
cluster_order = (
    df.groupby("군집_원번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_order[0]: "㉮",
    cluster_order[1]: "㉯",
    cluster_order[2]: "㉰"
}

df["영화 유형"] = df["군집_원번호"].map(cluster_labels)

# 표시 순서도 ㉮, ㉯, ㉰로 고정
type_order = ["㉮", "㉯", "㉰"]

# --------------------------------------------------
# 6. 2차원 산점도
# --------------------------------------------------
st.divider()
st.subheader("2. 2차원 산점도")
st.write(
    "가로축과 세로축의 속성을 바꾸어 영화 유형별 분포를 살펴보세요. "
    "점에 마우스를 올리면 영화 제목을 확인할 수 있습니다."
)

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축(X)",
        options=feature_names,
        index=0,
        key="scatter_x"
    )

with col_y:
    y_default = 1 if x_feature != feature_names[1] else 0
    y_feature = st.selectbox(
        "세로축(Y)",
        options=feature_names,
        index=y_default,
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
        "롱런 지수": ":.2f",
        x_feature: ":.3f",
        y_feature: ":.3f"
    },
    labels={
        "영화 유형": "영화 유형",
        "스크린 수(로그)": "스크린 수 (log10)",
        "누적 관객(로그)": "누적 관객 (log10)",
        "10위권 일수": "10위권 일수 (일)",
        "롱런 지수": "롱런 지수"
    },
    title="영화 유형별 분포"
)

fig2.update_traces(marker={"size": 7, "opacity": 0.75})
fig2.update_layout(
    height=550,
    legend_title_text="영화 유형",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# 7. 3차원 산점도
# --------------------------------------------------
st.divider()
st.subheader("3. 3차원 산점도")

st.write(
    "세 축의 속성을 선택해 영화 유형을 입체적으로 살펴보세요. "
    "그래프를 마우스로 드래그하면 회전할 수 있습니다."
)

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 위에서 군집 분석에 사용할 속성을 "
        "세 개 이상 선택해 주세요."
    )
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축",
            options=selected_features,
            index=0,
            key="x3"
        )

    with c2:
        y3 = st.selectbox(
            "Y축",
            options=selected_features,
            index=1,
            key="y3"
        )

    with c3:
        z3 = st.selectbox(
            "Z축",
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
            "스크린 수(로그)": "스크린 수 (log10)",
            "누적 관객(로그)": "누적 관객 (log10)",
            "10위권 일수": "10위권 일수 (일)",
            "롱런 지수": "롱런 지수"
        },
        title="영화 유형별 3차원 분포"
    )

    fig3.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )

    fig3.update_layout(
        height=650,
        legend_title_text="영화 유형",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# 8. 유형별 편수와 원래 단위의 평균
# --------------------------------------------------
st.divider()
st.subheader("4. 영화 유형별 요약")

st.caption(
    "㉮는 누적 관객 평균이 가장 높은 유형이고, "
    "㉯와 ㉰는 그다음 순서입니다. "
    "아래 평균은 로그 변환 전의 원래 단위로 계산합니다."
)

summary = (
    df.groupby("영화 유형")
    .agg(
        영화_편수=("movieNm", "size"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        10위권_일수_평균=("days_in_top10", "mean"),
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
        "10위권_일수_평균": "10위권 일수 평균",
        "롱런_지수_평균": "롱런 지수 평균"
    }
)

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
    display_summary["스크린 수 평균"].map(
        lambda x: f"{x:,.1f}"
    )
)
display_summary["누적 관객 평균"] = (
    display_summary["누적 관객 평균"].map(
        lambda x: f"{x:,.0f}"
    )
)

st.dataframe(
    display_summary,
    hide_index=True,
    use_container_width=True
)

# --------------------------------------------------
# 9. 유형별 누적 관객 상위 5편
# --------------------------------------------------
st.divider()
st.subheader("5. 영화 유형별 누적 관객 상위 5편")

st.write(
    "각 유형 안에서 누적 관객이 많은 영화부터 다섯 편씩 보여 줍니다."
)

for label in type_order:
    group_df = (
        df[df["영화 유형"] == label]
        .sort_values(
            ["total_audi", "movieNm"],
            ascending=[False, True]
        )
        .head(5)
    )

    st.markdown(f"### {label} 유형")

    if group_df.empty:
        st.write("해당 유형에 영화가 없습니다.")
        continue

    top5 = group_df[["movieNm", "total_audi"]].copy()
    top5.columns = ["영화 제목", "누적 관객"]

    top5["누적 관객"] = top5["누적 관객"].map(
        lambda x: f"{x:,.0f}명"
    )

    st.dataframe(
        top5.reset_index(drop=True),
        hide_index=True,
        use_container_width=True
    )

# --------------------------------------------------
# 10. 분석 방법 안내
# --------------------------------------------------
st.divider()
st.subheader("분석 방법")

st.markdown(
    """
    1. **데이터 정리:** 분석 속성에 필요한 값이 없거나
       첫 주 관객이 0인 영화 등을 제외합니다.
    2. **속성 변환:** 스크린 수와 누적 관객은 상용로그를 취합니다.
       롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며,
       20을 초과하면 20으로 제한합니다.
    3. **표준화:** 선택한 속도를 평균 0, 표준편차 1의 척도로 바꿉니다.
    4. **K-평균:** 세 유형으로 군집화하며 난수 시드를 고정합니다.
    5. **유형 이름 지정:** 군집별 누적 관객 평균이 큰 순서로
       ㉮, ㉯, ㉰를 지정합니다.

    **주의:** 군집은 데이터에서 발견한 상대적인 유사성입니다.
    ㉮가 모든 면에서 더 우수하다는 뜻은 아니며,
    선택한 속성에 따라 유형과 결과가 달라질 수 있습니다.
    """
)
