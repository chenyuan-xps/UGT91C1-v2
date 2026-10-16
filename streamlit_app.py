"""UGT91C1 突变体活性预测平台（Streamlit 入口）。

启动：

    streamlit run streamlit_app.py

页面逻辑放在 app_pages/，共享业务逻辑放在 prediction_core.py。
"""

import streamlit as st

st.set_page_config(
    page_title="UGT91C1 突变体活性预测平台",
    page_icon=":material/biotech:",
    layout="wide",
)

# 预测结果跨页面保留，供「候选预测」页在交互后继续展示。
st.session_state.setdefault("predict_result", None)

pages = st.navigation(
    [
        st.Page("app_pages/overview.py", title="概览", icon=":material/dashboard:", default=True),
        st.Page("app_pages/recommend.py", title="候选推荐", icon=":material/rocket_launch:"),
        st.Page("app_pages/predict.py", title="候选预测", icon=":material/science:"),
        st.Page("app_pages/ingest.py", title="数据入库", icon=":material/database_upload:"),
        st.Page("app_pages/compare.py", title="对比分析", icon=":material/compare_arrows:"),
        st.Page("app_pages/results.py", title="结果浏览", icon=":material/table_chart:"),
        st.Page("app_pages/model_doc.py", title="模型与数据", icon=":material/menu_book:"),
    ],
    position="top",
)

st.title(pages.title)
pages.run()
