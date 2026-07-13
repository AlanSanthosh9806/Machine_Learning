import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Universal Data Visualizer", layout="wide")

st.title("📊 Universal Dataset Visualizer")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    ###############################
    # Dataset Info
    ###############################

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    st.dataframe(df)

    ###############################
    # Search
    ###############################

    search = st.text_input("Global Search")

    filtered_df = df.copy()

    if search:
        mask = np.column_stack([
            filtered_df[col].astype(str).str.contains(search, case=False, na=False)
            for col in filtered_df.columns
        ])
        filtered_df = filtered_df.loc[mask.any(axis=1)]

    ###############################
    # Sidebar Filters
    ###############################

    st.sidebar.header("Filters")

    for column in filtered_df.columns:

        if pd.api.types.is_numeric_dtype(filtered_df[column]):

            min_val = float(filtered_df[column].min())
            max_val = float(filtered_df[column].max())

            values = st.sidebar.slider(
                column,
                min_val,
                max_val,
                (min_val, max_val)
            )

            filtered_df = filtered_df[
                filtered_df[column].between(values[0], values[1])
            ]

        elif pd.api.types.is_datetime64_any_dtype(filtered_df[column]):

            start, end = st.sidebar.date_input(
                column,
                [filtered_df[column].min(),
                 filtered_df[column].max()]
            )

            filtered_df = filtered_df[
                (filtered_df[column] >= pd.to_datetime(start)) &
                (filtered_df[column] <= pd.to_datetime(end))
            ]

        elif filtered_df[column].nunique() <= 30:

            options = st.sidebar.multiselect(
                column,
                filtered_df[column].dropna().unique(),
                default=filtered_df[column].dropna().unique()
            )

            filtered_df = filtered_df[
                filtered_df[column].isin(options)
            ]

        else:

            txt = st.sidebar.text_input(f"{column} contains")

            if txt:
                filtered_df = filtered_df[
                    filtered_df[column]
                    .astype(str)
                    .str.contains(txt, case=False, na=False)
                ]

    st.header("Filtered Dataset")

    st.write(filtered_df.shape)

    st.dataframe(filtered_df)

    ###############################
    # Statistics
    ###############################

    st.header("Statistics")

    st.dataframe(filtered_df.describe(include="all").T)

    ###############################
    # Missing Values
    ###############################

    st.header("Missing Values")

    miss = filtered_df.isna().sum()

    fig = px.bar(
        x=miss.index,
        y=miss.values,
        labels={"x":"Columns","y":"Missing Count"}
    )

    st.plotly_chart(fig, use_container_width=True)

    ###############################
    # Visualization
    ###############################

    st.header("Visualization")

    chart = st.selectbox(
        "Chart Type",
        [
            "Scatter",
            "Line",
            "Bar",
            "Histogram",
            "Box",
            "Violin",
            "Pie",
            "Heatmap"
        ]
    )

    numeric = filtered_df.select_dtypes(include=np.number).columns.tolist()

    categorical = filtered_df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if chart == "Scatter":

        x = st.selectbox("X", filtered_df.columns)
        y = st.selectbox("Y", numeric)

        color = st.selectbox(
            "Color",
            ["None"] + list(filtered_df.columns)
        )

        fig = px.scatter(
            filtered_df,
            x=x,
            y=y,
            color=None if color=="None" else color
        )

        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Line":

        x = st.selectbox("X", filtered_df.columns)
        y = st.selectbox("Y", numeric)

        fig = px.line(filtered_df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Bar":

        x = st.selectbox("Category", filtered_df.columns)
        y = st.selectbox("Value", numeric)

        fig = px.bar(filtered_df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Histogram":

        x = st.selectbox("Column", numeric)

        bins = st.slider("Bins", 5, 100, 20)

        fig = px.histogram(filtered_df, x=x, nbins=bins)

        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Box":

        y = st.selectbox("Column", numeric)

        fig = px.box(filtered_df, y=y)

        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Violin":

        y = st.selectbox("Column ", numeric)

        fig = px.violin(filtered_df, y=y, box=True)

        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Pie":

        names = st.selectbox("Category", categorical)

        fig = px.pie(filtered_df, names=names)

        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Heatmap":

        corr = filtered_df[numeric].corr()

        fig, ax = plt.subplots(figsize=(10,8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    ###############################
    # Correlation Matrix
    ###############################

    st.header("Correlation Matrix")

    if len(numeric) > 1:

        corr = filtered_df[numeric].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu"
        )

        st.plotly_chart(fig, use_container_width=True)

    ###############################
    # Download
    ###############################

    csv = filtered_df.to_csv(index=False).encode()

    st.download_button(
        "Download Filtered Dataset",
        csv,
        "filtered_data.csv",
        "text/csv"
    )