import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import altair as alt
import re


# Render text visuals
def render_text(text):
    # Convert **bold** to <strong>...</strong>
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Convert __italic__ to <em>...</em>
    text = re.sub(r"__(.+?)__", r"<em>\1</em>", text)
    # Convert newlines to <br>
    text = text.replace("\n", "<br>")

    # Render in styled left-aligned block
    st.markdown(
        f"""
        <div style='max-width: 800px; text-align: left; line-height: 1.6; font-size: 1.05rem;'>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

def visualize_stage_timeline(df):
    df1 = df.copy()
    df1['snapshot_date'] = pd.to_datetime(df1['snapshot_date'])
    df1 = df1.sort_values(['id', 'snapshot_date'])      # Sort
    
    # Convert id to string for categorical y-axis
    df1['id_str'] = df1['id'].astype(str)
    
    # Sort and set order for y-axis
    df1 = df1.sort_values('open_date', ascending=False)
    df1['id_str'] = pd.Categorical(df1['id_str'], categories=df1['id_str'].unique(), ordered=True)
    
    # Color map for stages
    color_map = {
        'Prospecting': 'gray',
        'Qualified': 'dodgerblue',
        'Proposal': '#4169E1',
        'Negotiation': 'navy',
        'Converted': 'green',
        'Lost': 'red',
    }
    color_scale = alt.Scale(domain=list(color_map.keys()), range=list(color_map.values()))
    
    # Add engagement score
    df1['engagement'] = df1['downloads'] + df1['website_visits'] + df1['interactions']
    
    # ---- Stage markers on y=lead ID ----
    stage_points = alt.Chart(df1).mark_point(shape='triangle-right', filled=True, size=400).encode(
        x='snapshot_date:T',
        y=alt.Y('id_str:N', title='Lead ID'),
        color=alt.Color('lead_stage:N', scale=color_scale, title='Pipeline Stage'),
        tooltip=['company_name', 'lead_stage', 'snapshot_date']
    )
    
    # ---- Engagement line chart (second Y axis) ----
    engagement_line = alt.Chart(df1).mark_line(interpolate='monotone', strokeDash=[4,2]).encode(
        x='snapshot_date:T',
        y=alt.Y('engagement:Q', title='Engagement Activity', axis=alt.Axis(titleColor='orange')),
        color=alt.value('orange'),
        tooltip=['company_name', 'engagement', 'snapshot_date']
    )
    
    # ---- Combine with layering and dual y-axis ----
    chart = alt.layer(stage_points, engagement_line).resolve_scale(
        y='independent'
    ).properties(
        width=1000,
        height=400,
        title='Lead Stage Progression and Engagement Over Time'
    )
    
    st.altair_chart(chart, use_container_width=False)


def funnel_chart(df):
    df1 = df.copy()
    # Ensure 'snapshot_date' is datetime type and 'lead_stage' is a categorical type
    df1['snapshot_date'] = pd.to_datetime(df1['snapshot_date'])
    df1['lead_stage'] = pd.Categorical(df1['lead_stage'], categories=["Prospecting", "Qualified", "Proposal", "Negotiation", "Converted"], ordered=True)
    
    # Remove "Lost" leads, lost from funnel
    df1 = df1[df1['lead_stage'] != "Lost"]

    # Get the number of leads per stage
    funnel_data = df1.groupby("lead_stage", observed=True)["id"].nunique().reset_index()
    funnel_data.columns = ["Lead Stage", "Lead Count"]

    # Calculate the percentage for each stage based on the total leads
    top_count = funnel_data.loc[funnel_data["Lead Stage"] == "Prospecting", "Lead Count"].values[0]
    funnel_data["Percentage"] = (funnel_data["Lead Count"] / top_count * 100).map(lambda x: f"{x:.1f}%")

    # Create funnel chart with Altair
    funnel_chart = alt.Chart(funnel_data).mark_bar().encode(
        x=alt.X("Lead Count", title="Number of Leads"),
        y=alt.Y("Lead Stage", title="Lead Stage", sort=["Prospecting", "Qualified", "Proposal", "Negotiation", "Converted"], type="nominal"),
        color="Lead Stage",  # Use the default color scale
        tooltip=["Lead Stage", "Lead Count", "Percentage"]
    ).properties(
        height=400  # Adjust the height for better visibility
    )

    # Add text labels (raw count) to the bars with improved visibility
    text = alt.Chart(funnel_data).mark_text(
    align='left',
    baseline='bottom',
    dx=5,
    dy=-5,
    size=14
    ).encode(
    x='Lead Count:Q',
    y=alt.Y('Lead Stage:N', sort=["Prospecting", "Qualified", "Proposal", "Negotiation", "Converted"]),
    text='Lead Count:Q',
    color=alt.value('black')
    )

# Text label for percentage (below bar end)
    percentage_text = alt.Chart(funnel_data).mark_text(
    align='left',
    baseline='top',
    dx=5,
    dy=5,
    size=14
    ).encode(
    x='Lead Count:Q',
    y=alt.Y('Lead Stage:N', sort=["Prospecting", "Qualified", "Proposal", "Negotiation", "Converted"]),
    text="Percentage:N",
    color=alt.value('black')
    )

    # Combine the chart and text labels
    combined_chart = funnel_chart + text + percentage_text

    return combined_chart



def visualize_comparable(subject_name, df_matches):
    """
    Parameters:
    - subject_name (str): Name of the subject lead.
    - df_matches (pd.DataFrame): DataFrame with 'company_name' and 'similarity' columns.
    """

    # Combine subject and matched data
    data = pd.DataFrame({
        "Company": [subject_name] + df_matches["company_name"].tolist(),
        "Similarity": [1.0] + df_matches["similarity"].tolist(),
        "Color": ['#018785'] + ['#023861'] * len(df_matches),
    })

    data["Label"] = data["Similarity"].apply(lambda x: f"{x:.2f}")
    data.reset_index(inplace=True)
    data.rename(columns={"index": "Position"}, inplace=True)

    # Build bar chart
    bars = alt.Chart(data).mark_bar().encode(
        x=alt.X('Position:O', axis=None),
        y=alt.Y('Similarity:Q', scale=alt.Scale(domain=[0, 1.4])),
        color=alt.Color('Color:N', scale=None),
        tooltip=['Company', 'Similarity']
    )

    # Add similarity scores inside bars
    text_inside = alt.Chart(data).mark_text(
        align='center',
        baseline='middle',
        dy = 20,
        color='white',
        fontSize=16 
    ).encode(
        x='Position:O',
        y='Similarity:Q',
        text='Label'
    )

    # Add company names below bars
    text_below = alt.Chart(data).mark_text(
        align='center',
        baseline='top',
        angle=315,
        dx=-20,
        dy=0,
        fontSize=16 
    ).encode(
        x='Position:O',
        y=alt.value(0),
        text='Company'
    )

    chart = (bars + text_inside + text_below).properties(
        width=800,
        height=300,
        title="Comparables by Cosine Similarity"
    ).configure_view(
        stroke=None
    ).configure_axis(
        grid=False,  # Remove horizontal grid lines
        domain=False  # Remove axis line itself (including ticks)
    ).configure_axisX(
        domain=False,  # Remove x-axis line (and ticks)
        ticks=False,  # Remove x-axis ticks
        labels=False
    ).configure_axisY(
        grid=False,  # Remove y-axis grid lines
        ticks=False,  # Optionally remove y-axis ticks if desired
        domain=False,  # Remove y-axis line
        labels=False
    )

    # Display chart in Streamlit
    st.altair_chart(chart, use_container_width=False)