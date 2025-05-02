import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import ast

from visuals import *
from agent import *


# --- Tab 1: Visualizations ---
def show_home():
    st.header("Lead Snapshot Visualizations")
    if st.session_state["data_snapshots"] is not None:
        df = st.session_state["data_snapshots"]
        # Example: Visualize Sales Funnel
        st.subheader("Lead Funnel Conversion Chart")
        st.altair_chart(funnel_chart(df), use_container_width=True)

        df = st.session_state["data_leads"]
        df = df.reset_index(drop=True)
        df.index = df.index + 1  # For table view

        st.subheader("Lead History")
        columns_to_show = [col for col in df.columns if col not in ["company_summary", "snapshot_summary"]]
        st.dataframe(df[columns_to_show])

        # Selection for more details
        lead_options = [f"{index}. {company}" for index, company in zip(df.index, df["company_name"])]
        selected_company = st.selectbox("Select a lead to view summary", lead_options)
        if selected_company:
            selected_company = selected_company.split(". ", 1)[1]   # Extract company name
            selected_row = df[df["company_name"] == selected_company].iloc[0]
            st.dataframe(selected_row[columns_to_show].astype(str))
            st.markdown(f"**ID:** {selected_row['id']}")
            # st.markdown(f"**Summary:** {selected_row['company_summary']}")
            render_text(f"**Summary:** <br>{selected_row['company_summary']}")
            st.markdown("\n")

            # Display agent option
            if st.button("View in Agent"):
                st.session_state.searching_lead = True
                st.session_state.jumped_searching_lead = True
                st.session_state.jump_to_lead_id = selected_row['id'][-4:] # Get ID last 4 digits
                st.session_state.selected_tab = "💬 Agent"
                # Force Streamlit to rerun and reflect the updated session state
                st.rerun()

    else:
        st.info("No data available to display visualizations.")

# --- Tab 2: Open Leads ---
def show_open_lead_page():
    st.header("Open Leads")
    if st.session_state["data_leads"] is not None:
        df = st.session_state["data_leads"]
        # Filter open leads
        df = df[df["status"] == "Open"].reset_index(drop=True)
        df.index = df.index + 1  # For table view

        st.subheader("Open Lead Current Distribution")
        stage_order = ["Prospecting", "Qualified", "Proposal", "Negotiation"]
        stage_counts = df["lead_stage"].value_counts().reindex(stage_order, fill_value=0).reset_index()
        stage_counts.columns = ["Lead Stage", "Count"]
        stage_counts["Lead Stage"] = pd.Categorical(stage_counts["Lead Stage"], categories=stage_order, ordered=True)
        chart = alt.Chart(stage_counts).mark_bar().encode(
            x=alt.X("Lead Stage", sort=stage_order),
            y="Count",
            tooltip=["Lead Stage", "Count"]
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Open Leads List")
        columns_to_show = [col for col in df.columns if col not in ["company_summary", "snapshot_summary", "converted", "close_date"]]
        st.dataframe(df[columns_to_show])

        # Selection for more details
        lead_options = [f"{index}. {company}" for index, company in zip(df.index, df["company_name"])]
        selected_company = st.selectbox("Select a lead to view summary", lead_options)
        if selected_company:
            selected_company = selected_company.split(". ", 1)[1]   # Extract company name
            selected_row = df[df["company_name"] == selected_company].iloc[0]
            st.dataframe(selected_row[columns_to_show].astype(str))
            st.markdown(f"**ID:** {selected_row['id']}")
            # st.markdown(f"**Summary:** {selected_row['company_summary']}")
            render_text(f"**Summary:** <br>{selected_row['company_summary']}")
            st.markdown("\n")

            # Display agent option
            if st.button("View in Agent"):
                st.session_state.searching_lead = True
                st.session_state.jumped_searching_lead = True
                st.session_state.jump_to_lead_id = selected_row['id'][-4:] # Get ID last 4 digits
                st.session_state.selected_tab = "💬 Agent"
                # Force Streamlit to rerun and reflect the updated session state
                st.rerun()

    else:
        st.info("No open leads to display.")

# --- Tab 3: Chat Agent ---
def show_agent_page():
    st.header("Agent")
    show_agent()




def main():
    st.set_page_config(page_title="Lead Scoring Demo", layout="wide")

    ## Load snapshots dataframe
    file_path = 'snapshots.csv'  # Change this path
    df_snapshots = pd.read_csv(file_path)
    df_snapshots = df_snapshots.drop(['mock_source', 'favorability'], axis=1)
    df_snapshots['embedding'] = df_snapshots['embedding'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float))   # Get embedding into numpy array form

    ## Create leads dataframe
    df_leads = df_snapshots.groupby('id').agg({
        'company_name': 'first',
        'industry': 'first',
        'status': 'first',
        'converted': 'first',
        'open_date': 'first',
        'close_date': 'first',
        'company_summary': 'first',
        'cyber_investment': 'last',
        'competitor_solution': 'last',
        'renewal_date': 'last',
        'deal_value': 'last',
        'snapshot_date': 'last',
        'snapshot_seq': 'last',
        'decision_maker_level': 'last',
        'lead_stage': 'last',
        'downloads': 'sum',
        'website_visits': 'sum',
        'interactions': 'sum',
        'snapshot_summary': 'last'
    }).reset_index()

    # Save to session state
    st.session_state["data_snapshots"] = df_snapshots
    st.session_state["data_leads"] = df_leads

    # Tabs for different sections
    tabs =  ["📊 Home", "📋 Open Leads", "💬 Agent"]
    st.sidebar.title("Navigation")

    # Use a local variable to store the selected radio option
    selected_tab = st.sidebar.radio("Go to", tabs,
    index=tabs.index(st.session_state.get("selected_tab", "📊 Home")))

    # Update session state ONLY if the user changed the tab
    if selected_tab != st.session_state.get("selected_tab"):
        st.session_state.selected_tab = selected_tab
        st.rerun()

    # Selection
    if st.session_state.selected_tab == tabs[0]:
        show_home()
    elif st.session_state.selected_tab == tabs[1]:
        show_open_lead_page()
    elif st.session_state.selected_tab == tabs[2]:
        show_agent_page()

 

# Run the app
if __name__ == "__main__":
    main()