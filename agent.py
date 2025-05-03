import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import openai
import re


from visuals import visualize_stage_timeline, render_text, visualize_comparable

# Open AI
with open('openai_key.txt', 'r') as f:
    key = f.readline().strip()
client = openai.OpenAI(api_key=key) 


def format_snapshot(row):
    return f"""- Date: {row['snapshot_date']}
    Lead Stage: {row['lead_stage']}
    Lead Deal Value: {row['deal_value']}
    Downloads Engagement: {row['downloads']}
    Website Visits Engagement: {row['website_visits']}
    Interactions Engagement: {row['interactions']}
    Most Senior Contact: {row['decision_maker_level']}

    """

def ask_about_lead(lead_id):    
    df_snapshots = st.session_state["data_snapshots"]
    df = df_snapshots.copy()

    # Get lead snapshots
    df = df[df['id']==lead_id].sort_values('snapshot_date')
    df = df.drop(['company_summary', 'embedding'], axis=1)   # Redundant with snapshot summary

    # Prompt user
    user_question = st.text_input("Ask a question about this lead:", key="lead_question_input")

    # User asks a question
    if user_question:
        # Final snapshot
        final_snapshot = df.iloc[-1].to_dict()
        final_snapshot = "\n".join([f"{key}: {value}" for key, value in final_snapshot.items()])

        # Temporal based features
        history = ''
        for i, row, in df.iterrows():
            history += format_snapshot(row)

        # Build prompt
        prompt = f"""Lead Information: {final_snapshot}

Lead History:
{history}
Question About This Lead: {user_question}
"""
        print(prompt)

        response = client.chat.completions.create(
            model="gpt-4",  # gpt-3.5-turbo likely not perform as well, but much cheaper
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )

        # Extract and display GPT's response
        response = response.choices[0].message.content.strip()
        render_text(f"**Question**: {user_question}")
        st.markdown("\n")
        render_text(f"**Answer**: {response}")
        st.markdown("\n")


#############################################################
#############################################################



def get_similar_leads(subject_embedding, df_snapshots_hist, k=5, threshold=0.7):
    # Ensure the subject_embedding is a 2D array (1 row, N columns)
    subject_embedding = np.array(subject_embedding).reshape(1, -1)
    # Ensure all embeddings are in a 2D array (N rows, M columns)
    all_embeddings = np.vstack(df_snapshots_hist['embedding'].values)
    
    # Compute cosine similarities
    similarities = cosine_similarity(subject_embedding, all_embeddings)[0]
    
    matches = []
    matched_ids = []
    similarity_scores = []
    valid_match_count = 0  # Keep track of the number of valid matches
    
    while valid_match_count < k:
        # Find the index of the highest similarity (most similar)
        top_index = similarities.argmax()
        
        # Retrieve the similarity value for this match
        top_similarity = similarities[top_index]

        # If the similarity value falls below the threshold, break out of the loop
        if top_similarity < threshold:
            break
        
        # Retrieve the id for the current match
        matched_id = df_snapshots_hist.iloc[top_index]['id']
        
        # If this id has already been matched, set similarity to a very low value
        # to avoid selecting it again.
        if matched_id in matched_ids:
            similarities[top_index] = -1  # Set to low to avoid selecting it again
            continue
        
        # Add the id to the matched list
        matched_ids.append(matched_id)
        # Add the corresponding row to the matches
        matches.append(df_snapshots_hist.iloc[top_index])
        # Add similarity score
        similarity_scores.append(top_similarity)
        # Increment the valid match count
        valid_match_count += 1
        
        # Set the similarity value of this lead to -1 so it’s not selected again
        similarities[top_index] = -1  # Make sure this lead is excluded from future iterations
    
    # Convert the list of matches (rows) into a DataFrame
    df_matches = pd.DataFrame(matches)
    df_matches['similarity'] = similarity_scores  # Add similarity column
    
    return df_matches



def score_lead(lead_id):
    df_snapshots = st.session_state["data_snapshots"]
    df = df_snapshots.copy()

    # Get subject embedding. # Filter by id, sort, get latest
    subject_name = df[df['id'] == lead_id].sort_values('snapshot_date').iloc[-1]['company_name']
    subject_summary = df[df['id'] == lead_id].sort_values('snapshot_date').iloc[-1]['snapshot_summary']
    subject_embedding = df[df['id'] == lead_id].sort_values('snapshot_date').iloc[-1]['embedding']

    # Historical comparables only
    df = df[df_snapshots['status'] == 'Closed']
    # Get k most similar
    df_matches = get_similar_leads(subject_embedding, df, k=5, threshold=0.6)

    # Visualize Comparable
    visualize_comparable(subject_name, df_matches)

    # Get comparable summaries and labels
    comparable_summaries = list(zip(df_matches['snapshot_summary'], df_matches['converted']))
    # Build prompt
    comparable_text = "\n\n".join(
        [f"Comparable Lead {i+1}:\nSummary: {summary}\nConverted: {'Yes' if converted == 1 else 'No'}"
         for i, (summary, converted) in enumerate(comparable_summaries)]
    )

    prompt = f"""
You are a B2B sales expert evaluating the quality of a sales lead using both observed historical data and your general knowledge of sales strategy, buying cycles, and lead behavior patterns.
You are evaluating the quality of a sales lead based on how similar historical leads performed. Assume today is May 1, 2025.

Subject Lead Summary:
{subject_summary}

comparable leads are sorted by similarity to the subject lead (most similar first).
Each includes a summary and whether the lead converted or not.
comparable Leads: 
{comparable_text}

Evaluate the Subject Lead on the following dimensions, each scored from 1–100. Use both observed patterns in the comparable leads and broader sales knowledge.

Dimensions:
1. Company Fit 
  - How well the lead aligns with the company’s ideal customer profile, including industry, size, scale, deal_value, cyber investment, and tech stack.
2. Engagement Quality
  - The level of activity and interest shown by the lead (downloads, meetings, decision-maker involvement)
  - The level of contact seniority involved.
  - Whether there is a current contract with a competitor solution and when the renewal date is
3. Pipeline Progress Potential 
  - The current stage of this lead. Later stages require more attention. 
  - The likelihood this lead will advance toward a sale based on similarity to converted leads and observed behaviors.

Ratings:
1. Company Fit: 
2. Engagement Quality: 
3. Pipeline Progress Potential: 

Provide a short explanation for each rating. 
Then, provide an overall **Lead Priority Score**, an integer from 1–100 that reflects how much attention this lead deserves. Make sure the response is in the format: 'Lead Priority Score: x' where x is an integer value.

Provide speculative recommendations or ideas to increase the likelihood of success for this lead. These can include engagement strategies, messaging adjustments, or other actions. Draw from both comparable patterns and general sales expertise.
""".strip()

    print(prompt)

    # Generate GPT Response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=600
    )
    # Extract and display GPT's response
    response = response.choices[0].message.content.strip()
    render_text(f"**Scoring Lead**: <br>{response}")
    st.markdown("\n")

    # Get score and save it
    match = re.search(r"Lead Priority Score:\s*(\d+)", response)

    # If a match is found, convert the number to an integer and store it
    if match:
        priority_score = int(match.group(1))  # Extracted number as integer
    else:
        print("No valid score found.")
    # Update table
    df_leads = st.session_state["data_leads"]
    df_leads.loc[df_leads['id'] == lead_id, 'priority_score'] = priority_score
    st.session_state["data_leads"] = df_leads

########################################################
########################################################


# Function to show the chatbot
def show_agent():
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'searching_lead' not in st.session_state:
        st.session_state.searching_lead = False
    if 'jumped_searching_lead' not in st.session_state:
        st.session_state.jumped_searching_lead = False

    # Data
    df_leads = st.session_state["data_leads"]
    df_snapshots = st.session_state["data_snapshots"]

    # ---- User path start ----
    st.write('Start Here')
    if st.button("Search a Lead"):
        st.session_state.searching_lead = True
        st.session_state.asking_about_lead = False
        st.session_state.scoring_lead = False

    # Clear All button
    if st.button("Clear All"):
        # Reset session state variables related to the chatbot
        st.session_state.history = []

        st.session_state.searching_lead = False
        st.session_state.lead_lookup_input = ""  # Clear any lead lookup input
        st.session_state.jumped_searching_lead = False

        st.session_state.ask_mode = False
        st.session_state.lead_question_input = False

        st.session_state.scoring_lead = False
        st.session_state.score_lead_id = None
 
        st.rerun()  # Rerun to reset everything

    if st.session_state.searching_lead:
        # Check for jump ID or user input
        if st.session_state.get("jumped_searching_lead", False):
            user_input = st.session_state.jump_to_lead_id
            st.info(f"Auto-searching for lead ID: {user_input}")
            st.session_state.jumped_searching_lead = False
        else:
            user_input = st.text_input("Enter the company name or lead ID (last 4 digits):", key="lead_lookup_input")

        if user_input:
            lead_info = None

            # Try to search by company name
            if not lead_info:
                lead_info = df_leads[df_leads['company_name'].str.contains(user_input, case=False, na=False)]
            # Try to search by lead ID if no company match
            if lead_info.empty:
                try:
                    lead_info = df_leads[df_leads['id'].str[-4:] == user_input]
                except ValueError:
                    pass  # No valid ID

            # Display lead info if found
            if not lead_info.empty:
                lead_row = lead_info.iloc[0]  # Take the first match
                lead_id = lead_row['id']        # Get lead id

                # Show the lead's status
                summary = lead_row['snapshot_summary']
                bold_labels = ['Lead Stage:', 'Lead Revenue', 'Investment', 'Engagement', 'Senior Contact'] 
                for label in bold_labels:
                    summary = summary.replace(label, f'**{label}**')
                render_text(
                    f"**Company:** {lead_row['company_name']}\n"
                    f"**ID:** {lead_id}\n"
                    f"**Status:** {lead_row['status']}\n"
                    f"**Open Date:** {lead_row['open_date']}\n"
                    f"**Close Date:** {lead_row['close_date']}\n"
                    f"**Lead Stage:** {lead_row['lead_stage']}\n\n"
                    f"**Summary:** {summary}\n")


                #Visualize the lead's stages
                df1 = df_snapshots.copy()       # Make copy
                df1 = df1[df1['id']==lead_id]      # Get snapshots from id
                visualize_stage_timeline(df1)

                # Ask about lead
                if st.button("Ask about this lead"):
                    st.session_state["ask_mode"] = True

                # Show Evaluate button only if lead is open
                if lead_row['status']=='Open':
                    if st.button("Score Lead"):
                        st.session_state.scoring_lead = True

                # Run ask step
                if st.session_state.get("ask_mode"):
                    ask_about_lead(lead_id)

                if st.session_state.get("scoring_lead"):
                    score_lead(lead_id)

            else:
                st.error("No lead found with that name or ID. Please try again.") 


