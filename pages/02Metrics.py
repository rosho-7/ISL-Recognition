import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load and display metrics from a CSV file
def display_metrics(csv_file_path, level):
    try:
        # Load the CSV file into a DataFrame
        df_to_display = pd.read_csv(csv_file_path)
        
        # Display the metrics table
        st.write(f"{level} Metrics Table")
        st.dataframe(df_to_display, use_container_width=True)
        
        if not df_to_display.empty:
            st.subheader("Precision, Recall, and F1 Score Overview")

            # Plot for Precision
            fig_precision = px.bar(df_to_display, x=level, y='Precision', title=f"Precision by {level}",
                                   labels={level: level, 'Precision': 'Precision'},
                                   text='Precision')
            fig_precision.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_precision.update_layout(bargap=0.2)
            
            # Display Precision plot
            st.plotly_chart(fig_precision, use_container_width=True)

            # Plot for Recall
            fig_recall = px.bar(df_to_display, x=level, y='Recall', title=f"Recall by {level}",
                                labels={level: level, 'Recall': 'Recall'},
                                text='Recall')
            fig_recall.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_recall.update_layout(bargap=0.2)
            
            # Display Recall plot
            st.plotly_chart(fig_recall, use_container_width=True)

            # Plot for F1 Score
            fig_f1 = px.bar(df_to_display, x=level, y='F1 Score', title=f"F1 Score by {level}",
                            labels={level: level, 'F1 Score': 'F1 Score'},
                            text='F1 Score')
            fig_f1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_f1.update_layout(bargap=0.2)
            
            # Display F1 Score plot
            st.plotly_chart(fig_f1, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")

# Set the title of the page
st.title("Model Performance Metrics")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Word Level Metrics", "Sentence Level Metrics"])

# CSV file paths
word_csv_file_path = "ensemble_model_metrics_new_augmented_data.csv"  # Update with your Word-level CSV file path
sentence_csv_file_path = "model_metrics_sentence.csv"  # Update with your Sentence-level CSV file path

# Display appropriate metrics based on user selection
if page == "Word Level Metrics":
    st.markdown("### Word Level Metrics")
    display_metrics(word_csv_file_path, 'Word')
elif page == "Sentence Level Metrics":
    st.markdown("### Sentence Level Metrics")
    display_metrics(sentence_csv_file_path, 'Sentence')
