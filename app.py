import streamlit as st
from predictor import predict_post
import time
import os

hf_token = os.getenv("HF_TOKEN")
# Page Configuration
st.set_page_config(
    page_title="Student Forum Discussion System",
    page_icon="📚",
    layout="centered"
)
import pandas as pd

# Initialize prediction history
if "history" not in st.session_state:
    st.session_state.history = []
# Title
st.title(" Student Forum Discussion System")
#st.write("Enter a forum post and the system will detect spam or classify the subject.")
st.markdown(
"""
This system detects whether a forum post is **Spam** or **Genuine**.
If genuine, it automatically classifies the post into the appropriate subject.
"""
)

st.divider()
# Input Box
post = st.text_area(
    "Enter your forum post:",
    height=150,
    placeholder="Example: Explain Binary Search Tree insertion..."
)
words = len(post.split())
MAX_WORDS = 200
if words > MAX_WORDS:
    st.error("Maximum 200 words allowed.")


# Predict Button
if st.button("Predict"):

    if post.strip() == "":
        st.warning("Please enter a forum post.")
    else:
        with st.spinner("Analyzing your post..."):

          time.sleep(1)
          result = predict_post(post)
        st.success("Analysis Completed")
        history_entry = {
                   "Post": post,
                   "Type": result["type"],
                   "Spam %": result["spam_confidence"],
                   "Genuine %": result["genuine_confidence"]
                    }

# Add subject details only if genuine
        if result["type"] == "Genuine":
                    history_entry["Subject"] = result["subject"]
                    
        else:
                    history_entry["Subject"] = "-"
                    

        st.session_state.history.insert(0, history_entry)

        st.divider()
        st.subheader("Prediction Result")

        # Spam Post
        if result["type"] == "Spam":

            st.error(" Spam/Toxic Post Detected")
            st.metric(
                "Spam Confidence",
                f"{result['spam_confidence']} %"
            )

            st.progress(result["spam_confidence"]/100)
        elif result["type"] == "May Be Spam":

            st.warning("⚠️ This post may be spam.")

            st.write(
                "The classifier is not confident enough to classify this post as either Spam or Genuine. Manual review needed!"
            )

            st.metric(
                "Spam Confidence",
                f"{result['spam_confidence']}%"
            )

            st.metric(
                "Genuine Confidence",
                f"{result['genuine_confidence']}%"
            )


        # Genuine Post
        else:

            st.success(" Genuine Post")
            st.metric(
                "Genuine Confidence",
                f"{result['genuine_confidence']} %"
            )

            st.progress(result["genuine_confidence"]/100)

            st.divider()


            #st.write(f"**Subject:** {result['subject']}")
            st.write(f"**Subject :**")
            # Optional subject-wise icons
            if result['subject'] == "DSA":
                st.info("Data Structures & Algorithms")

            elif result['subject'] == "OS":
                st.info(" Operating Systems")

            elif result['subject'] == "DBMS":
                st.info(" Database Management Systems")

            elif result['subject'] == "CN":
                st.info(" Computer Networks")

            elif result['subject'] == "OOP":
                st.info(" Object Oriented Programming")

            else:
                st.info(" Miscellaneous")

            

            
with st.expander(" Example Posts"):
    st.subheader("Copy Example Posts")

    st.code(" Explain Binary Search Tree insertion.", language=None)

    st.code(" What is deadlock prevention?", language=None)

    st.code(" Explain normalization in DBMS.", language=None)

    st.code(" Difference between TCP and UDP.", language=None)

    st.code(" What is polymorphism?", language=None)

    st.code(" Win a free iPhone now! Click here.", language=None)
st.divider()

st.subheader(" Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )

else:

    st.info("No predictions made yet.")
col1, col2 = st.columns([4,1])

with col2:

    if st.button(" Clear History"):

        st.session_state.history = []

        st.rerun()
if len(st.session_state.history) > 0:

    csv = pd.DataFrame(
        st.session_state.history
    ).to_csv(index=False)

    st.download_button(
        label=" Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
st.divider()
st.caption("©️ All Rights Reserved")