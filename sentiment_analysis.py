# Step 1: Data Cleaning & Preprocessing (Fixed Column Names)

import pandas as pd

# Load dataset (update this path when running in Colab)
file_path = "/content/smart_spoon_feedback.xlsx"
df = pd.read_excel(file_path)

# Standardize column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Confirm updated column names
print("Cleaned column names:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing key text responses
df_cleaned = df.dropna(subset=["feedback", "suggestions", "expectedfeatures"], how="any")

# Reset index
df_cleaned.reset_index(drop=True, inplace=True)

# Display sample cleaned data
print("\nCleaned Dataset Sample:")
df_cleaned[["feedback", "suggestions", "expectedfeatures"]].head()

# Step 2: Sentiment Analysis using TextBlob
!pip install textblob
from textblob import TextBlob

# Define a simple function to get sentiment
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to feedback and suggestions
df_cleaned["feedback_sentiment"] = df_cleaned["feedback"].apply(get_sentiment)
df_cleaned["suggestion_sentiment"] = df_cleaned["suggestions"].apply(get_sentiment)

# View results
df_cleaned[["feedback", "feedback_sentiment", "suggestions", "suggestion_sentiment"]].head()


# Step 3: Feature Keyword Extraction from expectedfeatures
!pip install rake-nltk
from rake_nltk import Rake

# Initialize RAKE
rake = Rake()

# Extract keywords from each response
def extract_keywords(text):
    rake.extract_keywords_from_text(text)
    return ", ".join(rake.get_ranked_phrases())

df_cleaned["expected_keywords"] = df_cleaned["expectedfeatures"].apply(extract_keywords)

# View the result
df_cleaned[["expectedfeatures", "expected_keywords"]].head()


from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Select behavior-related columns
segment_cols = ["lowsodiumdiet", "addsaltinlowsodium", "trysmartspoon", "wouldpurchase"]
df_seg = df_cleaned[segment_cols].copy()

# Convert to numeric using LabelEncoder
le = LabelEncoder()
for col in segment_cols:
    df_seg[col] = le.fit_transform(df_seg[col].astype(str))

# Use KMeans to find clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned["user_segment"] = kmeans.fit_predict(df_seg)

# View segmented data
df_cleaned[["lowsodiumdiet", "trysmartspoon", "wouldpurchase", "user_segment"]].head()


!pip install gradio
import gradio as gr
import pandas as pd

# Load the survey data
df = pd.read_excel("/content/smart_spoon_feedback.xlsx")

# Prepare summaries
# Prepare summaries with corrected column names
suggestions = df["Suggestions"].dropna().unique().tolist()
feedback = df["Feedback"].dropna().unique().tolist()
salt_opinions = df["SaltContentOpinion"].dropna().value_counts().to_dict()


# Define response function
def smart_spoon_chat(user_input):
    user_input = user_input.lower()

    if "feature" in user_input or "do" in user_input:
        return "🌟 The smart spoon tracks salt intake, sends health alerts, and connects to your mobile app."

    elif "buy" in user_input or "purchase" in user_input:
        return "🛒 You can buy it online after launch. Early users may get exciting offers!"

    elif "safe" in user_input or "health" in user_input:
        return "✅ Yes, it’s safe and especially helpful for those with high BP or sodium restrictions."

    elif "why" in user_input or "benefit" in user_input:
        return "💡 It helps prevent excess salt consumption, supporting heart and kidney health."

    elif "feedback" in user_input:
        return "📝 Here’s what some users said:\n- " + "\n- ".join(feedback[:5])

    elif "suggestion" in user_input:
        return "💡 Top user suggestions:\n- " + "\n- ".join(suggestions[:5])

    elif "salt" in user_input and ("opinion" in user_input or "content" in user_input):
        return "🧂 Opinions on salt content:\n" + "\n".join([f"{k}: {v} responses" for k, v in salt_opinions.items()])

    else:
        return "🤖 I’m your Smart Spoon Assistant! Ask me about features, benefits, safety, buying options, or user feedback."

# Launch the chatbot interface
gr.Interface(
    fn=smart_spoon_chat,
    inputs="text",
    outputs="text",
    title="Smart Spoon Chatbot"
).launch(share=True)


print(df.head())
