from googleapiclient.discovery import build
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from dotenv import load_dotenv
import os


load_dotenv()

API_key = os.getenv("API_key")
channel_id = "@freecodecamp"

youtube = build("youtube", "v3", developerKey=API_key)

# Select Playlist and use their playlist IDs to analyze the comments
playlist_ids = [
    "PLWKjhJtqVAbn21gs5UnLhCQ82f923WCgM",
    "PLWKjhJtqVAbmMuZ3saqRIBimAKIMYkt0E",
]


def get_video_ids(youtube, playlist_ids):
    """
    This function returns video IDs of the Playlist chosen from Youtube.

    """
    videos = []
    next_page_token = None

    for playlist_id in playlist_ids:
        while True:
            playlist_request = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=400,
                pageToken=next_page_token,
            )
            playlist_response = playlist_request.execute()

            videos += [
                item["contentDetails"]["videoId"] for item in playlist_response["items"]
            ]

            next_page_token = playlist_response.get("nextPageToken")

            if next_page_token is None:
                break
    return videos


video_ids = get_video_ids(youtube, playlist_ids)


def get_comments_for_video(youtube, video_id):
    """
    This function returns all comments from the video IDs that we got from running the previous function.

    """
    all_comments = []
    next_page_token = None

    while True:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            textFormat="plainText",
            maxResults=100,
        )
        comment_response = comment_request.execute()
        for item in comment_response["items"]:
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            all_comments.append(
                {
                    "Timestamp": top_comment["publishedAt"],
                    "Username": top_comment["authorDisplayName"],
                    "VideoID": video_id,  # Directly using video_id from function parameter
                    "Comment": top_comment["textDisplay"],
                    "Date": (
                        top_comment["updatedAt"]
                        if "updatedAt" in top_comment
                        else top_comment["publishedAt"]
                    ),
                }
            )

        next_page_token = comment_response.get("nextPageToken")
        if not next_page_token:
            break
    return all_comments


all_comments = []

# Using a for loop to store every comment from all videos in the playlist into an empty list.
for video_id in video_ids:
    video_comments = get_comments_for_video(youtube, video_id)
    all_comments.extend(video_comments)

df = pd.DataFrame(all_comments)

comment_ls = df["Comment"].tolist()

nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("spacytextblob")

# Keeping track of positive, negative and neutral comments

positive_comment_count = 0
negative_comment_count = 0
neutral_comment_count = 0


"""
Using the NLP Library to analyse the comment texts that we have stored in the list.
Polarity -  1 = Positive words are being used in the comments and -1 = Negative words are being used in the comments
I am also counting the positive, negative and neutral sentiments, so if positive comments are more than the negative comments,
the program will recommend the channel and vice versa.

I am also extracting data and saving everything in a CSV file.
"""

for i, text in enumerate(comment_ls):
    docx = nlp(text)
    polarity = docx._.polarity
    if polarity >= 0.2:
        df.loc[i, "Sentiment"] = "Positive"
        positive_comment_count += 1
    elif polarity <= -0.2:
        df.loc[i, "Sentiment"] = "Negative"
        negative_comment_count += 1
    else:
        df.loc[i, "Sentiment"] = "Neutral"
        neutral_comment_count += 1


if positive_comment_count > negative_comment_count:
    print("This Youtube channel is recommended based on positive comments.")
elif negative_comment_count > positive_comment_count:
    print("This Youtube channel is not recommneded based on negative comments.")

df.to_csv("sentiment_analysis.csv", sep="\t")
