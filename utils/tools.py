import os
import json
import pickle as pkl
import pandas as pd
from typing import Dict, Any, List

class Tools:
    def __init__(self):
        # Load the user segments
        self.user_segments = pd.DataFrame()
        PROJECT_ROOT = os.path.abspath(os.getcwd()) 
        DATA_PATH = os.path.join(PROJECT_ROOT, "data/")

        # Load the uswe segment data
        with open(os.path.join(DATA_PATH, "user_segments_viz.pkl"), "rb") as f:
            data = pkl.load(f)
            self.user_segments = pd.DataFrame.from_dict(data, orient="index").reset_index()
            self.user_segments.rename(columns={'index': 'id'}, inplace=True)

        # Load news topics data
        self.news_topics = pd.DataFrame()
        with open(os.path.join(DATA_PATH, "news_topics.pkl"), "rb") as f:
            self.news_topics = pkl.load(f)
            self.news_topics = pd.DataFrame.from_dict(data, orient="index").reset_index()
            self.news_topics.rename(columns={'index': 'id'}, inplace=True)

        # Load the raw news articles data
        self.news_raw = pd.DataFrame()
        with open(os.path.join(DATA_PATH, "news_viz2.json"), "r", encoding="utf-8") as f:
            self.news_raw = pd.DataFrame(json.load(f)).reset_index()
            self.news_raw.rename(columns={'index': 'id'}, inplace=True)
        
        self.TASK_FUNCS = {
            # User Segment tools
            "get_segment_description": self.get_segment_description, 
            "get_topic_transitions": self.get_topic_transitions, 
            "get_next_topic_prediction": self.get_next_topic_prediction, 
            "get_segment_engagement_stats": self.get_segment_engagement_stats, 
            "get_segment_regions": self.get_segment_regions, 
            "get_segment_time_activity": self.get_segment_time_activity, 
            "get_segment_articles_by_time": self.get_segment_articles_by_time, 
            "get_segment_engage_docs": self.get_segment_engage_docs, 
            "get_segment_not_engage_docs": self.get_segment_not_engage_docs, 
            "get_segment_high_rep_docs": self.get_segment_high_rep_docs, 
            "get_segment_activity_by_day_part": self.get_segment_activity_by_day_part, 

            # Articles topics tools
            "get_articles_info": self.get_articles_info, 
            "get_top_recent_articles": self.get_top_recent_articles, 
            "get_unique_clusters": self.get_unique_clusters, 

            # Raw News topics tools
            "get_news_topics_info": self.get_news_topics_info, 
            "get_news_topics_high_docs": self.get_news_topics_high_docs, 
            "get_news_topics_low_docs": self.get_news_topics_low_docs
        }


    # User Segment Analysis tools
    def get_segment_description(self, segment_id: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")

        df = self.user_segments.loc[segment_id].copy()
        return {
            "segment_id": segment_id,
            "title": df["title"],
            "description": df["desc"],
            "user_frequent": int(df["user_type_cnt"]["frequent"]),
            "user_nonfrequent": int(df["user_type_cnt"]["nonfrequent"]),
            "region_consumption":  dict(df["regions"])
        }

    def get_segment_engagement_stats(self, segment_id: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")

        df = self.user_segments.loc[segment_id, "df"].copy()
        df = df[df["true_engagement"] == True]
        return {
            "segment_id": segment_id,
            "avg_scroll_depth": round(float(df["avg_scroll_depth"].mean()), 2),
            "avg_engaged_secs": round(float(df["avg_engaged_secs"].mean()), 2),
            "avg_words_per_minute": round(float(df["avg_words_per_minute"].mean()), 2),
            "median_engaged_secs": round(float(df["avg_engaged_secs"].median()), 2),
            "engagement_rate": round(len(df[df["true_engagement"] == True]) / len(df), 3)
        }

    def get_topic_transitions(self, segment_id: int, top_n = 10) -> List[Dict[str, Any]]:
        try:
            segment_id = int(segment_id)
            top_n = int(top_n)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        seq_model = self.user_segments.loc[segment_id, "seq_model"]['df'].copy()
        if isinstance(seq_model, pd.DataFrame):
            transitions = []
            for from_topic in seq_model.index:
                for to_topic in seq_model.columns:
                    prob = seq_model.at[from_topic, to_topic]
                    transitions.append({
                        "from_topic": from_topic,
                        "to_topic": to_topic,
                        "probability": round(float(prob), 2)
                    })
        return sorted(transitions, key=lambda x: x["probability"], reverse=True)[:top_n]

    def get_next_topic_prediction(self, segment_id: int, current_topic: str, top_n = 3) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
            top_n = int(top_n)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        seq_model = self.user_segments.loc[segment_id, "seq_model"]['df'].copy()
        if isinstance(seq_model, pd.DataFrame):
            if current_topic not in seq_model.index:
                return {"error": f"Topic '{current_topic}' not found in model index."}
            next_topics_dict = seq_model.loc[current_topic].to_dict()
        elif isinstance(seq_model, dict):
            if current_topic not in seq_model:
                return {"error": f"Topic '{current_topic}' not found in model keys."}
            next_topics_dict = seq_model[current_topic]
        else:
            raise TypeError("seq_model must be a dict or pandas DataFrame.")
        sorted_topics = sorted(next_topics_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        predictions = [
            {"next_topic": topic, "probability": round(float(prob), 2)}
            for topic, prob in sorted_topics
        ]
        return {"current_topic": current_topic, "predictions": predictions}

    def get_segment_regions(self, segment_id: int, top_n: int = 7) -> List[Dict[str, Any]]:
        try:
            segment_id = int(segment_id)
            top_n = int(top_n)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        regions = self.user_segments.loc[segment_id, "regions"].copy()
        if not isinstance(regions, dict):
            raise TypeError("Expected 'regions' field to be a dictionary.")

        # Sort and format results
        sorted_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [{"region": r, "readers": int(v)} for r, v in sorted_regions]

    def get_segment_time_activity(self, segment_id: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        df = self.user_segments.loc[segment_id, "df"].copy()
        if "time_bin" in df.columns:
            time_col = "time_bin"
        elif "event_time" in df.columns:
            df["time_bin"] = df["event_time"].dt.strftime("%H:00-%H:59")
            time_col = "time_bin"
        else:
            raise ValueError("Expected a 'time_bin' or 'event_time' column in the dataframe.")
        activity = df.groupby(time_col).size().reset_index(name="reads")
        try:
            activity["hour_order"] = activity[time_col].str.slice(0, 2).astype(int)
            activity = activity.sort_values("hour_order")
        except Exception:
            pass
        # Identify the peak activity time
        peak_row = activity.loc[activity["reads"].idxmax()]
        peak_activity = str(peak_row[time_col])
        peak_value = int(peak_row["reads"])
        activity_list = activity[[time_col, "reads"]].rename(columns={time_col: "hour"}).to_dict(orient="records")
        return {
            "segment_id": segment_id,
            "activity_by_hour": activity_list,
            "peak_activity": peak_activity,
            "peak_value": peak_value
        }


    def get_segment_articles_by_time(self, segment_id: int, start_hour: int, end_hour: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
            start_hour = int(start_hour)
            end_hour = int(end_hour)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        df = self.user_segments.loc[segment_id, "df"].copy()
        df["hour"] = df["event_time"].dt.hour
        if start_hour <= end_hour:
            filtered_df = df[(df["hour"] >= start_hour) & (df["hour"] < end_hour)]
        else:
            filtered_df = df[(df["hour"] >= start_hour) | (df["hour"] < end_hour)]
        articles_read = filtered_df["id"].unique().tolist()
        return {
            "segment_id": segment_id,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "articles": articles_read[:10]   # Limit to 10 y now
        }

    def get_segment_engage_docs(self, segment_id: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        docs_engage = self.user_segments.loc[segment_id, "docs_engaged"].tolist() 
        return {
            "segment_id": segment_id,
            "docs_engage": docs_engage[:10]   # Limit to 10 y now
        }

    def get_segment_not_engage_docs(self, segment_id: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        docs_notengage = self.user_segments.loc[segment_id, "docs_notengaged"].tolist() 
        return {
            "segment_id": segment_id,
            "docs_notengage": docs_notengage[:10]   # Limit to 10 y now
        }

    def get_segment_high_rep_docs(self, segment_id: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        high_docs = self.user_segments.loc[segment_id, "high_docs"].tolist() 
        return {
            "segment_id": segment_id,
            "high_representative_docs": high_docs
        }

    def get_segment_activity_by_day_part(self, segment_id: int) -> Dict[str, Any]:
        try:
            segment_id = int(segment_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be an integer.")
        
        df = self.user_segments.loc[segment_id, "df"]
        if "event_on_day_part" not in df.columns:
            raise ValueError("The dataframe must contain an 'event_on_day_part' column.")
        day_part_counts = df["event_on_day_part"].value_counts().to_dict()
        standard_parts = ["morning", "afternoon", "evening", "night"]
        activity_by_day_part = {part: day_part_counts.get(part, 0) for part in standard_parts}
        peak_day_part = max(activity_by_day_part, key=activity_by_day_part.get)
        peak_value = activity_by_day_part[peak_day_part]
        return {
            "segment_id": segment_id,
            "activity_by_day_part": activity_by_day_part,
            "peak_day_part": peak_day_part,
            "peak_value": int(peak_value)
        }


    # Article tools 
    def get_articles_info(self, articles_ids: List[str]):
        cols = ['id', 'title', 'teaserText', 'first_publication_date']
        df = self.news_raw[self.news_raw['id'].isin(articles_ids)][cols].copy()
        # Convert to datetime safely (if numeric timestamps)
        df['first_publication_date'] = pd.to_datetime(
            df['first_publication_date'], unit='ms', errors='coerce'
        )
        # Convert to ISO-formatted string (safe for JSON serialization)
        df['first_publication_date'] = df['first_publication_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return df


    def get_top_recent_articles(self, articles_ids: List[str], top: int):
        cols = ['title', 'teaserText', 'first_publication_date']
        filtered = self.news_raw.loc[self.news_raw['id'].isin(articles_ids), cols]
        filtered = filtered.copy()
        # Convert to datetime safely
        filtered['first_publication_date'] = pd.to_datetime(
            filtered['first_publication_date'], unit='ms', errors='coerce'
        )
        # Convert Timestamp to string (ISO 8601)
        filtered['first_publication_date'] = filtered['first_publication_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Sort and return
        return filtered.sort_values(by='first_publication_date', ascending=False).head(top)


    def get_unique_clusters(self, articles_ids: List[str]):
        filtered = self.news_raw.loc[self.news_raw['id'].isin(articles_ids)]
        clusters = filtered['clusters'].dropna().tolist()
        unique_clusters = sorted(set([c for sublist in clusters for c in sublist]))
        return unique_clusters


    # News topics
    def get_news_topics_info(self, topics_id: List[int]):
        cols = ['title', 'desc']
        return self.news_topics.loc[self.news_topics['id'].isin(topics_id), cols]

    def get_news_topics_high_docs(self, topics_id: List[int]):
        row = self.news_topics[self.news_topics['id'].isin(topics_id)]
        if row.empty:
            return None
        return row.iloc[0]['high_docs'][:4].tolist()


    def get_news_topics_low_docs(self, topics_id: List[int]):
        row = self.news_topics[self.news_topics['id'].isin(topics_id)]
        if row.empty:
            return None
        return row.iloc[0]['low_docs'][:4].tolist()




