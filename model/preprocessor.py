import pandas as pd
import re
import string
from typing import List


def clean_tweets(df: pd.DataFrame) -> pd.DataFrame:
    escaped_chars = re.escape(string.punctuation + "Ã±ã¼â»§")
    new_df = df.copy(deep=True)
    new_df["clean_text"] = (
        new_df["OriginalTweet"]
            .str.replace("\r", " ")
            .str.replace("\n", " ")
            .str.lower()
            .str.replace("(?:\@|https?\://)\S+", "")
            .str.replace("[^\x00-\x7f]", " ")
            .str.replace(f"#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)", "")
            .str.replace(f"[{escaped_chars}]", " ")
            .str.replace("\s\s+", " ")
            .str.strip()
    )

    return new_df


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df[columns].copy(deep=True)
