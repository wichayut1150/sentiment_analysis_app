import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras
from transformers import BertTokenizerFast, TFBertModel

from model.preprocessor import clean_tweets, select_columns
from model import Tokenizer

app = Flask(__name__)
bert_tokenizer = Tokenizer(BertTokenizerFast.from_pretrained('bert-base-uncased'))
bert_sentiment_analysis_model = keras.models.load_model(
    "src/saved_model/bert_sentiment_analysis_model.h5",
    custom_objects={"TFBertModel": TFBertModel}
)
with open("src/model_metadata.json", "r") as f:
    model_metadata = json.load(f)


@app.route("/sentiment", methods=["GET"])
def get_sentiment():
    df = pd.DataFrame({'OriginalTweet': [request.get_json(force=True).get("text")]})
    df = clean_tweets(df)
    df = select_columns(df, ["clean_text"])
    X_test = df['clean_text'].values
    test_input_ids, test_attention_masks = bert_tokenizer.tokenize(X_test)
    result_bert = bert_sentiment_analysis_model.predict([test_input_ids, test_attention_masks])
    y_pred_bert = np.zeros_like(result_bert)
    y_pred_bert[np.arange(len(y_pred_bert)), result_bert.argmax(1)] = 1
    converted_y_pred_bert = np.array(
        [
            model_metadata["possible_labels"][index]
            for index in np.where(y_pred_bert == 1)[1]
        ],
        dtype=object
    )
    sentiment = converted_y_pred_bert[0]

    return jsonify({"sentiment": sentiment})


if __name__ == "__main__":
    app.run(debug=True)
