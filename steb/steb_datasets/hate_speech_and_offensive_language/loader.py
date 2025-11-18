
import os
import pandas as pd

def load_hate_speech_and_offensive_language(path: str):
    """
        GitHub: https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master
        Paper: https://ojs.aaai.org/index.php/ICWSM/article/view/14955
    """
    records = []
    df = pd.read_csv(os.path.join(path, "labeled_data.csv"))
    for _, row in df.iterrows():
        records.append({
            "text": row["tweet"],
            "label": row["class"],
        })
    return records
