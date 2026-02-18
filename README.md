Project Title
Text Classification Pipeline

```python
!pip install scipy
!pip install numpy<2
```

    Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (1.16.3)
    Requirement already satisfied: numpy<2.6,>=1.25.2 in /usr/local/lib/python3.12/dist-packages (from scipy) (2.0.2)
    /bin/bash: line 1: 2: No such file or directory



```python
import random
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram
```

Part 0: Get The Data


```python
train = sklearn.datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),)
test = sklearn.datasets.fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),)
print('train data size:', len(train.data))
print('test data size:', len(test.data))
```

    train data size: 11314
    test data size: 7532


Part 1: Classic Features: BoW / TF-IDF Classification


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# imports for when we try the different classification models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
```


```python
# loading the dataset, using all the categories and removing metadata (copied from Week 2 Lab Sample)
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"), random_state = 42)

X_text = data.data[:10000]
y = data.target[:10000]
label_names = data.target_names

df = pd.DataFrame({"text": X_text, "label": y})


print("Dataset loaded.")
print("Rows:", len(df))
print("Labels:", label_names)
print("\nLabel distribution:")
print(df["label"].value_counts(normalize=True).rename("share"))

print("\nSample rows:")
df.head(3)
```

    Dataset loaded.
    Rows: 10000
    Labels: ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    
    Label distribution:
    label
    8     0.0562
    5     0.0554
    12    0.0551
    15    0.0535
    14    0.0525
    10    0.0522
    9     0.0521
    3     0.0519
    13    0.0516
    4     0.0516
    11    0.0515
    6     0.0513
    1     0.0513
    2     0.0507
    7     0.0495
    17    0.0493
    16    0.0485
    0     0.0415
    18    0.0399
    19    0.0344
    Name: share, dtype: float64
    
    Sample rows:






  <div id="df-8f2e4521-1232-42b8-b5d7-095d4b2d80ce" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>\n\nI am sure some bashers of Pens fans are pr...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>My brother is in the market for a high-perform...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>\n\n\n\n\tFinally you said what you dream abou...</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8f2e4521-1232-42b8-b5d7-095d4b2d80ce')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8f2e4521-1232-42b8-b5d7-095d4b2d80ce button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8f2e4521-1232-42b8-b5d7-095d4b2d80ce');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Split into train+temp and test first
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

# split train_df into train and validation
train_df, val_df = train_test_split(train_df, test_size = 0.1765, random_state=42, stratify=train_df['label'])

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
```

    Train: 6999 Val: 1501 Test: 1500



```python
# checking for duplicates
def dup_rate(a, b):
    a_set = set(a["text"].astype(str))
    b_set = set(b["text"].astype(str))
    inter = a_set.intersection(b_set)
    return len(inter)

print("\nExact-duplicate counts across splits (should be ~0):")
print("Train ∩ Val:", dup_rate(train_df, val_df))
print("Train ∩ Test:", dup_rate(train_df, test_df))
print("Val   ∩ Test:", dup_rate(val_df, test_df))
```

    
    Exact-duplicate counts across splits (should be ~0):
    Train ∩ Val: 7
    Train ∩ Test: 13
    Val   ∩ Test: 4


## Bag of Words (BoW)

- with sklearn pipeline



```python
# BoW Pipeline

bow_models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "LinearSVM": LinearSVC(max_iter=5000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results_bow = {}

for name, model in bow_models.items():

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(
            max_features=20000,
            stop_words="english",
            min_df=2,
            ngram_range=(1,2)
        )),
        ("classifier", model)
    ])

    pipeline.fit(train_df["text"], train_df["label"])

    test_pred = pipeline.predict(test_df["text"])

    acc = accuracy_score(test_df["label"], test_pred)
    f1 = f1_score(test_df["label"], test_pred, average="macro")

    results_bow[name] = {
        "accuracy": acc,
        "macro_f1": f1,
        "model": pipeline,
        "predictions": test_pred
    }

    print(f"{name} — Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

```

    MultinomialNB — Accuracy: 0.6700 | Macro-F1: 0.6413
    LogisticRegression — Accuracy: 0.6500 | Macro-F1: 0.6428
    LinearSVM — Accuracy: 0.5933 | Macro-F1: 0.5858
    RandomForest — Accuracy: 0.6300 | Macro-F1: 0.6126



```python
# Find best BoW model based on Macro-F1
best_bow_name = max(results_bow, key=lambda x: results_bow[x]["macro_f1"])
best_bow_info = results_bow[best_bow_name]

print("\nBest BoW Model:", best_bow_name)

```

    
    Best BoW Model: LogisticRegression



```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_bow = confusion_matrix(test_df["label"], best_bow_info["predictions"])

plt.figure(figsize=(14,12))
sns.heatmap(
    cm_bow,
    xticklabels=label_names,
    yticklabels=label_names,
    cmap="Blues"
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title(f"Confusion Matrix — BoW ({best_bow_name})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

```
<img width="668" height="504" alt="Screenshot 2026-02-17 at 3 17 45 PM" src="https://github.com/user-attachments/assets/2062402a-af61-4770-be1f-ab3ed7a6b652" />


    

    


## Repeat for TF-IDF


```python
# TF-IDF Pipelines

tfidf_models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "LinearSVM": LinearSVC(max_iter=5000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results_tfidf = {}

for name, model in tfidf_models.items():

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=20000,
            stop_words="english",
            min_df=2,
            ngram_range=(1,2)
        )),
        ("classifier", model)
    ])

    pipeline.fit(train_df["text"], train_df["label"])

    test_pred = pipeline.predict(test_df["text"])

    acc = accuracy_score(test_df["label"], test_pred)
    f1 = f1_score(test_df["label"], test_pred, average="macro")

    results_tfidf[name] = {
        "accuracy": acc,
        "macro_f1": f1,
        "model": pipeline,          # store trained pipeline
        "predictions": test_pred    # store predictions
    }

    print(f"{name} — Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

```

    MultinomialNB — Accuracy: 0.6967 | Macro-F1: 0.6685
    LogisticRegression — Accuracy: 0.7167 | Macro-F1: 0.6997
    LinearSVM — Accuracy: 0.7220 | Macro-F1: 0.7135
    RandomForest — Accuracy: 0.6520 | Macro-F1: 0.6338



```python
best_model_name = max(results_tfidf, key=lambda x: results_tfidf[x]["macro_f1"])
best_model_info = results_tfidf[best_model_name]

print("Best TF-IDF Model:", best_model_name)

```

    Best TF-IDF Model: LinearSVM



```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_df["label"], best_model_info["predictions"])

plt.figure(figsize=(14,12))
sns.heatmap(
    cm,
    xticklabels=label_names,
    yticklabels=label_names,
    cmap="Blues"
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.title(f"Confusion Matrix — TF-IDF ({best_model_name})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

```
<img width="658" height="504" alt="Screenshot 2026-02-17 at 3 18 46 PM" src="https://github.com/user-attachments/assets/f6c775e6-135a-4ce7-8b76-252602d4da98" />


    
    


Part 2 - SentenceTransformer Embeddings + Classical Classifiers


```python
# Installing necessary package
!pip -q install sentence-transformers pypdf scikit-learn tqdm
```

    [?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/330.6 kB[0m [31m?[0m eta [36m-:--:--[0m
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━[0m [32m286.7/330.6 kB[0m [31m8.3 MB/s[0m eta [36m0:00:01[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m330.6/330.6 kB[0m [31m6.1 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import os, re
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```


```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 350
OVERLAP = 80
TOP_K = 10
BATCH_SIZE = 64
```


```python
def clean_text(s: str) -> str:
    s = s.replace("-\n", "")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str,source_id: str,max_chars: int = 900,overlap: int = 120):
    """
    Returns list of chunk dicts
    """
    text = clean_text(text or "")
    if not text.strip():
        return []

    chunks = []
    start = 0
    k = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text_ = text[start:end].strip()
        if chunk_text_:
            chunk_id = f"{source_id}::c{k}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": source_id,
                "start": start,
                "text": chunk_text_,
            })
            k += 1
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks
```


```python
def build_index(data_or_dir):
    all_chunks = []
    # --- Case A: sklearn Bunch (e.g., fetch_20newsgroups) ---
    if hasattr(data_or_dir, "data"):
        texts = data_or_dir.data
        targets = getattr(data_or_dir, "target", None)
        target_names = getattr(data_or_dir, "target_names", None)

        for i, doc in enumerate(texts):
            label = None
            if targets is not None and target_names is not None:
                label = target_names[targets[i]]

            source_id = f"20news::{label or 'doc'}::{i}"
            all_chunks.extend(chunk_text(doc, source_id, MAX_CHARS, OVERLAP))

    # --- Case B: list of raw strings ---
    elif isinstance(data_or_dir, list) and all(isinstance(x, str) for x in data_or_dir):
        for i, doc in enumerate(data_or_dir):
            source_id = f"text::{i}"
            all_chunks.extend(chunk_text(doc, source_id, MAX_CHARS, OVERLAP))

    else:
        raise TypeError(
            "build_index() expects a PDF directory path (str/pathlike), "
            "a sklearn Bunch with `.data`, or a list[str]."
        )

    if len(all_chunks) == 0:
        raise ValueError("No text extracted/chunked.")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # --- Embeddings (same as your original) ---
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in all_chunks]

    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[i:i+BATCH_SIZE]
        emb = model.encode(batch, normalize_embeddings=True)
        embs.append(emb)

    embeddings = np.vstack(embs).astype(np.float32)
    return all_chunks, embeddings, model
```


```python
# Approach 1
# DO NOT RUN
# DO NOT RUN
# DO NOT RUN
datatrain = fetch_20newsgroups(subset="train", remove=("headers","footers","quotes"), random_state = 42)

chunks, embeddings, model = build_index(datatrain)

print("\nIndex ready. Try a query like:")
print('results = search("your query", chunks, embeddings, model, top_k=5)')
```

    
    Total chunks: 51747


    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]



    config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    README.md: 0.00B [00:00, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]


    BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
    Key                     | Status     |  | 
    ------------------------+------------+--+-
    embeddings.position_ids | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.



    tokenizer_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]


    Embedding chunks: 100%|██████████| 809/809 [43:14<00:00,  3.21s/it]

    
    Index ready. Try a query like:
    results = search("your query", chunks, embeddings, model, top_k=5)


    



```python
# Approach 2
# Load data
train = fetch_20newsgroups(subset="train", remove=("headers","footers","quotes"), random_state=42)
test  = fetch_20newsgroups(subset="test",  remove=("headers","footers","quotes"), random_state=42)

y_train, y_test = train.target, test.target
target_names = train.target_names

# Embed docs
st_model = SentenceTransformer("all-MiniLM-L6-v2")
X_train_emb = st_model.encode(train.data, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
X_test_emb  = st_model.encode(test.data,  batch_size=64, show_progress_bar=True, normalize_embeddings=True)

X_train_emb = np.asarray(X_train_emb, dtype=np.float32)
X_test_emb  = np.asarray(X_test_emb, dtype=np.float32)
```


    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]


    BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
    Key                     | Status     |  | 
    ------------------------+------------+--+-
    embeddings.position_ids | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.



    Batches:   0%|          | 0/177 [00:00<?, ?it/s]



    Batches:   0%|          | 0/118 [00:00<?, ?it/s]



```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_model(name, clf, X_train, y_train, X_test, y_test, target_names=None):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1m:.4f}")

    if target_names is not None:
        print("\nClassification report:")
        print(classification_report(y_test, preds, target_names=target_names))
    else:
        print("\nClassification report:")
        print(classification_report(y_test, preds))

    print("Confusion matrix shape:", confusion_matrix(y_test, preds).shape)
    return {"model": name, "accuracy": acc, "macro_f1": f1m}

results = []

# 1) MNB on dense embeddings (usually poor; also requires non-negative features)
# If you used normalize_embeddings=True, values can be negative -> MNB will error.
# Workaround: shift features to be non-negative (for completeness only).
X_train_mnb = X_train_emb - X_train_emb.min()
X_test_mnb  = X_test_emb  - X_train_emb.min()

results.append(evaluate_model("MultinomialNB (shifted embeddings)", MultinomialNB(),
                              X_train_mnb, y_train, X_test_mnb, y_test, target_names))

# 2) Logistic Regression
results.append(evaluate_model("Logistic Regression",
                              LogisticRegression(max_iter=2000, n_jobs=-1),
                              X_train_emb, y_train, X_test_emb, y_test, target_names))

# 3) Linear SVM
results.append(evaluate_model("Linear SVM (LinearSVC)",
                              LinearSVC(),
                              X_train_emb, y_train, X_test_emb, y_test, target_names))

# 4) Random Forest
results.append(evaluate_model("Random Forest",
                              RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
                              X_train_emb, y_train, X_test_emb, y_test, target_names))

results
```

    
    === MultinomialNB (shifted embeddings) ===
    Accuracy: 0.6336
    Macro F1: 0.5899
    
    Classification report:
                              precision    recall  f1-score   support
    
                 alt.atheism       0.46      0.04      0.07       319
               comp.graphics       0.66      0.69      0.68       389
     comp.os.ms-windows.misc       0.62      0.62      0.62       394
    comp.sys.ibm.pc.hardware       0.51      0.65      0.57       392
       comp.sys.mac.hardware       0.72      0.43      0.54       385
              comp.windows.x       0.90      0.69      0.78       395
                misc.forsale       0.63      0.66      0.65       390
                   rec.autos       0.52      0.78      0.63       396
             rec.motorcycles       0.63      0.70      0.66       398
          rec.sport.baseball       0.87      0.81      0.84       397
            rec.sport.hockey       0.94      0.86      0.90       399
                   sci.crypt       0.70      0.68      0.69       396
             sci.electronics       0.46      0.51      0.49       393
                     sci.med       0.70      0.83      0.76       396
                   sci.space       0.57      0.76      0.66       394
      soc.religion.christian       0.45      0.91      0.60       398
          talk.politics.guns       0.53      0.62      0.57       364
       talk.politics.mideast       0.80      0.80      0.80       376
          talk.politics.misc       0.79      0.19      0.31       310
          talk.religion.misc       0.00      0.00      0.00       251
    
                    accuracy                           0.63      7532
                   macro avg       0.62      0.61      0.59      7532
                weighted avg       0.63      0.63      0.61      7532
    
    Confusion matrix shape: (20, 20)


    /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))


    
    === Logistic Regression ===
    Accuracy: 0.6841
    Macro F1: 0.6702
    
    Classification report:
                              precision    recall  f1-score   support
    
                 alt.atheism       0.48      0.47      0.47       319
               comp.graphics       0.66      0.70      0.68       389
     comp.os.ms-windows.misc       0.69      0.65      0.67       394
    comp.sys.ibm.pc.hardware       0.61      0.61      0.61       392
       comp.sys.mac.hardware       0.67      0.62      0.65       385
              comp.windows.x       0.86      0.75      0.80       395
                misc.forsale       0.72      0.71      0.71       390
                   rec.autos       0.54      0.78      0.64       396
             rec.motorcycles       0.72      0.72      0.72       398
          rec.sport.baseball       0.88      0.82      0.85       397
            rec.sport.hockey       0.89      0.88      0.89       399
                   sci.crypt       0.78      0.71      0.74       396
             sci.electronics       0.57      0.56      0.56       393
                     sci.med       0.79      0.83      0.81       396
                   sci.space       0.73      0.75      0.74       394
      soc.religion.christian       0.67      0.78      0.72       398
          talk.politics.guns       0.57      0.65      0.60       364
       talk.politics.mideast       0.84      0.77      0.80       376
          talk.politics.misc       0.50      0.46      0.48       310
          talk.religion.misc       0.34      0.21      0.26       251
    
                    accuracy                           0.68      7532
                   macro avg       0.68      0.67      0.67      7532
                weighted avg       0.69      0.68      0.68      7532
    
    Confusion matrix shape: (20, 20)
    
    === Linear SVM (LinearSVC) ===
    Accuracy: 0.6695
    Macro F1: 0.6546
    
    Classification report:
                              precision    recall  f1-score   support
    
                 alt.atheism       0.47      0.43      0.45       319
               comp.graphics       0.65      0.69      0.67       389
     comp.os.ms-windows.misc       0.66      0.62      0.64       394
    comp.sys.ibm.pc.hardware       0.61      0.59      0.60       392
       comp.sys.mac.hardware       0.65      0.62      0.64       385
              comp.windows.x       0.80      0.73      0.77       395
                misc.forsale       0.73      0.74      0.73       390
                   rec.autos       0.53      0.77      0.62       396
             rec.motorcycles       0.70      0.70      0.70       398
          rec.sport.baseball       0.88      0.81      0.84       397
            rec.sport.hockey       0.85      0.88      0.87       399
                   sci.crypt       0.75      0.72      0.73       396
             sci.electronics       0.57      0.51      0.54       393
                     sci.med       0.77      0.82      0.80       396
                   sci.space       0.74      0.71      0.73       394
      soc.religion.christian       0.64      0.74      0.69       398
          talk.politics.guns       0.55      0.63      0.58       364
       talk.politics.mideast       0.83      0.75      0.79       376
          talk.politics.misc       0.48      0.43      0.46       310
          talk.religion.misc       0.33      0.20      0.25       251
    
                    accuracy                           0.67      7532
                   macro avg       0.66      0.66      0.65      7532
                weighted avg       0.67      0.67      0.67      7532
    
    Confusion matrix shape: (20, 20)
    
    === Random Forest ===
    Accuracy: 0.6567
    Macro F1: 0.6289
    
    Classification report:
                              precision    recall  f1-score   support
    
                 alt.atheism       0.52      0.34      0.41       319
               comp.graphics       0.66      0.69      0.68       389
     comp.os.ms-windows.misc       0.65      0.60      0.63       394
    comp.sys.ibm.pc.hardware       0.56      0.63      0.59       392
       comp.sys.mac.hardware       0.68      0.57      0.62       385
              comp.windows.x       0.79      0.74      0.77       395
                misc.forsale       0.68      0.71      0.69       390
                   rec.autos       0.54      0.76      0.63       396
             rec.motorcycles       0.69      0.66      0.68       398
          rec.sport.baseball       0.82      0.83      0.82       397
            rec.sport.hockey       0.89      0.87      0.88       399
                   sci.crypt       0.74      0.68      0.71       396
             sci.electronics       0.56      0.46      0.50       393
                     sci.med       0.70      0.83      0.76       396
                   sci.space       0.66      0.69      0.67       394
      soc.religion.christian       0.55      0.85      0.66       398
          talk.politics.guns       0.48      0.66      0.56       364
       talk.politics.mideast       0.82      0.80      0.81       376
          talk.politics.misc       0.55      0.38      0.45       310
          talk.religion.misc       0.30      0.02      0.04       251
    
                    accuracy                           0.66      7532
                   macro avg       0.64      0.64      0.63      7532
                weighted avg       0.65      0.66      0.64      7532
    
    Confusion matrix shape: (20, 20)





    [{'model': 'MultinomialNB (shifted embeddings)',
      'accuracy': 0.633563462559745,
      'macro_f1': 0.5899052726873726},
     {'model': 'Logistic Regression',
      'accuracy': 0.6841476367498672,
      'macro_f1': 0.670223168124063},
     {'model': 'Linear SVM (LinearSVC)',
      'accuracy': 0.6695432819968136,
      'macro_f1': 0.6545737832582378},
     {'model': 'Random Forest',
      'accuracy': 0.6566648964418481,
      'macro_f1': 0.6289370954625403}]



Pt 1  
MultinomialNB — Accuracy: 0.6967 | Macro-F1: 0.6685  
LogisticRegression — Accuracy: 0.7167 | Macro-F1: 0.6997  
LinearSVM — Accuracy: 0.7220 | Macro-F1: 0.7135  
RandomForest — Accuracy: 0.6520 | Macro-F1: 0.6338  

Pt 2  
MultinomialNB — Accuracy: 0.6336 | Macro-F1: 0.5899  
LogisticRegression — Accuracy: 0.6841 | Macro-F1: 0.6702  
LinearSVM — Accuracy: 0.6695 | Macro-F1: 0.6546  
RandomForest — Accuracy: 0.6567 | Macro-F1: 0.6546  

Across the board, part 1 has better performance over all models. This may be due to the nature of the original data - specific keywords matter a lot in preserving meaning, and TF-IDF preserves these word signals much better than semantic embeddings. Embeddings usually perform better in instances where paraphrasing and synonyms are important - however, on larger datasets with many distinctive words and unique vocabulary sets per topic, TF-IDF oftens wins.

Part 3-A: Top Level Clustering


```python
!pip install sentence-transformers
```

    Requirement already satisfied: sentence-transformers in /usr/local/lib/python3.12/dist-packages (5.2.2)
    Requirement already satisfied: transformers<6.0.0,>=4.41.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (5.0.0)
    Requirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.4.0)
    Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (2.9.0+cpu)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (2.0.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.6.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.16.3)
    Requirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.15.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.67.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.20.3)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2025.3.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.2.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (0.28.1)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (26.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.3)
    Requirement already satisfied: shellingham in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.5.4)
    Requirement already satisfied: typer-slim in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (0.21.1)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (1.14.0)
    Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.6.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (2025.11.3)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.22.2)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.7.0)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (1.5.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (3.6.0)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (4.12.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (2026.1.4)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (1.0.9)
    Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (3.11)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (0.16.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.3)
    Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.12/dist-packages (from typer-slim->huggingface-hub>=0.20.0->sentence-transformers) (8.3.1)



```python
from sentence_transformers import SentenceTransformer
```


```python
#be careful when u run this cell. Took me 14 mins to execute.
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(train.data)
```


    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]


    BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
    Key                     | Status     |  | 
    ------------------------+------------+--+-
    embeddings.position_ids | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.



```python
def purity_score(c, y):
  A = np.c_[(c,y)]
  n_accurate = 0.
  for j in np.unique(A[:,0]):
    z = A[A[:,0] == j, 1]
    x = np.argmax(np.bincount(z))
    n_accurate += len(z[z == x])
  return n_accurate / A.shape[0]
```


```python
import nltk

purity = []
wcss=[]

krange = range(1,11)
for k in krange:
  tclusterer = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=5, algorithm="lloyd")
  clusts = tclusterer.fit_predict(embeddings)
  purity.append(purity_score(clusts, train.target))
  wcss.append(tclusterer.inertia_)
  print('k=',k,'done, purity:', purity[k-1])

plt.plot(krange, wcss)
```

    k= 1 done, purity: 0.0530316422131872
    k= 2 done, purity: 0.10173236697896411
    k= 3 done, purity: 0.14893052854870073
    k= 4 done, purity: 0.15016793353367508
    k= 5 done, purity: 0.19100229803782923
    k= 6 done, purity: 0.2267102704613753
    k= 7 done, purity: 0.2709033056390313
    k= 8 done, purity: 0.30228036061516705
    k= 9 done, purity: 0.3381651051794237
    k= 10 done, purity: 0.3808555771610394





    [<matplotlib.lines.Line2D at 0x7cf47ae53da0>]


<img width="571" height="397" alt="Screenshot 2026-02-17 at 8 50 53 PM" src="https://github.com/user-attachments/assets/eefd50fd-f1d9-4929-aff7-6898ca69940d" />




    
    


Do not see a clear "elbow" so select k=9.


```python
random.seed(a = 200)

clusterer = KMeans(n_clusters=9, init='k-means++', max_iter=100, n_init=10)
clusts = clusterer.fit_predict(embeddings)
plt.hist(clusts)
```




    (array([1463., 2511., 1482., 1407.,    0., 305., 1461., 566.,  1107.,
             1012.]),
     array([0. , 0.8, 1.6, 2.4, 3.2, 4. , 4.8, 5.6, 6.4, 7.2, 8. ]),
     <BarContainer object of 10 artists>)


<img width="511" height="361" alt="Screenshot 2026-02-17 at 8 53 19 PM" src="https://github.com/user-attachments/assets/2ad06dc6-50b9-4e5e-8869-dd9d55e75f9f" />

    




```python
def cluster_purity(c, y):
  numy = len(set(y))
  cvals = list(set(c)) #[str(ce) for ce in list(set(c))]
  numc = len(cvals)
  ind = [str(cval) for cval in cvals] #np.arange(numc)
  bottom = np.zeros(numc)
  for yidx in range(numy):
    counts = np.zeros(numc)
    for cidx in range(numc):
      num = len(list(filter(lambda p: p[0]==cvals[cidx] and p[1]==yidx, zip(c,y))))
      counts[cidx] = num
    plt.bar(ind, counts,label=train.target_names[yidx],bottom=bottom)
    bottom = bottom + counts
  plt.legend()
```


```python
cluster_purity(clusts, train.target)
print('Purity:', purity_score(clusts, train.target))
```

    Purity: 0.3402863708679512

<img width="558" height="439" alt="Screenshot 2026-02-17 at 8 54 18 PM" src="https://github.com/user-attachments/assets/6f88fd6c-7d02-4f9f-9948-2e4f47cd8e0c" />

    


Find documents closest to the cluster centroid.


```python
from sklearn.metrics.pairwise import cosine_distances
```


```python
centroids = clusterer.cluster_centers_
```


```python
nbrs = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine').fit(embeddings)
distance, indices = nbrs.kneighbors(centroids)
```


```python
labels = clusterer.labels_
centroids = clusterer.cluster_centers_
closest_docs = {}

for cluster_id in range(9):
    # Get indices of documents in this cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    # Get embeddings for this cluster
    cluster_embeddings = embeddings[cluster_indices]

    # Compute cosine distance to centroid
    distances = cosine_distances(
        cluster_embeddings,
        centroids[cluster_id].reshape(1, -1)
    ).flatten()

    # Find closest document index
    closest_idx = cluster_indices[np.argmin(distances)]

    closest_docs[cluster_id] = train.data[closest_idx]

    print(f"\nCluster {cluster_id} representative document: {closest_idx}")
    print(f" -> {train.data[closest_idx]}")
```

    
    Cluster 0 representative document: 6998
     -> No, not another false alarm, not a "It'll certainly be done by *next* week"
    message...  No, this is the real thing.  I repeat, this is *not* a drill!
    
    Batten down the hatches, hide the women, and lock up the cows, XV 3.00 has
    finally escaped.  I was cleaning its cage this morning when it overpowered
    me, broke down the office door, and fled the lab.  It was last seen heading
    in the general direction of export.lcs.mit.edu at nearly 30k per second...
    
    If found, it answers to the name of 'contrib/xv-3.00.tar.Z'.
    
    Have a blast.  I'm off to the vacation capital of the U.S.:  Waco, Texas.
    
    Cluster 1 representative document: 1089
     -> 
    
    
    
    
    
    
    Cluster 2 representative document: 1577
     -> 
    
    
    	Hate to be simple minded about this Tim, but I think its
    really very simple.  He was a dirty Jew.  And the only good Jew, in
    some peoples mind, is a dead Jew.  Thats what 40 years of propaganda
    that fails to discriminate between Jew and Zionist will do.  Thats
    what 20 years of statements like the ones I've appended will do to
    someones mind.  They make people sick.  They drag down political
    discourse to the point where killing your opponent is an honorable way
    to resolve a dispute.
    
    	What else can come of such demagogery?  Peace?
    
    Adam
    
    
    Arafat on political pluralism:
    
    	``Any Palestinian leader who suggests ending the intifada
    	exposes himself to the bullets of his own people and
    	endangers his life.  The PLO will know how to deal with
    	him.''
    	--- Arafat, Kuwaiti News Agency, 1/2/89
    
    Arafat on the massacre at Tienamin Square:
    
    	``...  on behalf of the Arab Palestinian People, their
            leadership, and myself...  [I] take this opportunity to express
            extreme gratification that you were able to restore normal order
            after the recent incidents in People's China.''
    	--- Arafat in telegram sent to the head of the Chinese Communist Party
    
    Yassir Arafat, humanitarian:
    
           ``Open fire on the new Jewish immigrants ...  be they from the
           Soviet Union, Ethiopia, or anywhere else.  It would be a disgrace if
           we did not lift a finger while herds of immigrants settle our
           territory.  I want you to shoot...  It makes no difference if they
           live in Jaffa or Jericho.  I give you explicit orders to open fire.
           Do everything to stop the flow of immigration.''
    	--- Yassir Arafat, Al Muharar (Lebanese weekly), April 10, 1990
    
    Yassir Arafat on genocide:
    
    	``When the Arabs set off their volcano, there will only be Arabs in
    	this part of the world.  Our people will continue to fuel the torch
    	of the revolution with rivers of blood until the whole of the
    	occupied homeland is liberated...''
    	--- Yasser Arafat, AP, 3/12/79
    
    
    
    
    Adam Shostack 				       adam@das.harvard.edu
    
    Cluster 3 representative document: 6733
     -> 386DX 25Mhz   (DTK motherboard  Intel microprocessor)
      128k internal cache
      4 megs Ram
      89 meg Harddrive    (IDE controller)
      1.2 meg floppy drive
      1.44 meg floppy drive
      2 serial ports
      1 parallel port
      Samsung VGA monitor
      VGA graphics card
      101 key keyboard
      2400 baud internal modem
    
      MS-DOS 6.0
      Procomm Plus  ver. 2.0
      Norton Utilities  ver. 4.5
      other varius utilities
    
    I'm upgrading and need to sell.  The system is reliable and ready to go.
    I've never had any problems with it.
    
    I'm asking  $1050 o.b.o.
    
    If you're interested, please respond by either E-mail or phone.
    
    TAE0460@zeus.tamu.edu
    or
    409-696-6043
    
    Cluster 4 representative document: 7568
     -> Visual Numerics Inc. (formerly IMSL and Precision Visuals) is in the
    process of securing sites for beta testing X Exponent Graphics 1.0 
    and C Exponent Graphics 2.0.  (Both X Exponent Graphics and C Exponent
    Graphics are 3GL products).  The beta period is from April 26 through 
    June 18.  The platform is HP9000/700 running under OS 8.07 with 
    ansi C 8.71 compiler.  The media will be sent on 4mm DAT cartridge 
    tape.  Here are some of the key facts about the two products.
     
    X Exponent Graphics 1.0 key facts:
     
    1. Complete collection of high-level 2D and 3D application plot types
       available through a large collection of X resources.
    2. Cstom widget for OSF/Motif developers.
    3. Built-in interactive GUI for plot customization.
    4. Easily-implemented callbacks for customized application feedback.
    5. XEG 1.0, being built on the Xt Toolkit provides the user a widget 
       library that conforms to the expected syntax and standards familar 
       to X programmers.
    6. XEG will also be sold as a bundle with Visual Edge's UIM/X product.
       This will enable user to use a GUI builder to create the graphical
       layout of an application.
     
    C Exponent Graphics 2.0 key facts:
     
    1. Written in C for C application programmers/developers.  The library
       is 100% written in C, and the programming interface conforms to C
       standards, taking advantage fo the most desirable features of C.
    2. Build-in GUI for interactive plot customization.  Through mouse 
       interaction, the user has complete interactive graph output control
       with over 200 graphics attributes for plot customization.
    3. Large collection of high-level application functions for "two-call"
       graph creation.  A wide variety of 2D and 3D plot types are available
       with minimal programming effort.
    4. User ability to interrupt and control the X event.  By controlling
       the X event loop, when the user use the mouse to manipulate the  plot
       the user can allow CEG to control the event loop or the user can 
       control the event loop.
     
    If anyone is interested in beta testing either of the products, please
    contact Wendy Hou at Visual Numerics via email at hou@imsl.com or call
    713-279-1066.
     
     
    -- 
    Jaclyn Brandt
    jbrandt@NeoSoft.com
    
    Cluster 5 representative document: 655
     -> NeXTstation 25MHz 68040 8/105
                         Moto 56001 DSP 
            Megapixel (perfect - no dimming or shaking)
    
            keyboard/mouse (of course :)
    
            2.1 installed
            2.1 docs
                Network and System Administration
                User's Reference
                Applications
    
            The NeXT Book, by Bruce Webster (New Copy)
    
            Black NeXTconnection modem cable
            30 HD disks (10 still in unwrapped box, others for backing up
                apps)
    
    I NEED to sell this pronto to get a car (my engine locked up)!
    Machine runs great... only used in my house.  Has been covered when
    not in use on the days I wasn't around.
    
    $2,300 INCLUDING Federal Express Second Day Air, OR best offer, COD to
    your doorstep (within continental US)!!  I need to sell this NOW, so
    if you don't agree with the price, make an offer, but within reason.
    ;)
    
    Thanks,
    JT
    
    Cluster 6 representative document: 7672
     -> Since everyone else seems to be running wild with predictions, I've
    decided to add my own fuel to the fire:
    They might seem a bit normal, but there are a few (albeit, small) surprises.
    
    American League East	 W	 L	GB
    1)New York Yankees	93	69	--
    2)Baltimore Orioles	90	72	 3
    3)Toronto Blue Jays	86	76	 7
    4)Cleveland Indians     84      78       9
    5)Boston Red Sox	77	85	16
    6)Milwaukee Brewers	74	88	19
    7)Detroit Tigers	73	89	20
    
    American League West	 W	 L	GB
    1)Minnesota Twins	94	68	--
    2)Kansas City Royals	92	70	 2
    3)Texas Rangers     	85	77	 9
    4)Chicago White Sox	77	85	17
    5)Oakland Athletics	74	88	20
    6)Seattle Mariners	70	92	24
    7)California Angels	65	97	29
    
    AL MVP-Kirby Puckett
    AL Cy Young-Kevin Appier
    AL Rookie of the Year-Tim Salmon
    AL Manager of the Year-Buck Showalter
    AL Comeback Player of the Year-Ozzie Guillen
    
    National League East	 W	 L	GB
    1)St. Louis Cardinals	91	71	--
    2)Philadelphia Phillies 89	73	 2
    3)Montreal Expos	88	74	 3
    4)New York Mets		84	78	 7
    5)Chicago Cubs		79	83	12
    6)Pittsburgh Pirates	73	89	18
    7)Florida Marlins	54     108	37
    
    National League West	 W	 L	GB
    1)Atlanta Braves	96	66	--
    2)Cincinnati Reds	94	68	 2
    3)Houston Astros	89	73	 7
    4)Los Angeles Dodgers	82	80	14
    5)San Francisco Giants	81	81	15
    6)San Diego Padres	75	87	21
    7)Colorado Rockies	59     103	37
    
    NL MVP-Barry Larkin
    NL Cy Young-John Smoltz
    NL Rookie of the Year-Wil Cordero
    NL Manager of the Year-Joe Torre
    NL Comeback Player of the Year-Eric Davis
    
    NL Champions-St. Louis Cardinals
    AL Champions-Minnesota Twins
    World Champions-St. Louis Cardinals
    
    The St. Louis picks are what my heart says.
    What my brain says, is they will win the division, lose to the Braves
    in the NLCS, and the Braves will win the Series against Minnesota.
    But for now, I'll stick with the Cards all the way.
    
    Cluster 7 representative document: 5794
     -> 
    Your last remark is a contradiction, but I'll let that pass.
    
    I was addressing the notion of the Great Commission, which
    you deleted in order to provide us with dull little homilies.
    Thank you, Bing Crosby.  Now you go right on back to sleep
    and mommy and daddy will tuck you in later.
    
    Oh, and how convenient his bible must have been to Michael
    Griffin, how convenient his Christianity.  "Well, I'll just
    skip the bit about not murdering people and loving the sinner
    and hating the sin and all that other stuff for now and
    concentrate on the part where it says that if someone is doing
    something wrong, you should shoot him in the back several times
    as he tries to hobble away on his crutches."
    
    I'll leave the "convert or die" program of the missionaries and
    their military escorts in the Americas for Nadja to explain as
    she knows much more about it than I.
    
    Must be awfully convenient, by the way, to offer platitudes
    as you have done, David, rather than addressing the arguments.
    
    Cluster 8 representative document: 11102
     -> From: "dan mckinnon" <dan.mckinnon@canrem.com>
    
    	   I have lurked here a bit lately, and though some of the math is
    	unknown to me, found it interesting. I thought I would post an article I
    	found in the Saturday, April 17, 1993 Toronto Star:
    
    	                  'CLIPPER CHIP' to protect privacy
    
    Politics is of course Dirty Pool, old man, and here we have a classic
    example: the NSA and the administration have been working on this for
    a *long* time, and in parallel with the announcement to us techies, we
    see they're hitting the press with propoganda.
    
    It's my bet the big magazines - Byte, Scientific American, et all - will
    be ready to run with a pre-written government-slanted story on this in
    the next issue.  ('Just keep us some pages spare boys, we'll give you
    the copy in time for the presses')
    
    We *must* get big names in the industry to write well argued pieces against
    this proposal (can you call it that when it's a de facto announcement?) and
    get them into the big magazines before too much damage is done.
    
    It would be well worth folks archiving all the discussions from here since
    the day of the announcement to keep all the arguments at our fingertips.  I
    think between us we could write quite a good piece.
    
    Now, who among us carries enough clout to guarantee publication?  Phil?
    Don Parker?  Mitch Kapor?


Use an LLM to generate a topic label for the cluster based on representative documents.


```python
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


def extract_label(text: str, max_words: int = 4, min_word_len: int = 2) -> str:
    """
    Extract a short label from text by taking the most frequent meaningful words.

    Args:
        text: Input text.
        max_words: Maximum number of words in the label.
        min_word_len: Minimum length for a word to be considered.

    Returns:
        A short label string.
    """
    if not text or not str(text).strip():
        return ""

    # Normalize: lowercase, keep letters and spaces
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    words = [w for w in cleaned.split() if len(w) >= min_word_len and w not in STOPWORDS]

    if not words:
        # Fallback: first few words of original (truncated)
        fallback = str(text).split()[:max_words]
        return " ".join(fallback) if fallback else ""

    # Count and take most frequent, then preserve order of first occurrence
    counts = Counter(words)
    ordered = sorted(counts.keys(), key=lambda w: (-counts[w], words.index(w)))
    return " ".join(ordered[:max_words]).title()


def generate_labels(data: dict, max_words: int = 4) -> dict:
    """
    Generate a label for each key based on its text value.

    Args:
        data: Dictionary mapping keys (e.g. numbers) to text values.
        max_words: Maximum words per label.

    Returns:
        Dictionary mapping same keys to generated labels.
    """
    return {key: extract_label(value, max_words=max_words) for key, value in data.items()}
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
labels = generate_labels(closest_docs)
print("Generated labels:")
for key, label in labels.items():
    print(f"  {key}: {label}")
```

    Generated labels:
      0: Xv Another False Alarm
      1: 
      2: Arafat People Jew Adam
      3: Meg Internal Floppy Drive
      4: Graphics User Exponent Plot
      5: Need Sell Offer Within
      6: Al Year Nl League
      7: Convenient Addressing Back Must
      8: Us Announcement Big Dan


Part 3-B: Second-Level Clustering on 2 Biggest Clusters.


```python
#Identify 2 largest clusters.
unique_elements, counts = np.unique(clusts, return_counts=True)

print(f"Unique elements: {unique_elements}")
print(f"Counts: {counts}")
```

    Unique elements: [0 1 2 3 4 5 6 7 8]
    Counts: [2548  303  537 1501 1388 1343 1017 1115 1562]



```python
#Extract docs in largest clusters
cl0 = [item for item, val in zip(train.data, clusts) if val == 0]
cl8 = [item for item, val in zip(train.data, clusts) if val == 8]
```


```python
cl0_target = [item for item, val in zip(train.target, clusts) if val == 0]
cl8_target = [item for item, val in zip(train.target, clusts) if val == 8]
```


```python
#This cell took 6 mins to execute. Be careful.
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_0 = model.encode(cl0)
embeddings_8 = model.encode(cl8)
```


    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]


    BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
    Key                     | Status     |  | 
    ------------------------+------------+--+-
    embeddings.position_ids | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.


Sub-clustering and labeling of cluster 0.


```python
random.seed(a = 200)

clusterer = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10)
clusts_0 = clusterer.fit_predict(embeddings_0)
plt.hist(clusts_0)
```




    (array([ 470.,    0.,    0.,    0.,    0., 462.,    0.,    0.,    0.,
             531.]),
     array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ]),
     <BarContainer object of 10 artists>)

<img width="456" height="320" alt="Screenshot 2026-02-17 at 8 55 39 PM" src="https://github.com/user-attachments/assets/34709859-f35f-4256-a00b-54006c44fcad" />




    



```python
cluster_purity(clusts_0, cl0_target)
print('Purity:', purity_score(clusts_0, cl0_target))
```

    Purity: 0.3626373626373626

<img width="538" height="420" alt="Screenshot 2026-02-17 at 8 56 59 PM" src="https://github.com/user-attachments/assets/f873ffb8-ea62-4237-8b4c-73a7f8320518" />



    
    



```python
centroids = clusterer.cluster_centers_
nbrs = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine').fit(embeddings_0)
distance, indices = nbrs.kneighbors(centroids)

labels = clusterer.labels_
centroids = clusterer.cluster_centers_
closest_docs_0 = {}

for cluster_id in range(3):
    # Get indices of documents in this cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    # Get embeddings for this cluster
    cluster_embeddings = embeddings_0[cluster_indices]

    # Compute cosine distance to centroid
    distances = cosine_distances(
        cluster_embeddings,
        centroids[cluster_id].reshape(1, -1)
    ).flatten()

    # Find closest document index
    closest_idx = cluster_indices[np.argmin(distances)]

    closest_docs_0[cluster_id] = cl0[closest_idx]
```


```python
labels = generate_labels(closest_docs_0)
print("Generated labels:")
for key, label in labels.items():
    print(f"  {key}: {label}")
```

    Generated labels:
      0: Driveways Mph Live Quite
      1: Sci World Lack Extraneously
      2: People Msg Effects Read


Sub-clustering and labeling of cluster 8.


```python
random.seed(a = 200)

clusterer = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10)
clusts_8 = clusterer.fit_predict(embeddings_8)
plt.hist(clusts_8)
```




    (array([199.,   0.,   0.,   0.,   0., 374.,   0.,   0.,   0., 439.]),
     array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ]),
     <BarContainer object of 10 artists>)


<img width="497" height="352" alt="Screenshot 2026-02-17 at 8 57 38 PM" src="https://github.com/user-attachments/assets/f70b5479-c390-43cb-acca-f4e94c95fda4" />

    



```python
cluster_purity(clusts_8, cl8_target)
print('Purity:', purity_score(clusts_8, cl8_target))
```

    Purity: 0.5691421254801536

<img width="475" height="350" alt="Screenshot 2026-02-17 at 8 58 35 PM" src="https://github.com/user-attachments/assets/b1d135df-f793-4c60-8dae-957fc61fc59b" />

    



```python
centroids = clusterer.cluster_centers_
nbrs = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine').fit(embeddings_8)
distance, indices = nbrs.kneighbors(centroids)

labels = clusterer.labels_
centroids = clusterer.cluster_centers_
closest_docs_8 = {}

for cluster_id in range(3):
    # Get indices of documents in this cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    # Get embeddings for this cluster
    cluster_embeddings = embeddings_8[cluster_indices]

    # Compute cosine distance to centroid
    distances = cosine_distances(
        cluster_embeddings,
        centroids[cluster_id].reshape(1, -1)
    ).flatten()

    # Find closest document index
    closest_idx = cluster_indices[np.argmin(distances)]

    closest_docs_8[cluster_id] = cl8[closest_idx]
```


```python
labels = generate_labels(closest_docs_8)
print("Generated labels:")
for key, label in labels.items():
    print(f"  {key}: {label}")
```

    Generated labels:
      0: Guns Gun Crime Study
      1: Would Government Found Clipper
      2: Public Become Clinton Text


Part 3-C: Show the “Partial Tree"


```python
import scipy.cluster.hierarchy as sch
```


```python
num_points = 500
data = train.data[:num_points]
target= train.target[:num_points]
features = TfidfVectorizer(ngram_range=(1,2), stop_words= 'english', lowercase=True, max_features=300)
vecs = features.fit_transform(data)
```


```python
plt.figure(figsize=(28,5))
dend = sch.dendrogram(sch.linkage(vecs.toarray(), method='ward'))
plt.show()
```

<img width="1133" height="218" alt="Screenshot 2026-02-17 at 3 19 44 PM" src="https://github.com/user-attachments/assets/237e2d9e-ad5e-494d-834a-42316fc0b6c1" />

    
    

