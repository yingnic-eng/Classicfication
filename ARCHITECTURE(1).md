# Architecture: Text Classification & Clustering

This document describes the **data flow** and **module responsibilities** for the classification project (20 Newsgroups).

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              RAW DATA SOURCE                                              │
│  sklearn.datasets.fetch_20newsgroups (train / test / all)                                │
│  → .data (text), .target (label index), .target_names (label strings)                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SPLIT & PREPROCESS (Part 0 / Part 1)                        │
│  • train_test_split (stratified) → train_df, val_df, test_df                             │
│  • Optional: subsample (e.g. 10k rows), duplicate check across splits                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│   PART 1: Classic Feats   │ │   PART 2: Embeddings      │ │   PART 3: Clustering      │
│   BoW / TF-IDF            │ │   SentenceTransformer     │ │   KMeans + Hierarchy      │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```

---

## Part 1: Classic Features (BoW / TF-IDF) — Data Flow

```
train_df["text"], train_df["label"]
         │
         ▼
┌─────────────────────────────────────┐
│  VECTORIZER (feature extraction)    │
│  • CountVectorizer (BoW)            │  →  X_train (sparse matrix)
│  • TfidfVectorizer                  │     max_features=20k, ngram_range=(1,2), stop_words
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  CLASSIFIER (sklearn Pipeline)      │
│  • MultinomialNB                    │
│  • LogisticRegression               │  →  fit(X_train, y_train)
│  • LinearSVC                        │
│  • RandomForestClassifier           │
└─────────────────────────────────────┘
         │
         ▼
test_df["text"] ──► same pipeline.predict() ──► test_pred
         │
         ▼
┌─────────────────────────────────────┐
│  EVALUATION                         │
│  accuracy_score, f1_score(macro),   │
│  confusion_matrix, classification_  │
│  report                             │
└─────────────────────────────────────┘
```

**Module responsibilities (Part 1):**

| Module / Component      | Responsibility |
|-------------------------|----------------|
| **Data**                | Load 20 Newsgroups; stratified train/val/test split; optional duplicate checks. |
| **Vectorizer**          | Turn raw text into numeric features (BoW counts or TF-IDF weights). |
| **Pipeline**            | Chain vectorizer + classifier; single `.fit()` / `.predict()` interface. |
| **Classifier**          | Supervised learning (NB, LR, SVM, RF) on vectorized text. |
| **Evaluation**          | Accuracy, macro F1, confusion matrix, classification report. |

---

## Part 2: SentenceTransformer Embeddings + Classical Classifiers — Data Flow

```
train.data, test.data (raw text)
         │
         ▼
┌─────────────────────────────────────┐
│  SentenceTransformer                │
│  model.encode(texts, batch_size=64, │  →  X_train_emb, X_test_emb (dense float32)
│  normalize_embeddings=True)         │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  CLASSICAL CLASSIFIER                │
│  • MultinomialNB (on shifted emb.)  │  →  fit(X_train_emb, y_train)
│  • LogisticRegression               │     predict(X_test_emb)
│  • LinearSVC                        │
│  • RandomForest                     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  evaluate_model()                    │
│  accuracy, macro F1, classification │
│  report, confusion matrix           │
└─────────────────────────────────────┘
```

**Module responsibilities (Part 2):**

| Module / Component      | Responsibility |
|-------------------------|----------------|
| **SentenceTransformer** | Map each document to a fixed-size dense embedding (e.g. all-MiniLM-L6-v2). |
| **Classifier**          | Same classical models as Part 1, but on embedding vectors instead of BoW/TF-IDF. |
| **evaluate_model()**    | Train, predict, and report accuracy, macro F1, and per-class metrics. |

---

## Part 3: Clustering — Data Flow

### 3-A: Top-Level Clustering

```
train.data (raw text)
         │
         ▼
┌─────────────────────────────────────┐
│  SentenceTransformer.encode()      │  →  embeddings (one per document)
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  KMeans(n_clusters=9)               │  →  clusts (cluster id per doc)
│  fit_predict(embeddings)           │      cluster_centers_
└─────────────────────────────────────┘
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────────┐
│  purity_score(clusts,   │          │  Centroid representatives   │
│  train.target)         │          │  cosine_distances to centroid│
│  WCSS / elbow (k)      │          │  → closest_docs per cluster  │
└─────────────────────────┘          └─────────────────────────────┘
                                                  │
                                                  ▼
                                     ┌─────────────────────────────┐
                                     │  generate_labels(closest_   │
                                     │  docs) — extract_label()     │  →  short topic labels
                                     │  (freq words, no LLM)       │
                                     └─────────────────────────────┘
```

### 3-B: Second-Level Clustering

```
Documents in 2 largest clusters (e.g. clusts==0, clusts==8)
         │
         ▼
┌─────────────────────────────────────┐
│  SentenceTransformer.encode()      │  →  embeddings_0, embeddings_8
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  KMeans(n_clusters=3) per cluster   │  →  clusts_0, clusts_8
└─────────────────────────────────────┘
         │
         ▼
Same as 3-A: purity, centroid reps, generate_labels() → sub-cluster labels
```

### 3-C: Partial Tree (Hierarchical)

```
train.data[:500] (subsample)
         │
         ▼
┌─────────────────────────────────────┐
│  TfidfVectorizer                    │  →  vecs (sparse matrix)
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  scipy.cluster.hierarchy            │
│  linkage(vecs.toarray(), 'ward')    │  →  dendrogram
│  dendrogram()                       │
└─────────────────────────────────────┘
```

**Module responsibilities (Part 3):**

| Module / Component       | Responsibility |
|--------------------------|----------------|
| **SentenceTransformer**  | Produce document embeddings for KMeans (3-A, 3-B). |
| **KMeans**               | Partition embeddings into k clusters; provide centroids and labels. |
| **purity_score()**       | Compare cluster assignment to ground-truth labels (supervised metric). |
| **NearestNeighbors / cosine_distances** | Find document(s) closest to each cluster centroid. |
| **extract_label() / generate_labels()** | Derive short topic labels from representative text (word frequency, no LLM). |
| **cluster_purity()**     | Stacked bar chart of label distribution per cluster. |
| **TfidfVectorizer + linkage** | Build hierarchical clustering tree (3-C) and plot dendrogram. |

---

## Optional: Indexing / Chunking Path (build_index)

Used for search or chunked corpora (e.g. when using PDFs or long docs):

```
data (Bunch or list[str])
         │
         ▼
┌─────────────────────────────────────┐
│  chunk_text(doc, source_id,         │  →  list of chunk dicts
│  max_chars, overlap)                 │      {chunk_id, source, start, text}
│  clean_text()                        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  SentenceTransformer.encode(chunks) │  →  embeddings (per chunk)
└─────────────────────────────────────┘
         │
         ▼
  all_chunks + embeddings + model  →  used for similarity search (e.g. top_k)
```

**Module responsibilities (indexing):**

| Module / Component | Responsibility |
|--------------------|----------------|
| **clean_text()**   | Normalize whitespace and line breaks. |
| **chunk_text()**   | Split long documents into overlapping text chunks with stable IDs. |
| **build_index()**  | Accept Bunch or list of strings; chunk; embed; return chunks, embeddings, model. |

---

## Summary: Data Types by Stage

| Stage           | Input(s)              | Output(s)                          |
|-----------------|------------------------|------------------------------------|
| Load            | —                      | Bunch / DataFrame (text, target)   |
| Split           | Full dataset           | train_df, val_df, test_df          |
| BoW/TF-IDF      | Raw text               | Sparse feature matrix              |
| Embeddings      | Raw text (or chunks)   | Dense float32 matrix               |
| Classifier      | Features + labels      | Predictions, trained model         |
| KMeans          | Embeddings             | Cluster IDs, centroids             |
| Label generation| Representative text    | Short topic label per cluster      |
| Hierarchy       | TF-IDF vectors         | Linkage matrix, dendrogram         |

---

## Dependencies (Libraries)

- **Data / split:** `sklearn.datasets`, `sklearn.model_selection`, `pandas`
- **Classic features:** `sklearn.feature_extraction.text` (CountVectorizer, TfidfVectorizer)
- **Classifiers:** `sklearn.linear_model`, `sklearn.naive_bayes`, `sklearn.svm`, `sklearn.ensemble`, `sklearn.pipeline`
- **Embeddings:** `sentence_transformers`
- **Clustering:** `sklearn.cluster` (KMeans), `sklearn.neighbors` (NearestNeighbors), `scipy.cluster.hierarchy`
- **Metrics:** `sklearn.metrics` (accuracy, F1, confusion_matrix, classification_report)
- **Utils / viz:** `numpy`, `matplotlib`, `seaborn`, `tqdm`, `nltk` (stopwords for label extraction)
