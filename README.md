## 1. SYSTEM OBJECTIVE

What is implemented:
- The codebase builds a stock-forecasting dataset from price data plus news text, computes FinBERT sentiment features, aligns modalities into rolling windows, then trains and evaluates three model variants:
  - Hybrid (price + sentiment + company id): [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py), [scripts/prepare_hybrid_model_input.py](scripts/prepare_hybrid_model_input.py), [scripts/train_hybrid_late_fusion.py](scripts/train_hybrid_late_fusion.py), [scripts/evaluate_hybrid_r2.py](scripts/evaluate_hybrid_r2.py)
  - Price-only ablation: [scripts/price_only_ablation/prepare_price_only_input.py](scripts/price_only_ablation/prepare_price_only_input.py), [scripts/price_only_ablation/train_price_only_late_fusion.py](scripts/price_only_ablation/train_price_only_late_fusion.py), [scripts/price_only_ablation/evaluate_price_only_r2.py](scripts/price_only_ablation/evaluate_price_only_r2.py)
  - News-only ablation: [scripts/news_only_ablation/prepare_news_only_input.py](scripts/news_only_ablation/prepare_news_only_input.py), [scripts/news_only_ablation/train_news_only_late_fusion.py](scripts/news_only_ablation/train_news_only_late_fusion.py), [scripts/news_only_ablation/evaluate_news_only_r2.py](scripts/news_only_ablation/evaluate_news_only_r2.py)

Inputs and outputs:
- Raw inputs:
  - Price data downloaded from Yahoo Finance in [src/data_pipeline/collectors/price_collector.py](src/data_pipeline/collectors/price_collector.py)
  - News loaded from research CSV in [src/data_pipeline/collectors/research_news_collector.py](src/data_pipeline/collectors/research_news_collector.py)
- Intermediate outputs:
  - Daily consolidated news+price, daily sentiment, aligned training CSV
- Final outputs:
  - NPZ model input files
  - Trained model state dict files
  - Metrics JSON files

Why this structure appears in code:
- The staged structure is explicit in stage headers and artifact writes in [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py).
- The reason for this decomposition is partially inferable from naming and script boundaries (clear separation of ETL, input preparation, training, evaluation).
- Any higher-level product rationale is not explicitly defined in the code.

---

## 2. RAW DATA INGESTION

Price ingestion:
- What:
  - For each ticker, prices are downloaded with OHLCV fields plus Adj Close.
- How:
  - Uses yfinance download call per ticker in [src/data_pipeline/collectors/price_collector.py](src/data_pipeline/collectors/price_collector.py).
  - Required columns are enforced against:
    - Date, Ticker, Open, High, Low, Close, Adj Close, Volume
  - MultiIndex columns are flattened if returned.
- Why:
  - Required columns list is explicitly declared; likely to enforce consistent downstream schema.
  - The reason for choosing yfinance specifically is not explicitly stated in code comments.

News ingestion:
- What:
  - Reads a large CSV and filters articles to company-specific research windows.
- How:
  - Reads selected columns:
    - Date, Article_title, Stock_symbol, Publisher, Lsa_summary
  - Processes in chunks with on_bad_lines=skip in [src/data_pipeline/collectors/research_news_collector.py](src/data_pipeline/collectors/research_news_collector.py).
  - Builds final news_text by concatenating title + summary and normalizing whitespace.
- Why:
  - Chunked read and raised CSV field size limit are directly implemented, indicating handling of very large text fields.
  - Exact business reason for these source columns is not explicitly stated, but they are the fields used to construct news_text and ticker/date filters.

---

## 3. DATA FILTERING AND SELECTION

What:
- Company set is loaded from top company windows CSV and can be further filtered by CLI company list.
- Price rows are filtered strictly to each company window after download.
- News rows are filtered by ticker and published_date within each ticker window.

How:
- Windows loading and normalization in [src/data_pipeline/collectors/research_news_collector.py](src/data_pipeline/collectors/research_news_collector.py).
- Price window filtering in _filter_prices_to_company_windows in [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py).
- News window filtering inside collect_news_from_research_csv in [src/data_pipeline/collectors/research_news_collector.py](src/data_pipeline/collectors/research_news_collector.py).

Why:
- Logging strings in code explicitly call this strict window filtering.
- Inferred reason from structure: training/evaluation is constrained to research-defined periods.
- Any deeper rationale for window selection policy is not explicitly defined in code.

---

## 4. NEWS PROCESSING (CODE-DRIVEN)

Article-level processing:
- What:
  - FinBERT is run per article text to generate positive/neutral/negative probabilities.
- How:
  - Model/tokenizer loaded via transformers AutoModelForSequenceClassification and AutoTokenizer in [src/data_pipeline/processors/sentiment_aggregator.py](src/data_pipeline/processors/sentiment_aggregator.py).
  - Device selected by preference with fallback order in _select_torch_device.
  - Softmax logits into probabilities.
  - Label mapping from model id2label to sent_pos/sent_neu/sent_neg.
- Why:
  - Code enforces label validation via LABEL_TO_COLUMN and raises on unexpected label, indicating schema safety intent.
  - Why this specific model identifier was chosen is only in config value, not explained in code.

News-to-trading-day mapping:
- What:
  - Each news item is assigned to a trading day bucket.
- How:
  - If exact date is trading day: use it.
  - Else assign to most recent previous trading day; if before first trading day, assign first day.
  - Implemented in _assign_news_to_trading_buckets in [src/data_pipeline/processors/sentiment_aggregator.py](src/data_pipeline/processors/sentiment_aggregator.py).
- Why:
  - Comments explicitly mention weekend/holiday handling and earliest-boundary behavior.

Aggregation:
- What:
  - Per ticker-day, keeps top_k articles by relevance_score and aggregates sentiment.
- How:
  - relevance_score is |sent_pos - sent_neg|.
  - Sort descending by relevance_score per ticker/day, keep top_k.
  - Compute weighted sums and weighted means with fallback to simple means when weight_sum is tiny.
  - Implemented in build_daily_sentiment in [src/data_pipeline/processors/sentiment_aggregator.py](src/data_pipeline/processors/sentiment_aggregator.py).
- Why:
  - Comments explicitly state relevance_score is for top-k filtering and weighted aggregation.
  - Why this exact relevance formula was chosen is not explicitly justified beyond naming and usage.

Features produced:
- sentiment_strength:
  - |sent_pos - sent_neg|
- net_sentiment:
  - sent_pos - sent_neg
- news_count:
  - Count of selected articles
- has_news:
  - 1 when news exists, else 0

How:
- Computed in [src/data_pipeline/processors/sentiment_aggregator.py](src/data_pipeline/processors/sentiment_aggregator.py), and also backfilled if reusing existing sentiment CSV in [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py).

Why:
- Feature names and later usage in alignment/model imply they are intended to encode signal strength and availability.
- Any additional conceptual rationale is not explicitly stated in code.

Missing data:
- What:
  - No-news trading days are explicit zero-signal rows.
- How:
  - Sentiment table is left-joined onto all trading days from prices and filled with zeros for sentiment fields and has_news, zero integer for news_count.
- Why:
  - Comment explicitly says this is to ensure complete per-ticker trading-day table and make no-news days explicit.

---

## 5. PRICE DATA PROCESSING

What:
- Builds rolling price windows and computes return-based sequences/targets.

How:
- In alignment stage:
  - close_window is extracted from price target_value history.
  - return_window uses log returns over window:
    - r_t = log(p_t / p_(t-1))
  - Target return:
    - target_return = log(target_close / previous_close)
  - direction derived as int(target_return > 0)
  - volatility derived as abs(target_return)
- Implemented in [src/data_pipeline/processors/alignment.py](src/data_pipeline/processors/alignment.py).

Why:
- Code comments explicitly connect return_window to sentiment alignment.
- The reason for log-return choice appears through variable names target_log_return and formulas, but explicit theoretical justification is not present.

---

## 6. DATA ALIGNMENT

What:
- Merges price and sentiment on ticker/date and builds training rows from rolling windows.

How:
- Price columns normalized to date/ticker/target_value.
- Sentiment published_date renamed to date.
- Merge key:
  - ticker + date (left join from price to sentiment).
- Missing sentiment columns filled from DEFAULT_SENTIMENT.
- Implemented in [src/data_pipeline/processors/alignment.py](src/data_pipeline/processors/alignment.py).

Why:
- Left join from prices ensures all price days are retained.
- This is inferable from join direction and no-news fill behavior.
- Any stronger justification is not explicitly stated.

---

## 7. WINDOW CREATION

What:
- For each ticker, samples are created from rolling windows of length window_size in aligned series.

How:
- Loop:
  - for idx in range(window_size, len(grp))
- Inputs:
  - price sequence: return_window derived from close_window
  - sentiment sequence: sentiment_window from selected sentiment columns
- Targets:
  - target_return / target_log_return (same numeric value)
  - direction
  - volatility
  - target_close and previous_close also stored
- Implemented in [src/data_pipeline/processors/alignment.py](src/data_pipeline/processors/alignment.py).

Why:
- Comment explicitly says sentiment is aligned to returns by shifting sentiment window by one step to match return indexing.
- The reason for this exact indexing policy is documented in that comment.

---

## 8. FEATURE ENGINEERING

Derived features and transformations:
- Sentiment hard gate:
  - use_news mask where sentiment_strength >= threshold
- Temporal decay:
  - decay = exp(-lambda * days_ago), with recent day = 1.0
- Sentiment scaling:
  - sentiment_window multiplied by sentiment_scale_factor

How:
- All three are implemented in [src/data_pipeline/processors/alignment.py](src/data_pipeline/processors/alignment.py).
- Threshold/lambda/scale configured in [src/config/pipeline_config.yaml](src/config/pipeline_config.yaml).

Why:
- Comments explicitly say:
  - hard gating masks low-signal days
  - temporal decay reduces older influence
  - scale down sentiment to reduce noise
- These reasons are explicitly stated in comments.

Dropout logic:
- Sentiment input dropout exists in hybrid and news-only model classes:
  - [src/hybrid_model/late_fusion_model.py](src/hybrid_model/late_fusion_model.py)
  - [src/news_only_ablation/model.py](src/news_only_ablation/model.py)
- Price-only has no sentiment branch and no sentiment dropout:
  - [src/price_only_ablation/model.py](src/price_only_ablation/model.py)

Why:
- Comments explicitly say sentiment dropout is to reduce noise from low-quality signals.

Hybrid-specific gating:
- In model forward pass:
  - last-step sentiment strength is fed through linear + sigmoid gate
  - gate scales sentiment representation before fusion
- Implemented in [src/hybrid_model/late_fusion_model.py](src/hybrid_model/late_fusion_model.py).

Why:
- Variable names news_gate and sent_strength_last indicate intent to modulate sentiment branch by strength.
- Explicit detailed rationale beyond naming/comments is not provided.

---

## 9. NORMALIZATION

What:
- Feature normalization is fold-local and done inside estimators during fit.

How:
- Hybrid estimator:
  - computes global mean/std over train samples and timesteps for price and sentiment separately
  - applies same scaler in predict
  - [src/hybrid_model/sklearn_estimator.py](src/hybrid_model/sklearn_estimator.py)
- Price-only estimator:
  - price mean/std
  - [src/price_only_ablation/sklearn_estimator.py](src/price_only_ablation/sklearn_estimator.py)
- News-only estimator:
  - sentiment mean/std
  - [src/news_only_ablation/sklearn_estimator.py](src/news_only_ablation/sklearn_estimator.py)

Why:
- Code comments explicitly mention fold-aware normalization in estimator.fit to avoid leakage.
- This is also reinforced by model-input builders preserving raw window values.

---

## 10. DATASET STRUCTURE

Hybrid NPZ structure:
- X:
  - flattened price window + flattened sentiment window + company_id
- y:
  - currently built as 3 columns in builders:
    - return, volatility, direction
- Metadata:
  - previous_close, target_close, window_size, feature dims
- Implemented in [src/hybrid_data_prep/build_model_input.py](src/hybrid_data_prep/build_model_input.py).

Price-only and news-only NPZ structures:
- Similar structure with their modality-specific flattened features plus company_id
- y also built as 3 columns in both builders:
  - [src/price_only_ablation/build_model_input.py](src/price_only_ablation/build_model_input.py)
  - [src/news_only_ablation/build_model_input.py](src/news_only_ablation/build_model_input.py)

Why:
- Flattened X is explicitly documented by comments as making GridSearchCV handling straightforward while preserving recoverability to sequence shape.

Important current state:
- Training/evaluation now consume only y column 0 (return), even though builders still store volatility/direction in y.
- This is explicit in _extract_return_target and _split_targets in training/estimator scripts.

---

## 11. MODEL ARCHITECTURE

Hybrid:
- Branches:
  - Price LSTM
  - Sentiment LSTM (with input dropout)
  - Company embedding
- Fusion:
  - concat(price_repr, gated_sent_repr, company_repr)
  - ANN head with two linear-ReLU-dropout blocks
- Output:
  - single linear return head
- File: [src/hybrid_model/late_fusion_model.py](src/hybrid_model/late_fusion_model.py)

Price-only:
- Branches:
  - Price LSTM
  - Company embedding
- Same ANN pattern
- Output:
  - single return head
- File: [src/price_only_ablation/model.py](src/price_only_ablation/model.py)

News-only:
- Branches:
  - Sentiment LSTM (with input dropout)
  - Company embedding
- Same ANN pattern
- Output:
  - single return head
- File: [src/news_only_ablation/model.py](src/news_only_ablation/model.py)

Why:
- Docstrings/comments explicitly describe ablation intent (fair comparison by removing branches while keeping design style).
- The reason for specific hidden sizes/layer choices is not explicitly defined in code; these are tuned via grid search.

---

## 12. TRAINING PIPELINE

What:
- Trains with return-only objective using MSE loss.

How:
- In each estimator fit:
  - decode flat features into sequences
  - fit fold-local scaler on train data
  - DataLoader shuffle=True
  - forward returns 1D return prediction
  - loss = MSE(return_pred, y_return)
  - optimizer selected from adam/adamw/rmsprop
- Files:
  - [src/hybrid_model/sklearn_estimator.py](src/hybrid_model/sklearn_estimator.py)
  - [src/price_only_ablation/sklearn_estimator.py](src/price_only_ablation/sklearn_estimator.py)
  - [src/news_only_ablation/sklearn_estimator.py](src/news_only_ablation/sklearn_estimator.py)

Why:
- Return-only design is explicit in code after refactor.
- Comments explicitly indicate return-only model-selection metric in train scripts.

Training script outputs:
- Saves:
  - best model state dict
  - best params JSON including fit-data return metrics, direction accuracy from return sign, and confidence summary
- Files:
  - [scripts/train_hybrid_late_fusion.py](scripts/train_hybrid_late_fusion.py)
  - [scripts/price_only_ablation/train_price_only_late_fusion.py](scripts/price_only_ablation/train_price_only_late_fusion.py)
  - [scripts/news_only_ablation/train_news_only_late_fusion.py](scripts/news_only_ablation/train_news_only_late_fusion.py)

---

## 13. CROSS-VALIDATION

What:
- Uses TimeSeriesSplit when configured as timeseries; otherwise uses plain fold count in GridSearchCV training scripts and KFold/TimeSeriesSplit in evaluation scripts.

How:
- Training:
  - if cv_strategy == timeseries: cv = TimeSeriesSplit(n_splits=cv_folds)
- Evaluation:
  - timeseries -> TimeSeriesSplit
  - else -> KFold(shuffle=False)
- Files:
  - [scripts/train_hybrid_late_fusion.py](scripts/train_hybrid_late_fusion.py)
  - [scripts/price_only_ablation/train_price_only_late_fusion.py](scripts/price_only_ablation/train_price_only_late_fusion.py)
  - [scripts/news_only_ablation/train_news_only_late_fusion.py](scripts/news_only_ablation/train_news_only_late_fusion.py)
  - [scripts/evaluate_hybrid_r2.py](scripts/evaluate_hybrid_r2.py)
  - [scripts/price_only_ablation/evaluate_price_only_r2.py](scripts/price_only_ablation/evaluate_price_only_r2.py)
  - [scripts/news_only_ablation/evaluate_news_only_r2.py](scripts/news_only_ablation/evaluate_news_only_r2.py)

Why:
- Config value and code branching explicitly implement this.
- Why timeseries is preferred is not explicitly justified in comments, but naming indicates temporal split intent.

---

## 14. EVALUATION

What:
- Computes:
  - return_r2
  - return_rmse
  - return_mae
- Derives:
  - direction from sign(pred_return)
  - direction accuracy by comparing against sign(true_return)
- Computes confidence:
  - confidence = min(1.0, abs(pred_return) / threshold)
  - threshold from model_training.confidence_threshold, default 0.02

How:
- Implemented fold-wise in:
  - [scripts/evaluate_hybrid_r2.py](scripts/evaluate_hybrid_r2.py)
  - [scripts/price_only_ablation/evaluate_price_only_r2.py](scripts/price_only_ablation/evaluate_price_only_r2.py)
  - [scripts/news_only_ablation/evaluate_news_only_r2.py](scripts/news_only_ablation/evaluate_news_only_r2.py)
- Outputs JSON metrics to configured artifact/metrics paths.

Why:
- Metric names and explicit formula code define behavior.
- Why confidence formula uses this specific thresholded absolute-return design is not explicitly justified beyond implementation.

---

## 15. MODEL VARIANTS

Hybrid vs price-only vs news-only differences:
- Hybrid:
  - Inputs: price sequence + sentiment sequence + company id
  - Has sentiment gating by last sentiment_strength
- Price-only:
  - Inputs: price sequence + company id
  - No sentiment branch
- News-only:
  - Inputs: sentiment sequence + company id
  - No price branch

Implemented in:
- Hybrid model/estimator: [src/hybrid_model/late_fusion_model.py](src/hybrid_model/late_fusion_model.py), [src/hybrid_model/sklearn_estimator.py](src/hybrid_model/sklearn_estimator.py)
- Price-only model/estimator: [src/price_only_ablation/model.py](src/price_only_ablation/model.py), [src/price_only_ablation/sklearn_estimator.py](src/price_only_ablation/sklearn_estimator.py)
- News-only model/estimator: [src/news_only_ablation/model.py](src/news_only_ablation/model.py), [src/news_only_ablation/sklearn_estimator.py](src/news_only_ablation/sklearn_estimator.py)

Why variants exist:
- Package docstrings explicitly describe these as ablations/fair comparisons.

Important code-level inconsistency currently present:
- In price-only and news-only estimators, constructor signature still includes loss_weight_return, loss_weight_direction, loss_weight_volatility, but these are not assigned to self attributes.
- This creates sklearn clone/get_params incompatibility risk because BaseEstimator expects constructor params to exist as attributes.
- This behavior is directly inferable from:
  - [src/price_only_ablation/sklearn_estimator.py](src/price_only_ablation/sklearn_estimator.py)
  - [src/news_only_ablation/sklearn_estimator.py](src/news_only_ablation/sklearn_estimator.py)

---

## 16. FINAL PIPELINE FLOW

End-to-end flow explicitly implemented:

1. Load YAML config
   - [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py), [src/config/pipeline_config.yaml](src/config/pipeline_config.yaml)

2. Load company windows
   - [src/data_pipeline/collectors/research_news_collector.py](src/data_pipeline/collectors/research_news_collector.py)

3. Collect price data per ticker (with lookback buffer), then strict window filter
   - [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py), [src/data_pipeline/collectors/price_collector.py](src/data_pipeline/collectors/price_collector.py)

4. Collect/filter news by ticker + date windows (or reuse prepared CSV depending on flag)
   - [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py), [src/data_pipeline/collectors/research_news_collector.py](src/data_pipeline/collectors/research_news_collector.py)

5. Build daily consolidated news+price table
   - [src/data_pipeline/processors/news_price_alignment.py](src/data_pipeline/processors/news_price_alignment.py)

6. Build daily sentiment
   - bucket news to trading days
   - FinBERT inference
   - top-k relevance filtering and weighted aggregation
   - explicit zero rows for no-news trading days
   - [src/data_pipeline/processors/sentiment_aggregator.py](src/data_pipeline/processors/sentiment_aggregator.py)

7. Build company id map
   - [src/data_pipeline/processors/alignment.py](src/data_pipeline/processors/alignment.py)

8. Align modalities and create rolling windows
   - merge on ticker/date
   - derive return_window, target_return, direction, volatility
   - apply sentiment hard gate + decay + scale
   - [src/data_pipeline/processors/alignment.py](src/data_pipeline/processors/alignment.py)

9. Save aligned CSV and artifacts
   - [scripts/run_data_pipeline.py](scripts/run_data_pipeline.py)

10. Build NPZ inputs from aligned CSV
- Hybrid NPZ: [scripts/prepare_hybrid_model_input.py](scripts/prepare_hybrid_model_input.py), [src/hybrid_data_prep/build_model_input.py](src/hybrid_data_prep/build_model_input.py)
- Price-only NPZ: [scripts/price_only_ablation/prepare_price_only_input.py](scripts/price_only_ablation/prepare_price_only_input.py), [src/price_only_ablation/build_model_input.py](src/price_only_ablation/build_model_input.py)
- News-only NPZ: [scripts/news_only_ablation/prepare_news_only_input.py](scripts/news_only_ablation/prepare_news_only_input.py), [src/news_only_ablation/build_model_input.py](src/news_only_ablation/build_model_input.py)

11. Train models (GridSearchCV + TimeSeriesSplit when configured)
- Return-only objective
- Save best model + params JSON
- [scripts/train_hybrid_late_fusion.py](scripts/train_hybrid_late_fusion.py), [scripts/price_only_ablation/train_price_only_late_fusion.py](scripts/price_only_ablation/train_price_only_late_fusion.py), [scripts/news_only_ablation/train_news_only_late_fusion.py](scripts/news_only_ablation/train_news_only_late_fusion.py)

12. Evaluate models
- Return metrics
- Derived direction accuracy
- Derived confidence
- [scripts/evaluate_hybrid_r2.py](scripts/evaluate_hybrid_r2.py), [scripts/price_only_ablation/evaluate_price_only_r2.py](scripts/price_only_ablation/evaluate_price_only_r2.py), [scripts/news_only_ablation/evaluate_news_only_r2.py](scripts/news_only_ablation/evaluate_news_only_r2.py)

---

## Explicit unknowns (as required)

- The reason for selecting the specific company list and research windows is not explicitly defined in the code.
- The reason for choosing ProsusAI FinBERT versus alternative sentiment models is not explicitly defined in comments/docstrings.
- The reason for exact threshold defaults (for sentiment gating and confidence) is not explicitly stated in code; values are configuration-driven.
- There is no single Python orchestrator that chains full ETL + all training/evaluation stages; this is done by shell command sequencing (visible in terminal usage and separate scripts).
