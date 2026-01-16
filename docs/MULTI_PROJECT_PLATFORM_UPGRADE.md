# Multi-Project GMP Forecasting Platform Upgrade Plan

**Version:** 1.0
**Date:** 2026-01-16
**Author:** ML Platform Architecture Team
**Status:** Proposed

---

## Executive Summary

This document details the end-to-end platform upgrade to transform the existing single-project GMP forecasting system into a scalable multi-project (multi-tenant) platform. The upgrade enables cross-project model training while maintaining project-specific forecasts, with strict data isolation and temporal integrity.

---

## 1. Assumptions

### 1.1 Business Context
- **Project Volume:** Platform will scale from 1 to 50+ active projects within 18 months
- **Project Size:** Each project has 15-40 GMP divisions (trades), 100-500 budget lines, 1K-50K direct cost transactions
- **Historical Data:** 3-5 years of historical project data available for backfill (10-15 completed projects)
- **Forecast Granularity:** Per-trade (GMP division), per-project, weekly/monthly time buckets
- **Regional Scope:** All projects share EAST/WEST/SHARED zone taxonomy (portfolio-wide)

### 1.2 Technical Context
- **Current Stack:** FastAPI, SQLAlchemy/SQLite, TensorFlow/Keras, scikit-learn, DVC/S3
- **Target Environment:** PostgreSQL (multi-tenant), containerized (Docker/K8s), cloud-native
- **Data Format:** Projects use similar but not identical trade taxonomies (CSI-based with variations)
- **Ingestion:** CSV/Excel imports from Procore, P6 schedules, owner breakdown files

### 1.3 Constraints
- **Backward Compatibility:** Existing single-project workflows must continue functioning during migration
- **Data Isolation:** Project data must be strictly segregated (no cross-project data leakage in queries)
- **Regulatory:** Construction contract data may have confidentiality requirements between owners
- **Latency:** Forecast inference must complete in <2 seconds for dashboard display

### 1.4 Non-Goals (Phase 1)
- Real-time streaming inference (batch is sufficient)
- Federated learning across organizational boundaries
- Automated hyperparameter optimization per project

---

## 2. Target Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-PROJECT PLATFORM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │  Project A   │   │  Project B   │   │  Project C   │  ... (N Projects)   │
│  │  (Active)    │   │  (Active)    │   │  (Completed) │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                              │
│         ▼                  ▼                  ▼                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    CANONICAL DATA LAYER                          │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │        │
│  │  │ Trade       │  │ Normalized  │  │ Time-Series │              │        │
│  │  │ Taxonomy    │  │ Costs       │  │ Features    │              │        │
│  │  │ (Master)    │  │ (Cents/SF)  │  │ (Sequences) │              │        │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    ML TRAINING LAYER                             │        │
│  │                                                                  │        │
│  │   ┌─────────────────────┐    ┌─────────────────────┐            │        │
│  │   │  Global Foundation  │───▶│  Project-Specific   │            │        │
│  │   │  Model (All Data)   │    │  Fine-Tuned Models  │            │        │
│  │   └─────────────────────┘    └─────────────────────┘            │        │
│  │            │                           │                         │        │
│  │            ▼                           ▼                         │        │
│  │   ┌─────────────────────────────────────────────┐               │        │
│  │   │         MODEL REGISTRY (MLflow/DVC)         │               │        │
│  │   │  - Global models (versioned)                │               │        │
│  │   │  - Per-project adapters                     │               │        │
│  │   │  - Evaluation metrics                       │               │        │
│  │   └─────────────────────────────────────────────┘               │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    INFERENCE LAYER                               │        │
│  │                                                                  │        │
│  │   Project A          Project B          Project C                │        │
│  │   ┌──────────┐       ┌──────────┐       ┌──────────┐            │        │
│  │   │ Forecast │       │ Forecast │       │ Backtest │            │        │
│  │   │ Service  │       │ Service  │       │ Archive  │            │        │
│  │   └──────────┘       └──────────┘       └──────────┘            │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Project Ingestion** | Per-project data validation & canonicalization | FastAPI + Pydantic |
| **Canonical Store** | Normalized cross-project data warehouse | PostgreSQL + partitioning |
| **Trade Taxonomy Service** | Master trade mapping & normalization | Redis cache + PostgreSQL |
| **Feature Store** | Offline/online feature computation & serving | Feast or custom |
| **Training Orchestrator** | Cross-project training pipelines | Airflow/Prefect + Celery |
| **Model Registry** | Version control for models & artifacts | MLflow + DVC/S3 |
| **Inference Service** | Project-specific forecast serving | FastAPI + TensorFlow Serving |
| **Monitoring** | Drift detection, performance tracking | Prometheus + Grafana |

---

## 3. Data & Schema

### 3.1 Multi-Project Data Model

#### 3.1.1 New Tables

```sql
-- Master Trade Taxonomy (Portfolio-wide)
CREATE TABLE canonical_trades (
    id SERIAL PRIMARY KEY,
    canonical_code VARCHAR(20) NOT NULL UNIQUE,  -- e.g., "03-CONCRETE"
    csi_division VARCHAR(2) NOT NULL,            -- e.g., "03"
    canonical_name VARCHAR(200) NOT NULL,        -- e.g., "Concrete Work"
    parent_trade_id INTEGER REFERENCES canonical_trades(id),
    hierarchy_level INTEGER NOT NULL DEFAULT 1,  -- 1=Division, 2=Subdivision, 3=Detail
    typical_pct_of_total FLOAT,                  -- Historical average % of GMP
    typical_duration_pct FLOAT,                  -- Typical % of project duration
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Project-to-Canonical Trade Mapping
CREATE TABLE project_trade_mappings (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    raw_division_name VARCHAR(200) NOT NULL,     -- Original name from project
    canonical_trade_id INTEGER NOT NULL REFERENCES canonical_trades(id),
    confidence FLOAT DEFAULT 1.0,                -- Mapping confidence
    mapping_method VARCHAR(30) NOT NULL,         -- 'manual', 'fuzzy', 'exact'
    created_by VARCHAR(100) DEFAULT 'system',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_id, raw_division_name)
);

-- Normalized Cost Features (Canonical Schema)
CREATE TABLE canonical_cost_features (
    id BIGSERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    canonical_trade_id INTEGER NOT NULL REFERENCES canonical_trades(id),
    period_date DATE NOT NULL,
    period_type VARCHAR(10) NOT NULL,            -- 'weekly', 'monthly'

    -- Normalized cost metrics (per square foot)
    cost_per_sf_cents INTEGER NOT NULL,          -- Actual cost / project SF
    cumulative_cost_per_sf_cents INTEGER NOT NULL,
    budget_per_sf_cents INTEGER NOT NULL,

    -- Progress metrics
    pct_complete FLOAT,                          -- 0.0-1.0
    schedule_pct_elapsed FLOAT,                  -- Time elapsed / duration

    -- Regional splits (normalized)
    pct_east FLOAT DEFAULT 0.5,
    pct_west FLOAT DEFAULT 0.5,

    -- Metadata
    is_backfill BOOLEAN DEFAULT FALSE,           -- Historical vs. real-time
    created_at TIMESTAMP DEFAULT NOW(),

    -- Partitioning key
    CONSTRAINT pk_canonical_cost PRIMARY KEY (project_id, canonical_trade_id, period_date, period_type)
) PARTITION BY RANGE (period_date);

-- Project Embeddings (for ML)
CREATE TABLE project_embeddings (
    project_id INTEGER PRIMARY KEY REFERENCES projects(id),
    embedding_vector FLOAT[] NOT NULL,           -- Learned project representation
    model_version VARCHAR(50) NOT NULL,
    computed_at TIMESTAMP DEFAULT NOW()
);

-- Cross-Project Training Datasets
CREATE TABLE training_datasets (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    project_ids INTEGER[] NOT NULL,              -- Projects included
    canonical_trade_ids INTEGER[],               -- Trades included (null = all)
    date_range_start DATE NOT NULL,
    date_range_end DATE NOT NULL,
    sample_count INTEGER NOT NULL,
    feature_schema JSONB NOT NULL,               -- Column definitions
    storage_path VARCHAR(500) NOT NULL,          -- S3/local path
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model Registry Extension
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,             -- 'global', 'project_finetuned', 'trade_specific'
    scope_project_id INTEGER REFERENCES projects(id),  -- NULL for global
    scope_canonical_trade_id INTEGER REFERENCES canonical_trades(id),  -- NULL for all-trade
    training_dataset_id INTEGER REFERENCES training_datasets(id),
    artifact_path VARCHAR(500) NOT NULL,
    metrics JSONB NOT NULL,                      -- MAE, MAPE, coverage, etc.
    hyperparameters JSONB,
    is_production BOOLEAN DEFAULT FALSE,
    promoted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_name, model_version)
);

-- Forecast Isolation (per-project forecasts)
CREATE TABLE project_forecasts (
    id BIGSERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    canonical_trade_id INTEGER NOT NULL REFERENCES canonical_trades(id),
    model_id INTEGER NOT NULL REFERENCES model_registry(id),
    forecast_date DATE NOT NULL,
    horizon_months INTEGER NOT NULL,

    -- Predictions
    predicted_eac_cents BIGINT NOT NULL,
    predicted_etc_cents BIGINT NOT NULL,
    confidence_lower_cents BIGINT,
    confidence_upper_cents BIGINT,
    confidence_level FLOAT DEFAULT 0.8,

    -- Regional breakdown
    eac_east_cents BIGINT,
    eac_west_cents BIGINT,

    -- Metadata
    is_current BOOLEAN DEFAULT TRUE,
    superseded_by_id BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(project_id, canonical_trade_id, forecast_date, model_id)
);
```

#### 3.1.2 Modified Existing Tables

```sql
-- Add to projects table
ALTER TABLE projects ADD COLUMN IF NOT EXISTS
    total_square_feet INTEGER,
    project_type VARCHAR(50),              -- 'commercial', 'residential', 'mixed'
    region VARCHAR(50),                    -- Geographic region
    owner_id INTEGER,                      -- For multi-owner scenarios
    is_training_eligible BOOLEAN DEFAULT TRUE,  -- Include in global training
    data_quality_score FLOAT;              -- 0.0-1.0

-- Add to gmp_entities table
ALTER TABLE gmp_entities ADD COLUMN IF NOT EXISTS
    canonical_trade_id INTEGER REFERENCES canonical_trades(id),
    normalized_amount_per_sf_cents INTEGER;  -- For cross-project comparison
```

### 3.2 Trade Taxonomy Normalization

#### 3.2.1 Canonical Trade Hierarchy

```
Level 1 (CSI Division):
├── 01-GENERAL_REQUIREMENTS
├── 02-EXISTING_CONDITIONS
├── 03-CONCRETE
├── 04-MASONRY
├── 05-METALS
├── 06-WOOD_PLASTICS_COMPOSITES
├── 07-THERMAL_MOISTURE
├── 08-OPENINGS
├── 09-FINISHES
├── 10-SPECIALTIES
├── 11-EQUIPMENT
├── 12-FURNISHINGS
├── 13-SPECIAL_CONSTRUCTION
├── 14-CONVEYING_EQUIPMENT
├── 21-FIRE_SUPPRESSION
├── 22-PLUMBING
├── 23-HVAC
├── 26-ELECTRICAL
├── 27-COMMUNICATIONS
├── 28-ELECTRONIC_SAFETY
├── 31-EARTHWORK
├── 32-EXTERIOR_IMPROVEMENTS
└── 33-UTILITIES

Level 2 (Subdivision): e.g., 03-CONCRETE → 0310-FORMWORK, 0320-REINFORCING, 0330-CAST_IN_PLACE
Level 3 (Detail): e.g., 0330-CAST_IN_PLACE → 033000-STRUCTURAL, 033100-LIGHTWEIGHT
```

#### 3.2.2 Mapping Algorithm

```python
def map_raw_division_to_canonical(raw_name: str, project_id: int) -> CanonicalTradeMapping:
    """
    Multi-stage mapping from project-specific division names to canonical trades.

    Priority order:
    1. Exact match in project_trade_mappings (user-confirmed)
    2. Exact match on canonical_code prefix (e.g., "03-" → 03-CONCRETE)
    3. Fuzzy match on canonical_name (RapidFuzz, threshold=85)
    4. Manual review queue (confidence < 0.7)
    """
    # Stage 1: Check existing confirmed mapping
    existing = db.query(ProjectTradeMapping).filter_by(
        project_id=project_id, raw_division_name=raw_name
    ).first()
    if existing:
        return existing

    # Stage 2: CSI prefix extraction
    csi_match = re.match(r'^(\d{2})-', raw_name)
    if csi_match:
        csi_div = csi_match.group(1)
        canonical = db.query(CanonicalTrade).filter_by(csi_division=csi_div).first()
        if canonical:
            return create_mapping(raw_name, canonical, method='exact', confidence=0.95)

    # Stage 3: Fuzzy matching
    all_canonicals = db.query(CanonicalTrade).all()
    best_match, score = rapidfuzz.extractOne(
        raw_name,
        [c.canonical_name for c in all_canonicals],
        scorer=rapidfuzz.fuzz.token_sort_ratio
    )
    if score >= 85:
        canonical = next(c for c in all_canonicals if c.canonical_name == best_match)
        return create_mapping(raw_name, canonical, method='fuzzy', confidence=score/100)

    # Stage 4: Queue for manual review
    return create_pending_mapping(raw_name, project_id, top_candidates=all_canonicals[:5])
```

### 3.3 Canonical Schema Specification

All projects map into this standardized feature schema:

```python
CANONICAL_FEATURE_SCHEMA = {
    # Identifiers
    "project_id": "int64",
    "canonical_trade_id": "int64",
    "period_date": "datetime64[D]",

    # Core Cost Features (normalized per SF)
    "cost_per_sf_cents": "int64",           # Period cost / project SF
    "cumulative_cost_per_sf_cents": "int64",# Running total / SF
    "budget_per_sf_cents": "int64",         # Trade budget / SF
    "committed_per_sf_cents": "int64",      # Committed / SF

    # Progress Features
    "pct_budget_spent": "float32",          # cumulative / budget
    "pct_schedule_elapsed": "float32",      # (period_date - start) / duration
    "pct_complete_reported": "float32",     # From schedule (if available)

    # Temporal Features
    "month_of_year": "int8",                # 1-12
    "quarter": "int8",                      # 1-4
    "project_month": "int16",               # Months since project start
    "trade_month": "int16",                 # Months since trade started

    # Project Context (static per project)
    "project_total_sf": "int32",
    "project_duration_months": "int16",
    "project_type_encoded": "int8",         # 0=commercial, 1=residential, 2=mixed

    # Trade Context (static per trade)
    "trade_budget_pct_of_total": "float32", # This trade's % of total GMP
    "trade_typical_pct": "float32",         # Historical avg % (from canonical_trades)

    # Regional
    "pct_east": "float32",
    "pct_west": "float32",
}
```

---

## 4. Modeling Strategy

### 4.1 Approach: Hierarchical Transfer Learning

We adopt a **two-stage hierarchical approach**:

1. **Global Foundation Model**: Trained on all historical project data in canonical schema
2. **Project-Specific Adapters**: Lightweight fine-tuning layers for active projects

```
┌─────────────────────────────────────────────────────────────────┐
│                    GLOBAL FOUNDATION MODEL                       │
│                                                                  │
│  Input: Canonical Features (all projects, all trades)           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Shared Feature Encoder (frozen after global training)   │    │
│  │  - LSTM/Transformer backbone (seq_len=12 months)         │    │
│  │  - Learns universal cost patterns across construction    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Project Embedding Layer                                 │    │
│  │  - Learned embedding per project (dim=32)                │    │
│  │  - Captures project-specific characteristics             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Trade Embedding Layer                                   │    │
│  │  - Learned embedding per canonical trade (dim=16)        │    │
│  │  - Captures trade-specific cost dynamics                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              PROJECT-SPECIFIC ADAPTER (per project)              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Adapter Layer (trainable, ~10K params)                  │    │
│  │  - 2-layer MLP with project-specific weights             │    │
│  │  - Fine-tuned on project's recent data only              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Probabilistic Output Head                               │    │
│  │  - Gaussian Mixture (mean + std for uncertainty)         │    │
│  │  - Per-trade EAC prediction                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Model Architecture

```python
class MultiProjectForecaster(tf.keras.Model):
    """
    Hierarchical model for cross-project cost forecasting.
    """
    def __init__(
        self,
        num_projects: int,
        num_trades: int,
        seq_len: int = 12,
        feature_dim: int = 24,
        project_embed_dim: int = 32,
        trade_embed_dim: int = 16,
        lstm_units: int = 64,
        adapter_units: int = 32,
    ):
        super().__init__()

        # Embeddings
        self.project_embedding = tf.keras.layers.Embedding(
            num_projects, project_embed_dim, name='project_embed'
        )
        self.trade_embedding = tf.keras.layers.Embedding(
            num_trades, trade_embed_dim, name='trade_embed'
        )

        # Shared encoder (frozen after global training)
        self.temporal_encoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True),
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            tf.keras.layers.LayerNormalization(),
        ], name='temporal_encoder')

        # Feature fusion
        self.fusion = tf.keras.layers.Dense(64, activation='relu')

        # Project-specific adapter (fine-tuned per project)
        self.adapter = tf.keras.Sequential([
            tf.keras.layers.Dense(adapter_units, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(adapter_units, activation='relu'),
        ], name='adapter')

        # Probabilistic output (mean + log_std)
        self.output_head = tf.keras.layers.Dense(2, name='output')  # [mean, log_std]

    def call(self, inputs, training=False):
        seq_features, project_id, trade_id = inputs

        # Encode temporal sequence
        temporal_encoding = self.temporal_encoder(seq_features, training=training)

        # Get embeddings
        proj_embed = self.project_embedding(project_id)
        trade_embed = self.trade_embedding(trade_id)

        # Fuse all features
        combined = tf.concat([
            temporal_encoding,
            tf.squeeze(proj_embed, axis=1),
            tf.squeeze(trade_embed, axis=1)
        ], axis=-1)

        fused = self.fusion(combined)

        # Project-specific adaptation
        adapted = self.adapter(fused, training=training)

        # Probabilistic output
        output = self.output_head(adapted)
        mean, log_std = tf.split(output, 2, axis=-1)
        std = tf.nn.softplus(log_std) + 1e-6  # Ensure positive

        return mean, std
```

### 4.3 Training Strategy

#### 4.3.1 Global Model Training

```python
def train_global_model(
    dataset: TrainingDataset,
    model: MultiProjectForecaster,
    epochs: int = 100,
    batch_size: int = 64,
) -> TrainingResult:
    """
    Train global foundation model on all historical projects.

    Key considerations:
    - Time-aware splitting: train on data before cutoff, validate after
    - Project-aware sampling: stratified by project to prevent dominance
    - Leakage prevention: no future data in training sequences
    """

    # Time-based split (critical for leakage prevention)
    cutoff_date = dataset.date_range_end - timedelta(months=6)
    train_data = dataset.filter(period_date < cutoff_date)
    val_data = dataset.filter(period_date >= cutoff_date)

    # Stratified sampling by project
    train_sampler = ProjectStratifiedSampler(train_data, batch_size)

    # Loss: Gaussian NLL for probabilistic output
    def gaussian_nll_loss(y_true, y_pred_mean, y_pred_std):
        variance = tf.square(y_pred_std)
        log_likelihood = -0.5 * (
            tf.math.log(2 * np.pi * variance) +
            tf.square(y_true - y_pred_mean) / variance
        )
        return -tf.reduce_mean(log_likelihood)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    for epoch in range(epochs):
        for batch in train_sampler:
            with tf.GradientTape() as tape:
                mean, std = model(batch.features, training=True)
                loss = gaussian_nll_loss(batch.targets, mean, std)

            gradients = tape.gradient(loss, model.trainable_variables)
            # Gradient clipping for stability
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Validation with project-level metrics
        val_metrics = evaluate_per_project(model, val_data)

        # Early stopping on worst-project MAPE (ensure no project left behind)
        if val_metrics.worst_project_mape < best_worst_mape:
            save_checkpoint(model, epoch)

    return TrainingResult(model, val_metrics)
```

#### 4.3.2 Project-Specific Fine-Tuning

```python
def finetune_for_project(
    global_model: MultiProjectForecaster,
    project_id: int,
    project_data: ProjectDataset,
    epochs: int = 20,
) -> ProjectAdapter:
    """
    Fine-tune adapter layers for a specific project.

    Key points:
    - Freeze encoder and embeddings (global knowledge)
    - Only train adapter layers (project-specific adjustment)
    - Use only this project's recent data
    """

    # Freeze global components
    global_model.temporal_encoder.trainable = False
    global_model.project_embedding.trainable = False
    global_model.trade_embedding.trainable = False
    global_model.fusion.trainable = False

    # Only adapter and output head are trainable
    global_model.adapter.trainable = True
    global_model.output_head.trainable = True

    # Use recent data only (last 12 months)
    recent_cutoff = datetime.now() - timedelta(months=12)
    recent_data = project_data.filter(period_date >= recent_cutoff)

    # Fine-tune with lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(epochs):
        # ... training loop ...
        pass

    # Save project-specific adapter weights
    adapter_weights = {
        'adapter': global_model.adapter.get_weights(),
        'output_head': global_model.output_head.get_weights(),
    }

    return ProjectAdapter(project_id, adapter_weights)
```

### 4.4 Leakage Prevention

#### 4.4.1 Temporal Leakage Prevention

```python
class TemporalSplitter:
    """
    Ensures strict temporal ordering to prevent future data leakage.
    """

    def create_sequences(
        self,
        data: pd.DataFrame,
        seq_len: int = 12,
        forecast_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training sequences with strict temporal boundaries.

        Rules:
        1. Input sequence: months [t-seq_len+1, t]
        2. Target: month [t+forecast_horizon]
        3. No overlap between train/val input and target periods
        """
        sequences = []
        targets = []

        # Sort by project, trade, date
        data = data.sort_values(['project_id', 'canonical_trade_id', 'period_date'])

        for (proj, trade), group in data.groupby(['project_id', 'canonical_trade_id']):
            values = group.values
            dates = group['period_date'].values

            for i in range(seq_len, len(values) - forecast_horizon):
                seq = values[i-seq_len:i]
                target = values[i + forecast_horizon - 1]

                # Verify temporal ordering
                assert dates[i-1] < dates[i + forecast_horizon - 1]

                sequences.append(seq)
                targets.append(target)

        return np.array(sequences), np.array(targets)

    def train_val_split(
        self,
        data: pd.DataFrame,
        val_months: int = 6,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Time-based split ensuring no temporal leakage.

        - Train: all data before cutoff
        - Validation: all data after cutoff (simulates production)
        """
        max_date = data['period_date'].max()
        cutoff = max_date - pd.DateOffset(months=val_months)

        train = data[data['period_date'] < cutoff]
        val = data[data['period_date'] >= cutoff]

        # Verify no overlap
        assert train['period_date'].max() < val['period_date'].min()

        return train, val
```

#### 4.4.2 Cross-Project Leakage Prevention

```python
class ProjectIsolator:
    """
    Ensures project data isolation during training and inference.
    """

    def validate_no_cross_project_leakage(
        self,
        model: MultiProjectForecaster,
        test_project_id: int,
    ) -> bool:
        """
        Verify that predictions for one project don't depend on
        other projects' concurrent data.
        """

        # Get prediction for project A at time T
        pred_a = model.predict(project_id=test_project_id, as_of_date=T)

        # Modify project B's data at time T
        modify_project_data(project_id=other_project_id, date=T)

        # Re-predict for project A
        pred_a_after = model.predict(project_id=test_project_id, as_of_date=T)

        # Predictions should be identical
        assert np.allclose(pred_a, pred_a_after), "Cross-project leakage detected!"

        return True

    def create_leave_one_project_out_cv(
        self,
        data: pd.DataFrame,
        target_project_id: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For new project evaluation, train on all other projects.
        """
        train = data[data['project_id'] != target_project_id]
        test = data[data['project_id'] == target_project_id]

        return train, test
```

---

## 5. Pipelines & MLOps

### 5.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE (Per Project)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [1. INGESTION]     [2. VALIDATION]    [3. CANONICALIZATION]                │
│       │                   │                    │                             │
│       ▼                   ▼                    ▼                             │
│  ┌─────────┐        ┌──────────┐        ┌─────────────┐                     │
│  │ Procore │───────▶│ Schema   │───────▶│ Trade       │                     │
│  │ P6      │        │ Validator│        │ Normalizer  │                     │
│  │ Excel   │        │ (Pydantic)│       │ (Taxonomy)  │                     │
│  └─────────┘        └──────────┘        └─────────────┘                     │
│                           │                    │                             │
│                           ▼                    ▼                             │
│                    ┌──────────┐        ┌─────────────┐                      │
│                    │ Quality  │        │ Canonical   │                      │
│                    │ Scores   │        │ Store       │                      │
│                    └──────────┘        └─────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FEATURE PIPELINE (Batch + Real-time)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [4. FEATURE COMPUTATION]                                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  OFFLINE (Batch - Daily)                                       │         │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │         │
│  │  │ Rolling      │  │ Lag          │  │ Aggregation  │         │         │
│  │  │ Statistics   │  │ Features     │  │ (Weekly/Mo)  │         │         │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                            │                                                 │
│                            ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  FEATURE STORE                                                 │         │
│  │  - Offline: Parquet in S3 (historical)                         │         │
│  │  - Online: Redis (latest features for inference)               │         │
│  │  - Schema: canonical_feature_schema v1.0                       │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE (Scheduled)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [5. DATASET CREATION]   [6. TRAINING]      [7. EVALUATION]                 │
│          │                    │                   │                          │
│          ▼                    ▼                   ▼                          │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│  │ Cross-Project │   │ Global Model  │   │ Backtesting   │                  │
│  │ Dataset       │──▶│ Training      │──▶│ (Per Project) │                  │
│  │ Assembly      │   │ (GPU Cluster) │   │               │                  │
│  └───────────────┘   └───────────────┘   └───────────────┘                  │
│                                                 │                            │
│                                                 ▼                            │
│                                    ┌───────────────┐                        │
│                                    │ Model Registry│                        │
│                                    │ (MLflow)      │                        │
│                                    └───────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVING PIPELINE (Real-time)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [8. MODEL LOADING]     [9. INFERENCE]       [10. MONITORING]               │
│          │                    │                    │                         │
│          ▼                    ▼                    ▼                         │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│  │ Load Global + │   │ Per-Project   │   │ Drift         │                  │
│  │ Project       │──▶│ Forecast      │──▶│ Detection     │                  │
│  │ Adapter       │   │ Generation    │   │ (Evidently)   │                  │
│  └───────────────┘   └───────────────┘   └───────────────┘                  │
│                                                 │                            │
│                                                 ▼                            │
│                                    ┌───────────────┐                        │
│                                    │ Alerts &      │                        │
│                                    │ Retraining    │                        │
│                                    └───────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Orchestration (Airflow DAGs)

```python
# dag_global_training.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-platform',
    'retries': 2,
    'retry_delay': timedelta(minutes=15),
}

with DAG(
    'global_model_training',
    default_args=default_args,
    schedule_interval='0 2 * * 0',  # Weekly, Sunday 2 AM
    catchup=False,
) as dag:

    validate_data_quality = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_all_projects_data_quality,
    )

    create_training_dataset = PythonOperator(
        task_id='create_training_dataset',
        python_callable=assemble_cross_project_dataset,
    )

    train_global_model = PythonOperator(
        task_id='train_global_model',
        python_callable=train_global_foundation_model,
        execution_timeout=timedelta(hours=4),
    )

    backtest_all_projects = PythonOperator(
        task_id='backtest_all_projects',
        python_callable=run_backtesting_suite,
    )

    evaluate_and_promote = PythonOperator(
        task_id='evaluate_and_promote',
        python_callable=evaluate_and_promote_if_better,
    )

    (validate_data_quality
     >> create_training_dataset
     >> train_global_model
     >> backtest_all_projects
     >> evaluate_and_promote)


# dag_project_finetuning.py
with DAG(
    'project_finetuning',
    default_args=default_args,
    schedule_interval='0 3 * * *',  # Daily, 3 AM
    catchup=False,
) as dag:

    detect_projects_needing_finetuning = PythonOperator(
        task_id='detect_projects',
        python_callable=identify_stale_project_models,
    )

    finetune_project_adapters = PythonOperator(
        task_id='finetune_adapters',
        python_callable=finetune_all_stale_projects,
    )

    deploy_updated_adapters = PythonOperator(
        task_id='deploy_adapters',
        python_callable=deploy_project_adapters_to_inference,
    )

    (detect_projects_needing_finetuning
     >> finetune_project_adapters
     >> deploy_updated_adapters)
```

### 5.3 Feature Store Implementation

```python
class FeatureStore:
    """
    Offline/online feature store with parity guarantees.
    """

    def __init__(self, offline_path: str, redis_client: Redis):
        self.offline_path = offline_path
        self.redis = redis_client
        self.schema_version = "1.0"

    def compute_features_batch(
        self,
        project_id: int,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Compute features for a date range (offline batch).
        """
        # Load raw data
        costs = load_canonical_costs(project_id, start_date, end_date)

        # Compute features using canonical schema
        features = self._compute_feature_set(costs)

        # Validate schema
        self._validate_schema(features)

        # Store offline
        self._store_offline(features, project_id)

        # Update online store with latest
        self._update_online_store(features, project_id)

        return features

    def get_features_online(
        self,
        project_id: int,
        trade_id: int,
        seq_len: int = 12,
    ) -> np.ndarray:
        """
        Retrieve latest features for inference (online).
        """
        key = f"features:{project_id}:{trade_id}"
        cached = self.redis.get(key)

        if cached:
            return np.frombuffer(cached, dtype=np.float32).reshape(-1, seq_len)

        # Fallback to offline store
        return self._load_from_offline(project_id, trade_id, seq_len)

    def _compute_feature_set(self, costs: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features matching CANONICAL_FEATURE_SCHEMA.
        """
        features = costs.copy()

        # Rolling statistics (3-month window)
        for col in ['cost_per_sf_cents']:
            features[f'{col}_rolling_mean_3m'] = (
                features.groupby(['project_id', 'canonical_trade_id'])[col]
                .rolling(3, min_periods=1).mean().reset_index(drop=True)
            )
            features[f'{col}_rolling_std_3m'] = (
                features.groupby(['project_id', 'canonical_trade_id'])[col]
                .rolling(3, min_periods=1).std().reset_index(drop=True)
            )

        # Lag features
        for lag in [1, 2, 3]:
            features[f'cost_lag_{lag}'] = (
                features.groupby(['project_id', 'canonical_trade_id'])['cost_per_sf_cents']
                .shift(lag)
            )

        # Progress features
        features['pct_budget_spent'] = (
            features['cumulative_cost_per_sf_cents'] /
            features['budget_per_sf_cents'].clip(lower=1)
        ).clip(0, 2)  # Cap at 200% overspend

        # Temporal features
        features['month_of_year'] = features['period_date'].dt.month
        features['quarter'] = features['period_date'].dt.quarter

        return features
```

### 5.4 Model Registry & Versioning

```python
class ModelRegistry:
    """
    MLflow-based model registry with promotion workflow.
    """

    def __init__(self, mlflow_uri: str):
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()

    def register_model(
        self,
        model: MultiProjectForecaster,
        model_type: str,  # 'global', 'project_adapter'
        metrics: Dict[str, float],
        project_id: Optional[int] = None,
    ) -> str:
        """
        Register a trained model with full lineage.
        """
        with mlflow.start_run() as run:
            # Log model
            mlflow.tensorflow.log_model(model, "model")

            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("project_id", project_id or "global")
            mlflow.log_param("schema_version", CANONICAL_FEATURE_SCHEMA_VERSION)

            # Log training data lineage
            mlflow.log_param("training_dataset_id", metrics.get('dataset_id'))

            # Register model version
            model_name = f"gmp_forecaster_{model_type}"
            if project_id:
                model_name = f"{model_name}_project_{project_id}"

            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)

            return mv.version

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        approval_notes: str,
    ) -> bool:
        """
        Promote model version to production after approval.
        """
        # Archive current production model
        current_prod = self._get_production_version(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod,
                stage="Archived"
            )

        # Promote new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        # Log promotion
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="promotion_notes",
            value=approval_notes
        )

        return True
```

### 5.5 Monitoring & Retraining Triggers

```python
class DriftDetector:
    """
    Monitor for data and model drift.
    """

    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data
        self.thresholds = {
            'psi_threshold': 0.2,        # Population Stability Index
            'mape_degradation': 0.1,     # 10% MAPE increase triggers alert
            'feature_drift_ratio': 0.15, # 15% of features drifted
        }

    def check_drift(
        self,
        current_data: pd.DataFrame,
        project_id: int,
    ) -> DriftReport:
        """
        Check for data drift in a project's features.
        """
        report = DriftReport(project_id=project_id)

        # Feature drift (PSI per feature)
        for col in CANONICAL_FEATURE_SCHEMA.keys():
            if col in current_data.columns:
                psi = self._calculate_psi(
                    self.reference[col],
                    current_data[col]
                )
                if psi > self.thresholds['psi_threshold']:
                    report.drifted_features.append((col, psi))

        # Check if too many features drifted
        drift_ratio = len(report.drifted_features) / len(CANONICAL_FEATURE_SCHEMA)
        report.requires_retraining = drift_ratio > self.thresholds['feature_drift_ratio']

        return report

    def check_prediction_drift(
        self,
        project_id: int,
        recent_window_days: int = 30,
    ) -> PredictionDriftReport:
        """
        Compare recent predictions to actuals.
        """
        # Load recent predictions and actuals
        predictions = load_recent_predictions(project_id, recent_window_days)
        actuals = load_recent_actuals(project_id, recent_window_days)

        # Calculate MAPE
        mape = np.mean(np.abs(predictions - actuals) / np.clip(actuals, 1, None))

        # Compare to baseline MAPE
        baseline_mape = get_baseline_mape(project_id)
        degradation = (mape - baseline_mape) / baseline_mape

        report = PredictionDriftReport(
            project_id=project_id,
            current_mape=mape,
            baseline_mape=baseline_mape,
            degradation=degradation,
            requires_finetuning=degradation > self.thresholds['mape_degradation']
        )

        return report


class RetrainingTrigger:
    """
    Automated retraining triggers.
    """

    TRIGGERS = {
        'data_drift': 'Significant feature distribution shift detected',
        'prediction_degradation': 'MAPE increased >10% from baseline',
        'new_project': 'New project added, requires adapter',
        'scheduled': 'Weekly global model refresh',
        'user_feedback': 'User reported poor forecasts',
        'data_volume': 'New project has >6 months of data',
    }

    def evaluate_triggers(self, project_id: int) -> List[str]:
        """
        Evaluate all retraining triggers for a project.
        """
        triggered = []

        # Check data drift
        drift_report = self.drift_detector.check_drift(
            current_data=load_recent_data(project_id),
            project_id=project_id
        )
        if drift_report.requires_retraining:
            triggered.append('data_drift')

        # Check prediction degradation
        pred_drift = self.drift_detector.check_prediction_drift(project_id)
        if pred_drift.requires_finetuning:
            triggered.append('prediction_degradation')

        # Check data volume threshold
        data_months = get_project_data_months(project_id)
        if data_months >= 6 and not has_finetuned_adapter(project_id):
            triggered.append('data_volume')

        return triggered
```

---

## 6. APIs/UI Changes

### 6.1 New API Endpoints

```python
# app/api/v2/projects.py

@router.post("/projects", response_model=ProjectResponse)
def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new project with automatic taxonomy mapping.
    """
    # Create project
    db_project = Project(**project.dict())
    db.add(db_project)
    db.commit()

    # Initialize project embedding (placeholder until training)
    init_project_embedding(db_project.id)

    # Create default forecast config
    create_default_forecast_config(db_project.id)

    return db_project


@router.get("/projects/{project_id}/forecasts", response_model=List[TradeForecast])
def get_project_forecasts(
    project_id: int,
    as_of_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get current forecasts for all trades in a project.
    """
    # Verify project access
    verify_project_access(current_user, project_id)

    # Load project-specific model
    model = load_project_model(project_id)

    # Get latest features
    features = feature_store.get_features_online(project_id)

    # Generate forecasts
    forecasts = model.predict(features, as_of_date=as_of_date or date.today())

    return [TradeForecast.from_prediction(f) for f in forecasts]


@router.post("/projects/{project_id}/train", response_model=TrainingJob)
def trigger_project_training(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Trigger fine-tuning for a specific project.
    """
    verify_project_access(current_user, project_id)

    # Check if enough data
    data_months = get_project_data_months(project_id)
    if data_months < 3:
        raise HTTPException(400, "Insufficient data for training (need 3+ months)")

    # Queue training job
    job = TrainingJob(
        project_id=project_id,
        triggered_by=current_user.id,
        status='queued'
    )
    db.add(job)
    db.commit()

    # Start background training
    background_tasks.add_task(finetune_project_adapter, project_id, job.id)

    return job


# app/api/v2/taxonomy.py

@router.get("/taxonomy/trades", response_model=List[CanonicalTrade])
def list_canonical_trades(
    level: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    List all canonical trades in the taxonomy.
    """
    query = db.query(CanonicalTrade)
    if level:
        query = query.filter(CanonicalTrade.hierarchy_level == level)
    return query.all()


@router.post("/taxonomy/mappings", response_model=TradeMapping)
def create_trade_mapping(
    mapping: TradeMappingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Manually map a project's trade name to canonical taxonomy.
    """
    verify_project_access(current_user, mapping.project_id)

    db_mapping = ProjectTradeMapping(
        **mapping.dict(),
        mapping_method='manual',
        confidence=1.0,
        created_by=current_user.email
    )
    db.add(db_mapping)
    db.commit()

    # Trigger feature recomputation
    invalidate_project_features(mapping.project_id)

    return db_mapping


# app/api/v2/analytics.py

@router.get("/analytics/cross-project", response_model=CrossProjectAnalytics)
def get_cross_project_analytics(
    trade_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get cross-project analytics for benchmarking.

    Returns aggregated statistics across all accessible projects.
    """
    accessible_projects = get_user_accessible_projects(current_user)

    analytics = compute_cross_project_stats(
        project_ids=accessible_projects,
        trade_id=trade_id,
    )

    return analytics
```

### 6.2 UI Changes

#### 6.2.1 Project Selector

```typescript
// components/ProjectSelector.tsx
export const ProjectSelector: React.FC = () => {
  const { projects, activeProject, setActiveProject } = useProjectContext();

  return (
    <Select
      value={activeProject?.id}
      onChange={(id) => setActiveProject(projects.find(p => p.id === id))}
    >
      {projects.map(project => (
        <SelectItem key={project.id} value={project.id}>
          <ProjectBadge project={project} />
          <span>{project.name}</span>
          <span className="text-muted">{project.code}</span>
        </SelectItem>
      ))}
    </Select>
  );
};
```

#### 6.2.2 Trade Mapping Review UI

```typescript
// components/TradeMappingReview.tsx
export const TradeMappingReview: React.FC<{ projectId: number }> = ({ projectId }) => {
  const { pendingMappings } = usePendingMappings(projectId);

  return (
    <Card>
      <CardHeader>
        <h3>Pending Trade Mappings</h3>
        <p>Review and confirm trade mappings to canonical taxonomy</p>
      </CardHeader>
      <CardContent>
        {pendingMappings.map(mapping => (
          <MappingRow key={mapping.id}>
            <div className="source">
              <Label>Project Trade</Label>
              <span>{mapping.raw_division_name}</span>
            </div>
            <ArrowRight />
            <div className="target">
              <Label>Canonical Trade</Label>
              <TradeSelect
                value={mapping.suggested_trade_id}
                options={canonicalTrades}
                confidence={mapping.confidence}
                onChange={(tradeId) => updateMapping(mapping.id, tradeId)}
              />
            </div>
            <ConfirmButton onClick={() => confirmMapping(mapping.id)} />
          </MappingRow>
        ))}
      </CardContent>
    </Card>
  );
};
```

#### 6.2.3 Cross-Project Benchmarking Dashboard

```typescript
// components/CrossProjectBenchmark.tsx
export const CrossProjectBenchmark: React.FC = () => {
  const { analytics } = useCrossProjectAnalytics();

  return (
    <Dashboard>
      <MetricCard
        title="Cost/SF Distribution by Trade"
        chart={
          <BoxPlot
            data={analytics.tradeStats}
            xField="canonical_trade"
            yField="cost_per_sf"
            highlightProject={activeProject.id}
          />
        }
      />

      <MetricCard
        title="Forecast Accuracy by Project"
        chart={
          <ScatterPlot
            data={analytics.projectAccuracy}
            xField="data_months"
            yField="mape"
            sizeField="total_gmp"
            colorField="project_type"
          />
        }
      />

      <MetricCard
        title="Trade Completion Curves"
        chart={
          <LineChart
            data={analytics.completionCurves}
            xField="pct_schedule_elapsed"
            yField="pct_cost_spent"
            groupField="project_id"
            showConfidenceBand
          />
        }
      />
    </Dashboard>
  );
};
```

---

## 7. Migration Plan

### 7.1 Phase Overview

```
Phase 1: Foundation (Weeks 1-4)
├── Database schema migration
├── Trade taxonomy setup
└── Canonical data model

Phase 2: Data Migration (Weeks 5-8)
├── Historical project backfill
├── Trade mapping review
└── Feature store population

Phase 3: ML Pipeline (Weeks 9-14)
├── Global model training
├── Evaluation framework
└── Project adapter system

Phase 4: Integration (Weeks 15-18)
├── API v2 deployment
├── UI updates
└── Monitoring setup

Phase 5: Cutover (Weeks 19-20)
├── Shadow mode testing
├── Production cutover
└── Legacy deprecation
```

### 7.2 Detailed Migration Steps

#### 7.2.1 Phase 1: Foundation

```sql
-- Step 1.1: Add new tables (non-breaking)
CREATE TABLE canonical_trades (...);
CREATE TABLE project_trade_mappings (...);
CREATE TABLE canonical_cost_features (...);
CREATE TABLE model_registry (...);
CREATE TABLE project_forecasts (...);

-- Step 1.2: Add columns to existing tables
ALTER TABLE projects ADD COLUMN total_square_feet INTEGER;
ALTER TABLE projects ADD COLUMN project_type VARCHAR(50);
ALTER TABLE projects ADD COLUMN is_training_eligible BOOLEAN DEFAULT TRUE;

ALTER TABLE gmp_entities ADD COLUMN canonical_trade_id INTEGER;
ALTER TABLE gmp_entities ADD COLUMN normalized_amount_per_sf_cents INTEGER;

-- Step 1.3: Create indexes for performance
CREATE INDEX idx_canonical_cost_project_trade_date
ON canonical_cost_features(project_id, canonical_trade_id, period_date);

CREATE INDEX idx_project_forecasts_current
ON project_forecasts(project_id, canonical_trade_id)
WHERE is_current = TRUE;
```

#### 7.2.2 Phase 2: Data Migration

```python
def migrate_existing_project(project_id: int):
    """
    Migrate existing project data to canonical format.
    """
    project = db.query(Project).get(project_id)

    # Step 2.1: Infer project metadata
    if not project.total_square_feet:
        project.total_square_feet = infer_square_feet_from_budget(project_id)

    # Step 2.2: Map trades to canonical taxonomy
    gmp_divisions = db.query(GMP).filter_by(project_id=project_id).all()
    for gmp in gmp_divisions:
        mapping = map_raw_division_to_canonical(
            raw_name=gmp.division,
            project_id=project_id
        )
        gmp.canonical_trade_id = mapping.canonical_trade_id

        # Normalize amount per SF
        if project.total_square_feet:
            gmp.normalized_amount_per_sf_cents = (
                gmp.original_amount_cents // project.total_square_feet
            )

    # Step 2.3: Backfill canonical cost features
    historical_costs = load_historical_costs(project_id)
    canonical_features = compute_canonical_features(
        historical_costs,
        project_id,
        is_backfill=True
    )
    bulk_insert(CanonicalCostFeatures, canonical_features)

    db.commit()

    return MigrationResult(
        project_id=project_id,
        trades_mapped=len(gmp_divisions),
        periods_backfilled=len(canonical_features)
    )


def run_full_migration():
    """
    Migrate all existing projects.
    """
    projects = db.query(Project).all()

    for project in tqdm(projects, desc="Migrating projects"):
        try:
            result = migrate_existing_project(project.id)
            log_migration_success(result)
        except Exception as e:
            log_migration_failure(project.id, e)
            # Continue with other projects

    # Generate migration report
    generate_migration_report()
```

#### 7.2.3 Phase 3: ML Pipeline

```python
def setup_ml_pipeline():
    """
    Initialize multi-project ML training pipeline.
    """
    # Step 3.1: Create initial training dataset
    eligible_projects = db.query(Project).filter(
        Project.is_training_eligible == True,
        Project.total_square_feet.isnot(None)
    ).all()

    dataset = create_cross_project_dataset(
        project_ids=[p.id for p in eligible_projects],
        date_range_start=date(2021, 1, 1),
        date_range_end=date.today() - timedelta(days=30)
    )

    # Step 3.2: Train global foundation model
    global_model = train_global_foundation_model(dataset)

    # Step 3.3: Register model
    model_version = model_registry.register_model(
        model=global_model,
        model_type='global',
        metrics=evaluate_model(global_model, dataset)
    )

    # Step 3.4: Create project adapters for active projects
    active_projects = [p for p in eligible_projects if p.is_active]
    for project in active_projects:
        adapter = finetune_for_project(global_model, project.id)
        model_registry.register_model(
            model=adapter,
            model_type='project_adapter',
            project_id=project.id,
            metrics=evaluate_project_adapter(adapter, project.id)
        )

    return SetupResult(
        global_model_version=model_version,
        adapters_created=len(active_projects)
    )
```

#### 7.2.4 Phase 4: Integration

```python
# Feature flag for gradual rollout
MULTI_PROJECT_ENABLED = FeatureFlag('multi_project_forecasting')

@router.get("/projects/{project_id}/forecasts")
def get_forecasts(project_id: int):
    if MULTI_PROJECT_ENABLED.is_enabled(project_id):
        # New multi-project path
        return get_forecasts_v2(project_id)
    else:
        # Legacy single-project path
        return get_forecasts_legacy(project_id)
```

#### 7.2.5 Phase 5: Cutover

```python
def execute_cutover(project_id: int):
    """
    Cut over a project from legacy to multi-project system.
    """
    # Step 5.1: Verify data quality
    quality_score = compute_data_quality_score(project_id)
    if quality_score < 0.8:
        raise CutoverBlockedError(f"Data quality {quality_score} < 0.8")

    # Step 5.2: Compare forecasts (shadow mode)
    legacy_forecasts = get_forecasts_legacy(project_id)
    new_forecasts = get_forecasts_v2(project_id)

    comparison = compare_forecasts(legacy_forecasts, new_forecasts)
    if comparison.max_divergence > 0.2:
        raise CutoverBlockedError(f"Forecast divergence {comparison.max_divergence} > 0.2")

    # Step 5.3: Enable new system
    MULTI_PROJECT_ENABLED.enable(project_id)

    # Step 5.4: Archive legacy forecasts
    archive_legacy_forecasts(project_id)

    return CutoverResult(
        project_id=project_id,
        cutover_time=datetime.now(),
        quality_score=quality_score,
        forecast_divergence=comparison.max_divergence
    )
```

### 7.3 Backward Compatibility

```python
class CompatibilityLayer:
    """
    Ensures backward compatibility during migration.
    """

    def __init__(self):
        self.legacy_api_enabled = True

    def get_gmp_division(self, gmp_id: int) -> Dict:
        """
        Return GMP in both legacy and new format.
        """
        gmp = db.query(GMP).get(gmp_id)

        response = {
            # Legacy fields (unchanged)
            'id': gmp.id,
            'division': gmp.division,
            'zone': gmp.zone,
            'original_amount_cents': gmp.original_amount_cents,

            # New fields (additive)
            'canonical_trade_id': gmp.canonical_trade_id,
            'canonical_trade_code': gmp.canonical_trade.canonical_code if gmp.canonical_trade else None,
            'normalized_amount_per_sf_cents': gmp.normalized_amount_per_sf_cents,
        }

        return response

    def create_forecast_legacy_format(self, forecast: ProjectForecast) -> Dict:
        """
        Convert new forecast format to legacy ForecastSnapshot format.
        """
        return {
            'gmp_division': forecast.canonical_trade.gmp_divisions[0].division,
            'eac_cents': forecast.predicted_eac_cents,
            'eac_east_cents': forecast.eac_east_cents,
            'eac_west_cents': forecast.eac_west_cents,
            'etc_cents': forecast.predicted_etc_cents,
            'var_cents': forecast.bac_cents - forecast.predicted_eac_cents,
            'confidence_band': self._score_to_band(forecast.confidence_level),
        }
```

### 7.4 Rollback Plan

```python
def rollback_project(project_id: int):
    """
    Rollback a project to legacy system if issues detected.
    """
    # Step 1: Disable new system
    MULTI_PROJECT_ENABLED.disable(project_id)

    # Step 2: Restore legacy forecasts from archive
    restore_legacy_forecasts(project_id)

    # Step 3: Clear new forecasts
    db.query(ProjectForecast).filter_by(project_id=project_id).delete()

    # Step 4: Log rollback
    log_rollback(project_id, reason="Manual rollback triggered")

    # Step 5: Alert team
    send_alert(f"Project {project_id} rolled back to legacy system")
```

---

## 8. Risks & Mitigations

### 8.1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Data Heterogeneity** | High | High | Canonical schema + robust normalization (per-SF costs) |
| **Concept Drift** | Medium | High | Weekly monitoring + automated retraining triggers |
| **Sparse Projects** | High | Medium | Dynamic sequence lengths (3/6/12 months) + global model fallback |
| **Privacy/Segregation** | Medium | Critical | Project-level isolation in queries + embedding-only cross-project sharing |
| **Latency Degradation** | Medium | Medium | Redis feature cache + adapter preloading |
| **Training Cost** | Medium | Medium | Scheduled weekly training + incremental fine-tuning |
| **Taxonomy Misalignment** | High | Medium | Manual review queue + confidence thresholds |

### 8.2 Detailed Mitigations

#### 8.2.1 Data Heterogeneity

**Problem:** Different projects have different scales (SF, total GMP), making cross-project comparison difficult.

**Mitigation:**
```python
def normalize_costs(costs: pd.DataFrame, project: Project) -> pd.DataFrame:
    """
    Normalize costs to per-SF basis for cross-project comparability.
    """
    if project.total_square_feet is None or project.total_square_feet < 1000:
        raise DataQualityError("Project SF required for normalization")

    costs['cost_per_sf_cents'] = (
        costs['gross_amount_cents'] / project.total_square_feet
    ).astype(int)

    # Log-transform to handle scale differences
    costs['cost_per_sf_log'] = np.log1p(costs['cost_per_sf_cents'])

    # Robust z-score (handles outliers)
    median = costs['cost_per_sf_cents'].median()
    iqr = costs['cost_per_sf_cents'].quantile(0.75) - costs['cost_per_sf_cents'].quantile(0.25)
    costs['cost_per_sf_zscore'] = (costs['cost_per_sf_cents'] - median) / (iqr + 1)

    return costs
```

#### 8.2.2 Privacy/Segregation

**Problem:** Project owners may require data isolation between projects.

**Mitigation:**
```python
class ProjectIsolationMiddleware:
    """
    Ensure all queries are scoped to authorized projects.
    """

    def __call__(self, request: Request, call_next):
        # Get user's authorized projects
        user = get_current_user(request)
        authorized_projects = get_user_project_ids(user)

        # Inject into query context
        request.state.authorized_projects = authorized_projects

        return call_next(request)


def query_with_isolation(query, model_class):
    """
    Automatically filter queries by project access.
    """
    authorized = get_authorized_projects()

    if hasattr(model_class, 'project_id'):
        query = query.filter(model_class.project_id.in_(authorized))

    return query
```

#### 8.2.3 Sparse Projects

**Problem:** New projects have insufficient data for training.

**Mitigation:**
```python
def get_model_for_project(project_id: int) -> Model:
    """
    Select appropriate model based on data availability.
    """
    data_months = get_project_data_months(project_id)

    if data_months >= 12:
        # Full project adapter
        return load_project_adapter(project_id)
    elif data_months >= 6:
        # Global model with project embedding
        return load_global_model_with_embedding(project_id)
    elif data_months >= 3:
        # Global model, no project embedding (cold start)
        return load_global_model_cold_start()
    else:
        # Heuristic baseline (historical averages)
        return HeuristicBaselineModel(project_id)
```

---

## 9. Test Plan

### 9.1 Unit Tests

```python
class TestCanonicalFeatures:
    def test_normalization_per_sf(self):
        """Verify cost normalization produces expected values."""
        costs = pd.DataFrame({
            'gross_amount_cents': [100000, 200000],
            'project_id': [1, 1]
        })
        project = Project(id=1, total_square_feet=1000)

        normalized = normalize_costs(costs, project)

        assert normalized['cost_per_sf_cents'].tolist() == [100, 200]

    def test_temporal_split_no_leakage(self):
        """Verify train/val split has no temporal overlap."""
        splitter = TemporalSplitter()
        train, val = splitter.train_val_split(sample_data, val_months=6)

        assert train['period_date'].max() < val['period_date'].min()

    def test_trade_mapping_fuzzy(self):
        """Verify fuzzy matching works for similar names."""
        mapping = map_raw_division_to_canonical("03 - Concrete Work", project_id=1)

        assert mapping.canonical_trade.canonical_code == "03-CONCRETE"
        assert mapping.confidence >= 0.85


class TestModelTraining:
    def test_global_model_trains(self):
        """Verify global model training completes without errors."""
        dataset = create_test_dataset(num_projects=3, months_per_project=24)
        model = MultiProjectForecaster(num_projects=3, num_trades=10)

        result = train_global_model(dataset, model, epochs=5)

        assert result.val_metrics['mape'] < 0.5  # Reasonable for test data

    def test_project_finetuning_improves(self):
        """Verify fine-tuning improves project-specific performance."""
        global_model = load_pretrained_global_model()
        project_data = load_test_project_data(project_id=1)

        before_mape = evaluate_model(global_model, project_data)['mape']

        finetuned = finetune_for_project(global_model, project_id=1, project_data=project_data)

        after_mape = evaluate_model(finetuned, project_data)['mape']

        assert after_mape < before_mape


class TestProjectIsolation:
    def test_no_cross_project_data_leak(self):
        """Verify project queries don't return other projects' data."""
        user = create_test_user(authorized_projects=[1, 2])

        with authorized_context(user):
            costs = query_direct_costs(project_id=1)

            # Should only contain project 1 data
            assert all(c.project_id == 1 for c in costs)

            # Should not access project 3
            with pytest.raises(UnauthorizedError):
                query_direct_costs(project_id=3)
```

### 9.2 Integration Tests

```python
class TestEndToEndPipeline:
    def test_ingestion_to_forecast(self):
        """Test full pipeline from data ingestion to forecast generation."""
        # Step 1: Ingest new project
        project = create_project(name="Test Project", total_square_feet=50000)

        # Step 2: Import GMP data
        import_gmp_data(project.id, test_gmp_csv)

        # Step 3: Import cost data
        import_direct_costs(project.id, test_costs_csv)

        # Step 4: Run canonicalization
        canonicalize_project(project.id)

        # Step 5: Verify features computed
        features = get_project_features(project.id)
        assert len(features) > 0

        # Step 6: Generate forecasts
        forecasts = get_project_forecasts(project.id)

        # Step 7: Verify forecasts are reasonable
        for forecast in forecasts:
            assert forecast.predicted_eac_cents > 0
            assert forecast.confidence_level >= 0.5

    def test_migration_preserves_forecasts(self):
        """Verify migration doesn't significantly change forecast values."""
        # Get legacy forecasts before migration
        legacy_forecasts = get_legacy_forecasts(project_id=1)

        # Run migration
        migrate_existing_project(project_id=1)

        # Get new forecasts
        new_forecasts = get_project_forecasts(project_id=1)

        # Compare
        for legacy, new in zip(legacy_forecasts, new_forecasts):
            divergence = abs(legacy.eac_cents - new.predicted_eac_cents) / legacy.eac_cents
            assert divergence < 0.15, f"Forecast diverged by {divergence:.1%}"
```

### 9.3 Performance Tests

```python
class TestPerformance:
    def test_inference_latency(self):
        """Verify inference meets <2 second SLA."""
        model = load_production_model()
        features = load_test_features(project_id=1)

        start = time.perf_counter()
        forecasts = model.predict(features)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Inference took {elapsed:.2f}s, exceeds 2s SLA"

    def test_feature_store_throughput(self):
        """Verify feature store can handle concurrent requests."""
        async def fetch_features():
            return await feature_store.get_features_online(project_id=1, trade_id=1)

        # Simulate 100 concurrent requests
        tasks = [fetch_features() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        assert all(r is not None for r in results)

    def test_training_scalability(self):
        """Verify training scales with data volume."""
        # Test with increasing dataset sizes
        times = []
        for num_projects in [5, 10, 20]:
            dataset = create_test_dataset(num_projects=num_projects)

            start = time.perf_counter()
            train_global_model(dataset, epochs=1)
            times.append(time.perf_counter() - start)

        # Time should scale sub-linearly (GPU parallelism)
        assert times[2] < times[0] * 4  # 4x projects shouldn't be 4x time
```

---

## 10. Next Steps

### 10.1 Immediate Actions (Week 1)

1. **Architecture Review:** Schedule review with stakeholders
2. **Database Migration Script:** Draft Alembic migration for new tables
3. **Trade Taxonomy:** Compile CSI-based canonical trade list
4. **Data Audit:** Inventory existing project data quality

### 10.2 Short-Term (Weeks 2-4)

1. **Prototype Canonical Schema:** Implement in dev environment
2. **Trade Mapping Algorithm:** Build fuzzy matching pipeline
3. **Feature Store POC:** Set up offline feature computation
4. **Unit Test Suite:** Achieve 80% coverage on new code

### 10.3 Medium-Term (Weeks 5-12)

1. **Global Model Training:** Train on historical projects
2. **Evaluation Framework:** Build backtesting infrastructure
3. **API v2 Development:** Implement new endpoints
4. **UI Prototypes:** Design project selector and mapping review

### 10.4 Long-Term (Weeks 13-20)

1. **Shadow Mode Deployment:** Run new system in parallel
2. **Cutover Execution:** Migrate projects incrementally
3. **Monitoring Setup:** Deploy drift detection
4. **Documentation:** Complete user and developer guides

---

## Appendix A: Canonical Trade Codes

| Code | Name | CSI Division | Level |
|------|------|--------------|-------|
| 01-GENERAL | General Requirements | 01 | 1 |
| 02-EXISTING | Existing Conditions | 02 | 1 |
| 03-CONCRETE | Concrete | 03 | 1 |
| 04-MASONRY | Masonry | 04 | 1 |
| 05-METALS | Metals | 05 | 1 |
| 06-WOOD | Wood, Plastics, Composites | 06 | 1 |
| 07-THERMAL | Thermal & Moisture Protection | 07 | 1 |
| 08-OPENINGS | Openings | 08 | 1 |
| 09-FINISHES | Finishes | 09 | 1 |
| 10-SPECIALTIES | Specialties | 10 | 1 |
| 11-EQUIPMENT | Equipment | 11 | 1 |
| 12-FURNISHINGS | Furnishings | 12 | 1 |
| 13-SPECIAL | Special Construction | 13 | 1 |
| 14-CONVEYING | Conveying Equipment | 14 | 1 |
| 21-FIRE | Fire Suppression | 21 | 1 |
| 22-PLUMBING | Plumbing | 22 | 1 |
| 23-HVAC | HVAC | 23 | 1 |
| 26-ELECTRICAL | Electrical | 26 | 1 |
| 27-COMMS | Communications | 27 | 1 |
| 28-SAFETY | Electronic Safety & Security | 28 | 1 |
| 31-EARTHWORK | Earthwork | 31 | 1 |
| 32-EXTERIOR | Exterior Improvements | 32 | 1 |
| 33-UTILITIES | Utilities | 33 | 1 |

---

## Appendix B: Feature Schema Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-16 | Initial canonical schema |

---

*Document maintained by ML Platform Team. Last updated: 2026-01-16*
