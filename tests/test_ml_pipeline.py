"""
Tests for the Phase 3 ML Pipeline.

Tests:
1. MultiProjectForecaster model architecture
2. Training dataset service
3. Model training service
4. Leakage prevention validators
5. Feature flags
6. Backward compatibility layer
"""
import pytest
import numpy as np
from datetime import date, timedelta
from unittest.mock import MagicMock, patch


class TestMultiProjectForecaster:
    """Tests for MultiProjectForecaster model."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        from app.forecasting.models import get_multi_project_forecaster
        MultiProjectForecaster = get_multi_project_forecaster()

        model = MultiProjectForecaster(
            num_projects=10,
            num_trades=24,
            seq_len=12,
            feature_dim=5,
            project_embed_dim=32,
            trade_embed_dim=16,
            lstm_units=32,  # Smaller for tests
            adapter_units=16,
            dropout=0.1,
        )
        model.build_model()
        return model

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert model.num_projects == 10
        assert model.num_trades == 24
        assert model.seq_len == 12

    def test_model_forward_pass(self, model):
        """Test forward pass produces valid output."""
        import tensorflow as tf

        # Create dummy inputs
        batch_size = 4
        seq_features = tf.random.normal((batch_size, 12, 5))
        project_ids = tf.constant([[1], [2], [3], [4]], dtype=tf.int32)
        trade_ids = tf.constant([[1], [2], [3], [4]], dtype=tf.int32)

        # Forward pass
        mean, std = model((seq_features, project_ids, trade_ids), training=False)

        # Check output shapes
        assert mean.shape == (batch_size, 1)
        assert std.shape == (batch_size, 1)

        # Check std is positive
        assert tf.reduce_all(std > 0).numpy()

    def test_model_predict_with_uncertainty(self, model):
        """Test prediction with uncertainty quantification."""
        seq_features = np.random.randn(12, 5).astype(np.float32)

        result = model.predict_with_uncertainty(
            seq_features,
            project_id=1,
            trade_id=1,
            confidence_level=0.80,
        )

        assert result.point_estimate is not None
        assert result.lower_bound < result.point_estimate < result.upper_bound
        assert result.std > 0
        assert result.confidence_level == 0.80

    def test_model_config(self, model):
        """Test model configuration is preserved."""
        config = model.get_config()

        assert config['num_projects'] == 10
        assert config['num_trades'] == 24
        assert config['seq_len'] == 12
        assert config['feature_dim'] == 5

    def test_model_parameter_count(self, model):
        """Test model has reasonable parameter count."""
        param_count = model.count_params()

        # Model should have parameters
        assert param_count > 0

        # But not too many for a small config
        assert param_count < 1_000_000  # Less than 1M params


class TestTrainingDatasetService:
    """Tests for TrainingDatasetService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_feature_names(self):
        """Test feature names are defined."""
        from app.domain.services.training_dataset_service import TrainingDatasetService

        assert len(TrainingDatasetService.FEATURE_NAMES) == 5
        assert 'cost_per_sf_cents' in TrainingDatasetService.FEATURE_NAMES
        assert 'cumulative_cost_per_sf_cents' in TrainingDatasetService.FEATURE_NAMES

    def test_id_mappings(self, mock_db):
        """Test ID mapping creation."""
        from app.domain.services.training_dataset_service import TrainingDatasetService

        service = TrainingDatasetService(mock_db)

        # Mock project query
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        service._build_id_mappings([1, 2, 3])

        # Check mappings are created
        assert 1 in service._project_id_map
        assert 2 in service._project_id_map
        assert 3 in service._project_id_map

        # Check mappings are 1-indexed (0 reserved for unknown)
        assert all(v >= 1 for v in service._project_id_map.values())


class TestLeakagePrevention:
    """Tests for leakage prevention validators."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_temporal_validator_valid_split(self, mock_db):
        """Test temporal validator accepts valid split."""
        from app.domain.services.leakage_prevention import TemporalLeakageValidator

        validator = TemporalLeakageValidator(mock_db)

        result = validator.validate_train_val_split(
            train_end_date=date(2025, 6, 30),
            val_start_date=date(2025, 7, 1),
            gap_days=0,
        )

        assert result.is_valid

    def test_temporal_validator_invalid_split(self, mock_db):
        """Test temporal validator rejects overlapping split."""
        from app.domain.services.leakage_prevention import TemporalLeakageValidator

        validator = TemporalLeakageValidator(mock_db)

        result = validator.validate_train_val_split(
            train_end_date=date(2025, 7, 15),
            val_start_date=date(2025, 7, 1),  # Before train end
            gap_days=0,
        )

        assert not result.is_valid
        assert result.leakage_type == "train_val_overlap"

    def test_cross_project_validator(self, mock_db):
        """Test cross-project isolation validator."""
        from app.domain.services.leakage_prevention import CrossProjectLeakageValidator

        validator = CrossProjectLeakageValidator(mock_db)

        # Mock query to return features with correct project_id
        mock_feature = MagicMock()
        mock_feature.project_id = 1
        mock_db.query.return_value.filter.return_value.limit.return_value.all.return_value = [mock_feature]

        result = validator.validate_project_isolation([1])

        assert result.is_valid


class TestFeatureFlags:
    """Tests for feature flag system."""

    def setup_method(self):
        """Reset feature flags before each test."""
        from app.infrastructure.feature_flags import FeatureFlag
        FeatureFlag.reset()

    def test_feature_flag_default_disabled(self):
        """Test feature flags are disabled by default."""
        from app.infrastructure.feature_flags import FeatureFlag

        flag = FeatureFlag('test_feature')

        assert not flag.is_enabled()
        assert not flag.is_enabled(entity_id=123)

    def test_feature_flag_enable_globally(self):
        """Test enabling feature flag globally."""
        from app.infrastructure.feature_flags import FeatureFlag

        flag = FeatureFlag('test_feature')
        flag.enable()

        assert flag.is_enabled()
        assert flag.is_enabled(entity_id=123)

    def test_feature_flag_enable_per_entity(self):
        """Test enabling feature flag for specific entity."""
        from app.infrastructure.feature_flags import FeatureFlag

        flag = FeatureFlag('test_feature')
        flag.enable(entity_id=123)

        assert flag.is_enabled(entity_id=123)
        assert not flag.is_enabled(entity_id=456)

    def test_feature_flag_percentage_rollout(self):
        """Test percentage-based rollout."""
        from app.infrastructure.feature_flags import FeatureFlag

        flag = FeatureFlag('test_feature')
        flag.set_percentage(50)

        # With 50% rollout, roughly half of entities should be enabled
        # Using a large sample for statistical validity
        enabled_count = sum(
            1 for i in range(1000)
            if flag.is_enabled(entity_id=i)
        )

        # Should be roughly 50% (with some tolerance)
        assert 400 < enabled_count < 600

    def test_feature_flag_deterministic(self):
        """Test percentage rollout is deterministic for same entity."""
        from app.infrastructure.feature_flags import FeatureFlag

        flag = FeatureFlag('test_feature')
        flag.set_percentage(50)

        # Same entity should always get same result
        result1 = flag.is_enabled(entity_id=123)
        result2 = flag.is_enabled(entity_id=123)

        assert result1 == result2

    def test_feature_flags_registry(self):
        """Test FeatureFlags registry."""
        from app.infrastructure.feature_flags import FeatureFlags

        # Check predefined flags exist
        assert FeatureFlags.MULTI_PROJECT_FORECASTING is not None
        assert FeatureFlags.CANONICAL_TRADE_MAPPING is not None

    def test_enable_for_project(self):
        """Test enabling all features for a project."""
        from app.infrastructure.feature_flags import FeatureFlags, FeatureFlag

        # Reset first
        FeatureFlag.reset()

        # Re-initialize
        from app.infrastructure import feature_flags
        import importlib
        importlib.reload(feature_flags)

        FeatureFlags.enable_for_project(123)

        assert FeatureFlags.MULTI_PROJECT_FORECASTING.is_enabled(123)
        assert FeatureFlags.CANONICAL_TRADE_MAPPING.is_enabled(123)


class TestCompatibilityLayer:
    """Tests for backward compatibility layer."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_get_gmp_with_canonical_info(self, mock_db):
        """Test GMP response includes both legacy and new fields."""
        from app.domain.services.compatibility_layer import CompatibilityLayer

        # Mock GMP
        mock_gmp = MagicMock()
        mock_gmp.id = 1
        mock_gmp.division = "03 - Concrete"
        mock_gmp.zone = "EAST"
        mock_gmp.original_amount_cents = 50000000
        mock_gmp.current_amount_cents = 50000000
        mock_gmp.committed_cost_cents = 45000000
        mock_gmp.project_id = 1
        mock_gmp.canonical_trade_id = 3
        mock_gmp.normalized_amount_per_sf_cents = 30000

        mock_db.query.return_value.get.return_value = mock_gmp

        # Mock canonical trade
        mock_trade = MagicMock()
        mock_trade.canonical_code = "03-CONCRETE"
        mock_trade.canonical_name = "Concrete"
        mock_db.query.return_value.get.side_effect = [mock_gmp, mock_trade]

        layer = CompatibilityLayer(mock_db)
        result = layer.get_gmp_with_canonical_info(1)

        # Check legacy fields
        assert result['id'] == 1
        assert result['division'] == "03 - Concrete"
        assert result['zone'] == "EAST"

        # Check new fields
        assert result['canonical_trade_id'] == 3

    def test_confidence_band_conversion(self, mock_db):
        """Test confidence score to band conversion."""
        from app.domain.services.compatibility_layer import CompatibilityLayer

        layer = CompatibilityLayer(mock_db)

        assert layer._score_to_band(0.02) == 'HIGH'
        assert layer._score_to_band(0.10) == 'MEDIUM'
        assert layer._score_to_band(0.25) == 'LOW'


class TestProjectCutoverService:
    """Tests for project cutover service."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_validate_cutover_project_not_found(self, mock_db):
        """Test validation fails for non-existent project."""
        from app.domain.services.compatibility_layer import ProjectCutoverService

        mock_db.query.return_value.get.return_value = None

        service = ProjectCutoverService(mock_db)
        is_ready, issues = service.validate_cutover_readiness(999)

        assert not is_ready
        assert "Project not found" in issues

    def test_validate_cutover_low_quality(self, mock_db):
        """Test validation fails for low quality project."""
        from app.domain.services.compatibility_layer import ProjectCutoverService

        mock_project = MagicMock()
        mock_project.data_quality_score = 0.5  # Below 0.8 threshold
        mock_db.query.return_value.get.return_value = mock_project
        mock_db.query.return_value.filter.return_value.count.return_value = 0

        service = ProjectCutoverService(mock_db)
        is_ready, issues = service.validate_cutover_readiness(1)

        assert not is_ready
        assert any("quality score" in issue.lower() for issue in issues)


class TestIntegration:
    """Integration tests for full ML pipeline."""

    @pytest.fixture
    def model(self):
        """Create a model for integration testing."""
        from app.forecasting.models import get_multi_project_forecaster
        MultiProjectForecaster = get_multi_project_forecaster()

        model = MultiProjectForecaster(
            num_projects=5,
            num_trades=10,
            seq_len=6,  # Shorter for tests
            feature_dim=5,
            project_embed_dim=16,
            trade_embed_dim=8,
            lstm_units=16,
            adapter_units=8,
        )
        model.build_model()
        return model

    def test_end_to_end_prediction(self, model):
        """Test end-to-end prediction pipeline."""
        # Create synthetic input data
        seq_features = np.random.randn(6, 5).astype(np.float32)

        # Get prediction
        result = model.predict_with_uncertainty(
            seq_features,
            project_id=1,
            trade_id=1,
        )

        # Validate output
        assert result.point_estimate is not None
        assert result.std > 0
        assert result.lower_bound < result.upper_bound

    def test_batch_prediction(self, model):
        """Test batch prediction."""
        import tensorflow as tf

        batch_size = 8
        seq_features = tf.random.normal((batch_size, 6, 5))
        project_ids = tf.constant([[i % 5] for i in range(batch_size)], dtype=tf.int32)
        trade_ids = tf.constant([[i % 10] for i in range(batch_size)], dtype=tf.int32)

        mean, std = model((seq_features, project_ids, trade_ids), training=False)

        assert mean.shape == (batch_size, 1)
        assert std.shape == (batch_size, 1)

    def test_model_save_load(self, model, tmp_path):
        """Test model save and load."""
        from app.forecasting.models import get_multi_project_forecaster

        # Save model
        save_path = tmp_path / "test_model"
        model.save(str(save_path))

        # Load model
        MultiProjectForecaster = get_multi_project_forecaster()
        loaded_model = MultiProjectForecaster(
            num_projects=5,
            num_trades=10,
            seq_len=6,
            feature_dim=5,
            project_embed_dim=16,
            trade_embed_dim=8,
            lstm_units=16,
            adapter_units=8,
        )
        loaded_model.load(str(save_path))

        # Verify loaded model works
        seq_features = np.random.randn(6, 5).astype(np.float32)
        result = loaded_model.predict_with_uncertainty(seq_features, 1, 1)

        assert result.point_estimate is not None

    def test_deterministic_predictions(self, model):
        """Test predictions are deterministic given same inputs."""
        seq_features = np.random.randn(6, 5).astype(np.float32)

        result1 = model.predict_with_uncertainty(seq_features, 1, 1)
        result2 = model.predict_with_uncertainty(seq_features, 1, 1)

        assert np.isclose(result1.mean, result2.mean, rtol=1e-5)
        assert np.isclose(result1.std, result2.std, rtol=1e-5)
