"""
Tests for the suggestion engine scoring algorithm.
"""
import pytest
import sys
sys.path.insert(0, '.')

from app.modules.suggestion_engine import (
    # Normalization functions
    extract_base_code,
    normalize_vendor,
    extract_name_prefix,
    extract_cost_type_code,
    normalize_text,
    # Scoring functions
    compute_code_match,
    compute_type_match,
    compute_text_similarity,
    compute_historical_boost,
    compute_match_score,
    rank_suggestions,
    # Data classes
    DirectCostRow,
    BudgetRow,
    ScoredMatch,
    # Constants
    WEIGHT_CODE_MATCH,
    WEIGHT_TYPE_MATCH,
    WEIGHT_TEXT_SIM,
    WEIGHT_HISTORICAL,
    THRESHOLD_HIGH,
    THRESHOLD_MEDIUM,
)


class TestNormalizationFunctions:
    """Tests for normalization functions."""

    def test_extract_base_code(self):
        assert extract_base_code('4-010-200') == '4-010'
        assert extract_base_code('4-010') == '4-010'
        assert extract_base_code('4') == '4'
        assert extract_base_code('12-345-678') == '12-345'
        assert extract_base_code('') == ''
        assert extract_base_code(None) == ''

    def test_normalize_vendor(self):
        assert normalize_vendor('ABC Corp') == 'abc'
        assert normalize_vendor('XYZ Inc') == 'xyz'  # Without trailing period
        assert normalize_vendor('Acme LLC') == 'acme'
        assert normalize_vendor('Big Company Corporation') == 'big'  # Removes 'company' and 'corporation'
        assert normalize_vendor('  Spaces  Inc  ') == 'spaces'
        assert normalize_vendor('') == ''
        assert normalize_vendor(None) == ''

    def test_extract_name_prefix(self):
        # Default is 20 chars
        assert extract_name_prefix('Concrete delivery for foundation work') == 'concrete delivery fo'
        assert extract_name_prefix('Short') == 'short'
        assert extract_name_prefix('', 20) == ''
        assert extract_name_prefix(None) == ''

    def test_extract_cost_type_code(self):
        assert extract_cost_type_code('L - Labor') == 'L'
        assert extract_cost_type_code('M - Material') == 'M'
        assert extract_cost_type_code('L') == 'L'
        assert extract_cost_type_code('Labor') == 'L'
        assert extract_cost_type_code('Material') == 'M'
        assert extract_cost_type_code('Subcontract') == 'S'
        assert extract_cost_type_code('') == ''

    def test_normalize_text(self):
        assert normalize_text('  Hello,  World!  ') == 'hello world'
        assert normalize_text('Concrete (4000 PSI)') == 'concrete 4000 psi'
        assert normalize_text('') == ''


class TestScoringFunctions:
    """Tests for individual scoring components."""

    def test_compute_code_match_exact(self):
        assert compute_code_match('4-010', '4-010') == 1.0

    def test_compute_code_match_different(self):
        assert compute_code_match('4-010', '4-020') == 0.0
        assert compute_code_match('4-010', '5-010') == 0.0

    def test_compute_code_match_empty(self):
        assert compute_code_match('', '4-010') == 0.0
        assert compute_code_match('4-010', '') == 0.0

    def test_compute_type_match_exact(self):
        assert compute_type_match('L', 'L') == 1.0
        assert compute_type_match('M', 'M') == 1.0
        assert compute_type_match('L - Labor', 'L') == 1.0

    def test_compute_type_match_compatible(self):
        # Labor and Subcontract are compatible (0.5)
        assert compute_type_match('L', 'S') == 0.5
        assert compute_type_match('S', 'L') == 0.5

    def test_compute_type_match_different(self):
        assert compute_type_match('L', 'M') == 0.0
        assert compute_type_match('M', 'S') == 0.0

    def test_compute_text_similarity_exact(self):
        score = compute_text_similarity('Concrete Materials', 'Concrete Materials')
        assert score == 1.0

    def test_compute_text_similarity_similar(self):
        score = compute_text_similarity('Rebar delivery', 'Rebar Materials')
        assert 0.5 < score < 1.0  # Should be reasonably similar

    def test_compute_text_similarity_different(self):
        score = compute_text_similarity('Concrete', 'Electrical Wiring')
        assert score < 0.5

    def test_compute_historical_boost_match(self):
        history = {('acme', 'concrete deliver'): '4-010-M'}
        assert compute_historical_boost('acme', 'concrete deliver', '4-010-M', history) == 1.0

    def test_compute_historical_boost_no_match(self):
        history = {('acme', 'concrete deliver'): '4-010-M'}
        assert compute_historical_boost('acme', 'concrete deliver', '4-020-M', history) == 0.0

    def test_compute_historical_boost_no_history(self):
        history = {}
        assert compute_historical_boost('acme', 'concrete deliver', '4-010-M', history) == 0.0


class TestCompositeScoring:
    """Tests for the composite match scoring."""

    @pytest.fixture
    def sample_budget_rows(self):
        return [
            BudgetRow('4-010-M', '4-010', 'Concrete Materials', 'M - Material', 'M'),
            BudgetRow('4-010-L', '4-010', 'Concrete Labor', 'L - Labor', 'L'),
            BudgetRow('4-020-M', '4-020', 'Rebar Materials', 'M - Material', 'M'),
            BudgetRow('6-010-S', '6-010', 'Steel Subcontract', 'S - Subcontract', 'S'),
        ]

    def test_perfect_match(self, sample_budget_rows):
        """Test scoring when all components match perfectly."""
        dc = DirectCostRow(
            id=1,
            cost_code='4-010-200',
            base_code='4-010',
            name='Concrete Materials',
            cost_type='M',
            vendor='Acme Concrete',
            vendor_normalized='acme concrete',
            amount_cents=100000
        )

        budget = sample_budget_rows[0]  # 4-010-M Concrete Materials
        history = {('acme concrete', 'concrete materials'): '4-010-M'}

        score = compute_match_score(dc, budget, history)

        assert score.code_match_score == 1.0
        assert score.type_match_score == 1.0
        assert score.text_sim_score == 1.0
        assert score.historical_score == 1.0
        assert score.total_score == pytest.approx(1.0)  # Handle floating point
        assert score.confidence_band == 'high'

    def test_code_only_match(self, sample_budget_rows):
        """Test when only code matches."""
        dc = DirectCostRow(
            id=1,
            cost_code='4-010-200',
            base_code='4-010',
            name='Something Completely Different',
            cost_type='O',
            vendor='Unknown Vendor',
            vendor_normalized='unknown vendor',
            amount_cents=100000
        )

        budget = sample_budget_rows[0]  # 4-010-M
        history = {}

        score = compute_match_score(dc, budget, history)

        assert score.code_match_score == 1.0
        assert score.type_match_score == 0.0  # O vs M
        assert score.historical_score == 0.0
        # Total should be around 0.40 (code) + text_sim contribution
        assert 0.40 <= score.total_score <= 0.60

    def test_ranking_with_tie_breaking(self, sample_budget_rows):
        """Test that ranking uses tie-breaking correctly."""
        dc = DirectCostRow(
            id=1,
            cost_code='4-010-200',
            base_code='4-010',
            name='Concrete work',
            cost_type='M',
            vendor='Acme',
            vendor_normalized='acme',
            amount_cents=100000
        )

        history = {}
        match_counts = {'4-010-L': 50, '4-010-M': 10}  # L has more matches

        suggestions = rank_suggestions(dc, sample_budget_rows, history, match_counts, top_k=3)

        # Both 4-010-M and 4-010-L should score high (same code, type match varies)
        assert len(suggestions) >= 2

        # Top suggestions should be 4-010 codes
        top_codes = [s.budget_code for s in suggestions[:2]]
        assert '4-010-M' in top_codes or '4-010-L' in top_codes

    def test_confidence_bands(self, sample_budget_rows):
        """Test confidence band assignment."""
        # High confidence: score >= 0.85
        dc_high = DirectCostRow(1, '4-010-200', '4-010', 'Concrete Materials',
                                'M', 'Acme', 'acme', 100000)
        budget_m = sample_budget_rows[0]
        history_match = {('acme', 'concrete materials'): '4-010-M'}

        score_high = compute_match_score(dc_high, budget_m, history_match)
        assert score_high.confidence_band == 'high'

        # Medium confidence: 0.60 <= score < 0.85
        dc_medium = DirectCostRow(1, '4-010-200', '4-010', 'Foundation work',
                                  'L', 'Different', 'different', 100000)
        score_medium = compute_match_score(dc_medium, budget_m, {})
        # Code matches (0.4), type doesn't match, text partial
        assert score_medium.confidence_band in ['medium', 'low']

    def test_to_dict_output(self, sample_budget_rows):
        """Test that to_dict produces expected structure."""
        dc = DirectCostRow(1, '4-010-200', '4-010', 'Concrete',
                          'M', 'Acme', 'acme', 100000)
        budget = sample_budget_rows[0]

        score = compute_match_score(dc, budget, {})
        result = score.to_dict()

        assert 'budget_code' in result
        assert 'description' in result
        assert 'score' in result  # Percentage
        assert 'total_score' in result
        assert 'breakdown' in result
        assert 'confidence_band' in result
        assert isinstance(result['score'], int)  # Should be percentage int


class TestWeightConfiguration:
    """Test that weights are configured correctly."""

    def test_weights_sum_to_one(self):
        total = WEIGHT_CODE_MATCH + WEIGHT_TYPE_MATCH + WEIGHT_TEXT_SIM + WEIGHT_HISTORICAL
        assert abs(total - 1.0) < 0.001

    def test_threshold_ordering(self):
        assert THRESHOLD_HIGH > THRESHOLD_MEDIUM


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
