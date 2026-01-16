"""
Trade Mapping Service - Maps raw GMP divisions to canonical trades.

Implements fuzzy matching and keyword-based mapping from project-specific
division names to the CSI-based canonical trade taxonomy.
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from difflib import SequenceMatcher
from sqlalchemy.orm import Session

from app.models import (
    CanonicalTrade,
    ProjectTradeMapping,
    GMP,
    Project,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeMappingResult:
    """Result of a trade mapping operation."""
    raw_division_name: str
    canonical_trade_id: int
    canonical_code: str
    canonical_name: str
    confidence: float
    mapping_method: str  # 'exact', 'keyword', 'fuzzy', 'csi_prefix'


@dataclass
class TradeMappingSuggestion:
    """Suggested mapping with confidence score."""
    canonical_trade: CanonicalTrade
    confidence: float
    method: str
    match_reason: str


class TradeMappingService:
    """
    Service for mapping raw GMP division names to canonical trades.

    Mapping Strategy (in priority order):
    1. Exact match on canonical code (e.g., "03-CONCRETE")
    2. CSI prefix extraction (e.g., "03 - Concrete Work" -> Division 03)
    3. Keyword matching (e.g., "concrete", "electrical", "plumbing")
    4. Fuzzy string matching on canonical names
    """

    # Keyword mappings to CSI divisions
    KEYWORD_MAPPINGS = {
        # Division 01 - General Requirements
        '01': ['general', 'requirements', 'conditions', 'admin', 'overhead', 'fee', 'insurance', 'bond'],
        # Division 02 - Existing Conditions
        '02': ['demolition', 'demo', 'existing', 'abatement', 'hazmat', 'asbestos'],
        # Division 03 - Concrete
        '03': ['concrete', 'cement', 'rebar', 'formwork', 'foundation', 'slab', 'footing'],
        # Division 04 - Masonry
        '04': ['masonry', 'brick', 'block', 'stone', 'mortar', 'cmu'],
        # Division 05 - Metals
        '05': ['steel', 'metal', 'structural', 'iron', 'railing', 'stair'],
        # Division 06 - Wood, Plastics, Composites
        '06': ['wood', 'carpentry', 'framing', 'millwork', 'casework', 'cabinet', 'plastic'],
        # Division 07 - Thermal & Moisture Protection
        '07': ['roofing', 'waterproof', 'insulation', 'fireproof', 'sealant', 'caulk', 'membrane'],
        # Division 08 - Openings
        '08': ['door', 'window', 'glass', 'glazing', 'hardware', 'curtainwall', 'storefront'],
        # Division 09 - Finishes
        '09': ['drywall', 'gyp', 'paint', 'flooring', 'tile', 'ceiling', 'carpet', 'finish', 'plaster'],
        # Division 10 - Specialties
        '10': ['toilet', 'partition', 'locker', 'signage', 'accessory', 'specialt'],
        # Division 11 - Equipment
        '11': ['equipment', 'appliance', 'kitchen', 'laundry', 'food service'],
        # Division 12 - Furnishings
        '12': ['furniture', 'furnishing', 'seating', 'case goods', 'artwork'],
        # Division 13 - Special Construction
        '13': ['special', 'pool', 'fountain', 'clean room', 'vault'],
        # Division 14 - Conveying Equipment
        '14': ['elevator', 'escalator', 'lift', 'conveying', 'dumbwaiter'],
        # Division 21 - Fire Suppression
        '21': ['fire suppression', 'sprinkler', 'fire protect', 'standpipe'],
        # Division 22 - Plumbing
        '22': ['plumbing', 'piping', 'fixture', 'sanitary', 'water heater', 'domestic water'],
        # Division 23 - HVAC
        '23': ['hvac', 'mechanical', 'air condition', 'heating', 'ventilation', 'duct', 'chiller', 'boiler'],
        # Division 26 - Electrical
        '26': ['electrical', 'power', 'lighting', 'panel', 'switchgear', 'generator', 'wire', 'conduit'],
        # Division 27 - Communications
        '27': ['communication', 'data', 'telecom', 'av', 'audio', 'video', 'network', 'low voltage'],
        # Division 28 - Electronic Safety & Security
        '28': ['security', 'alarm', 'access control', 'cctv', 'camera', 'fire alarm'],
        # Division 31 - Earthwork
        '31': ['earthwork', 'excavat', 'grading', 'soil', 'backfill', 'compaction'],
        # Division 32 - Exterior Improvements
        '32': ['paving', 'landscape', 'sidewalk', 'curb', 'parking', 'asphalt', 'fence'],
        # Division 33 - Utilities
        '33': ['utilit', 'sewer', 'storm', 'gas', 'underground'],
    }

    def __init__(self, db: Session):
        self.db = db
        self._canonical_trades: Optional[Dict[int, CanonicalTrade]] = None
        self._trades_by_code: Optional[Dict[str, CanonicalTrade]] = None
        self._trades_by_division: Optional[Dict[str, CanonicalTrade]] = None

    def _load_canonical_trades(self) -> None:
        """Load and cache canonical trades."""
        if self._canonical_trades is not None:
            return

        trades = self.db.query(CanonicalTrade).filter(
            CanonicalTrade.is_active == True
        ).all()

        self._canonical_trades = {t.id: t for t in trades}
        self._trades_by_code = {t.canonical_code.upper(): t for t in trades}
        self._trades_by_division = {t.csi_division: t for t in trades}

    def get_all_canonical_trades(self) -> List[CanonicalTrade]:
        """Get all active canonical trades."""
        self._load_canonical_trades()
        return list(self._canonical_trades.values())

    def suggest_mapping(
        self,
        raw_division_name: str,
        top_n: int = 3
    ) -> List[TradeMappingSuggestion]:
        """
        Suggest canonical trade mappings for a raw division name.

        Returns top N suggestions ranked by confidence.
        """
        self._load_canonical_trades()
        suggestions: List[TradeMappingSuggestion] = []
        normalized = raw_division_name.lower().strip()

        # 1. Try exact match on canonical code
        upper_name = raw_division_name.upper().strip()
        if upper_name in self._trades_by_code:
            trade = self._trades_by_code[upper_name]
            suggestions.append(TradeMappingSuggestion(
                canonical_trade=trade,
                confidence=1.0,
                method='exact',
                match_reason=f"Exact match on code '{upper_name}'"
            ))
            return suggestions[:top_n]

        # 2. Try CSI prefix extraction (e.g., "03 - Concrete" -> "03")
        csi_match = re.match(r'^(\d{2})\s*[-–—]?\s*', raw_division_name)
        if csi_match:
            csi_division = csi_match.group(1)
            if csi_division in self._trades_by_division:
                trade = self._trades_by_division[csi_division]
                suggestions.append(TradeMappingSuggestion(
                    canonical_trade=trade,
                    confidence=0.95,
                    method='csi_prefix',
                    match_reason=f"CSI division prefix '{csi_division}'"
                ))

        # 3. Keyword matching
        for csi_div, keywords in self.KEYWORD_MAPPINGS.items():
            for keyword in keywords:
                if keyword in normalized:
                    if csi_div in self._trades_by_division:
                        trade = self._trades_by_division[csi_div]
                        # Don't add duplicates
                        if not any(s.canonical_trade.id == trade.id for s in suggestions):
                            suggestions.append(TradeMappingSuggestion(
                                canonical_trade=trade,
                                confidence=0.85,
                                method='keyword',
                                match_reason=f"Keyword '{keyword}' matches division {csi_div}"
                            ))
                        break

        # 4. Fuzzy matching on canonical names
        for trade in self._canonical_trades.values():
            # Skip if already in suggestions
            if any(s.canonical_trade.id == trade.id for s in suggestions):
                continue

            # Calculate similarity
            ratio = SequenceMatcher(
                None,
                normalized,
                trade.canonical_name.lower()
            ).ratio()

            if ratio > 0.5:
                suggestions.append(TradeMappingSuggestion(
                    canonical_trade=trade,
                    confidence=ratio * 0.8,  # Scale down fuzzy matches
                    method='fuzzy',
                    match_reason=f"Fuzzy match ({ratio:.0%} similarity)"
                ))

        # Sort by confidence descending
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:top_n]

    def map_division(
        self,
        project_id: int,
        raw_division_name: str,
        canonical_trade_id: Optional[int] = None,
        created_by: str = 'system'
    ) -> TradeMappingResult:
        """
        Map a raw division name to a canonical trade.

        If canonical_trade_id is provided, use it directly (manual mapping).
        Otherwise, use the best suggestion.
        """
        self._load_canonical_trades()

        if canonical_trade_id:
            # Manual mapping
            trade = self._canonical_trades.get(canonical_trade_id)
            if not trade:
                raise ValueError(f"Canonical trade {canonical_trade_id} not found")

            confidence = 1.0
            method = 'manual'
        else:
            # Auto-suggest
            suggestions = self.suggest_mapping(raw_division_name, top_n=1)
            if not suggestions:
                raise ValueError(f"No mapping found for '{raw_division_name}'")

            best = suggestions[0]
            trade = best.canonical_trade
            confidence = best.confidence
            method = best.method

        # Check for existing mapping
        existing = self.db.query(ProjectTradeMapping).filter(
            ProjectTradeMapping.project_id == project_id,
            ProjectTradeMapping.raw_division_name == raw_division_name
        ).first()

        if existing:
            # Update existing mapping
            existing.canonical_trade_id = trade.id
            existing.confidence = confidence
            existing.mapping_method = method
        else:
            # Create new mapping
            mapping = ProjectTradeMapping(
                project_id=project_id,
                raw_division_name=raw_division_name,
                canonical_trade_id=trade.id,
                confidence=confidence,
                mapping_method=method,
                created_by=created_by
            )
            self.db.add(mapping)

        return TradeMappingResult(
            raw_division_name=raw_division_name,
            canonical_trade_id=trade.id,
            canonical_code=trade.canonical_code,
            canonical_name=trade.canonical_name,
            confidence=confidence,
            mapping_method=method
        )

    def map_all_project_divisions(
        self,
        project_id: int,
        auto_confirm_threshold: float = 0.9,
        created_by: str = 'system'
    ) -> List[TradeMappingResult]:
        """
        Map all GMP divisions for a project to canonical trades.

        Args:
            project_id: Project to map
            auto_confirm_threshold: Auto-confirm mappings above this confidence
            created_by: User/system identifier

        Returns:
            List of mapping results
        """
        # Get all GMPs for project
        gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id
        ).all()

        results = []
        for gmp in gmps:
            try:
                result = self.map_division(
                    project_id=project_id,
                    raw_division_name=gmp.division,
                    created_by=created_by
                )

                # Update GMP with canonical trade
                if result.confidence >= auto_confirm_threshold:
                    gmp.canonical_trade_id = result.canonical_trade_id

                results.append(result)

            except ValueError as e:
                logger.warning(f"Failed to map division '{gmp.division}': {e}")
                continue

        return results

    def get_unmapped_divisions(self, project_id: int) -> List[str]:
        """Get list of GMP divisions without canonical trade mappings."""
        gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.canonical_trade_id == None
        ).all()

        return [gmp.division for gmp in gmps]

    def get_project_mappings(self, project_id: int) -> List[ProjectTradeMapping]:
        """Get all trade mappings for a project."""
        return self.db.query(ProjectTradeMapping).filter(
            ProjectTradeMapping.project_id == project_id
        ).all()

    def get_low_confidence_mappings(
        self,
        project_id: int,
        threshold: float = 0.8
    ) -> List[ProjectTradeMapping]:
        """Get mappings below confidence threshold for review."""
        return self.db.query(ProjectTradeMapping).filter(
            ProjectTradeMapping.project_id == project_id,
            ProjectTradeMapping.confidence < threshold
        ).all()
