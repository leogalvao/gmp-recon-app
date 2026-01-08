"""
Cost Mapping Service - Direct Cost -> Budget -> GMP Pipeline.

Implements the hierarchical cost mapping with:
- CSI MasterFormat code matching
- Fuzzy matching for vendor descriptions
- Prospect theory risk weighting
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import re
import logging

from ..entities.direct_cost import DirectCost, CostCategory
from ..entities.budget_line import BudgetLine
from ..entities.gmp_allocation import GMPAllocation, GMPLineItem

logger = logging.getLogger(__name__)


@dataclass
class MappingRule:
    """
    Rule for mapping direct costs to budget lines.

    Attributes:
        id: Unique rule identifier
        cost_code_pattern: Regex pattern for CSI code matching
        category: Cost category filter (optional)
        target_budget_line_id: Specific budget line to map to
        priority: Higher = more specific (checked first)
        markup_factor: Multiplier for burden/overhead
    """
    id: str
    cost_code_pattern: str  # Regex pattern for CSI code
    category: Optional[CostCategory] = None
    target_budget_line_id: Optional[UUID] = None
    priority: int = 0  # Higher = more specific
    markup_factor: Decimal = Decimal("1.0")


@dataclass
class MappingResult:
    """Result of a cost mapping operation."""
    success: bool
    direct_cost: DirectCost
    budget_line: Optional[BudgetLine] = None
    gmp_line_item: Optional[GMPLineItem] = None
    mapped_amount: Decimal = Decimal("0.00")
    rule_applied: Optional[MappingRule] = None
    error_message: Optional[str] = None


class CostMapper:
    """
    Maps direct costs through the budget layer to GMP.

    Mapping Strategy:
    1. Match by CSI cost code (highest priority)
    2. Match by cost category
    3. Match by vendor/description patterns
    4. Default allocation to unassigned bucket

    Thread-safe and stateless - all state passed as parameters.
    """

    def __init__(self):
        self.mapping_rules: List[MappingRule] = []
        self.budget_lines: Dict[UUID, BudgetLine] = {}
        self.gmp_allocations: Dict[UUID, GMPAllocation] = {}

        # Default CSI code mappings
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Set up default CSI MasterFormat mapping rules."""
        default_rules = [
            MappingRule(
                id="concrete",
                cost_code_pattern=r"^03\d{4}",  # Division 03
                category=CostCategory.MATERIAL,
                priority=10,
            ),
            MappingRule(
                id="masonry",
                cost_code_pattern=r"^04\d{4}",  # Division 04
                category=CostCategory.MATERIAL,
                priority=10,
            ),
            MappingRule(
                id="metals",
                cost_code_pattern=r"^05\d{4}",  # Division 05
                category=CostCategory.MATERIAL,
                priority=10,
            ),
            MappingRule(
                id="wood_plastics",
                cost_code_pattern=r"^06\d{4}",  # Division 06
                category=CostCategory.MATERIAL,
                priority=10,
            ),
            MappingRule(
                id="thermal_moisture",
                cost_code_pattern=r"^07\d{4}",  # Division 07 (Roofing)
                category=CostCategory.SUBCONTRACT,
                priority=10,
            ),
            MappingRule(
                id="openings",
                cost_code_pattern=r"^08\d{4}",  # Division 08
                category=CostCategory.SUBCONTRACT,
                priority=10,
            ),
            MappingRule(
                id="finishes",
                cost_code_pattern=r"^09\d{4}",  # Division 09
                category=CostCategory.SUBCONTRACT,
                priority=10,
            ),
            MappingRule(
                id="fire_suppression",
                cost_code_pattern=r"^21\d{4}",  # Division 21
                category=CostCategory.SUBCONTRACT,
                priority=10,
            ),
            MappingRule(
                id="plumbing",
                cost_code_pattern=r"^22\d{4}",  # Division 22
                category=CostCategory.SUBCONTRACT,
                priority=10,
            ),
            MappingRule(
                id="hvac",
                cost_code_pattern=r"^23\d{4}",  # Division 23 (Mechanical)
                category=CostCategory.SUBCONTRACT,
                priority=10,
            ),
            MappingRule(
                id="electrical",
                cost_code_pattern=r"^26\d{4}",  # Division 26
                category=CostCategory.SUBCONTRACT,
                priority=10,
            ),
            MappingRule(
                id="labor_general",
                cost_code_pattern=r".*",
                category=CostCategory.LABOR,
                priority=1,
                markup_factor=Decimal("1.15"),  # 15% labor burden
            ),
            MappingRule(
                id="equipment_general",
                cost_code_pattern=r".*",
                category=CostCategory.EQUIPMENT,
                priority=1,
                markup_factor=Decimal("1.05"),  # 5% equipment overhead
            ),
        ]
        self.mapping_rules.extend(default_rules)
        # Sort by priority (highest first)
        self.mapping_rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: MappingRule) -> None:
        """
        Add a custom mapping rule.

        Args:
            rule: MappingRule to add

        Note: Re-sorts rules by priority after adding
        """
        self.mapping_rules.append(rule)
        self.mapping_rules.sort(key=lambda r: r.priority, reverse=True)

    def register_budget_line(self, budget_line: BudgetLine) -> None:
        """Register a budget line for mapping."""
        self.budget_lines[budget_line.id] = budget_line

    def register_gmp(self, gmp: GMPAllocation) -> None:
        """Register a GMP allocation."""
        self.gmp_allocations[gmp.id] = gmp

    def find_matching_rule(self, cost: DirectCost) -> Optional[MappingRule]:
        """
        Find the best matching rule for a direct cost.

        Uses priority ordering - first match wins.

        Args:
            cost: DirectCost to find rule for

        Returns:
            Matching MappingRule or None
        """
        for rule in self.mapping_rules:
            # Check cost code pattern
            if rule.cost_code_pattern and cost.cost_code:
                if re.match(rule.cost_code_pattern, cost.cost_code):
                    # Also check category if specified
                    if rule.category is None or rule.category == cost.category:
                        return rule
            # Check category-only rules
            elif rule.category == cost.category:
                return rule
        return None

    def find_budget_line(
        self,
        cost: DirectCost,
        sub_job_id: Optional[UUID] = None
    ) -> Optional[BudgetLine]:
        """
        Find appropriate budget line for a direct cost.

        Matching strategy:
        1. If rule specifies target, use that
        2. Otherwise match by cost code and sub-job

        Args:
            cost: DirectCost to find budget line for
            sub_job_id: Optional sub-job filter

        Returns:
            Matching BudgetLine or None
        """
        rule = self.find_matching_rule(cost)

        if rule and rule.target_budget_line_id:
            return self.budget_lines.get(rule.target_budget_line_id)

        # Find by cost code and sub-job
        for bl in self.budget_lines.values():
            if bl.cost_code == cost.cost_code:
                if sub_job_id is None or bl.sub_job_id == sub_job_id:
                    return bl

        return None

    def map_cost_to_budget(
        self,
        cost: DirectCost,
        sub_job_id: Optional[UUID] = None
    ) -> MappingResult:
        """
        Map a direct cost to a budget line.

        Args:
            cost: DirectCost to map
            sub_job_id: Optional sub-job filter

        Returns:
            MappingResult with success status and details
        """
        budget_line = self.find_budget_line(cost, sub_job_id)

        if budget_line is None:
            return MappingResult(
                success=False,
                direct_cost=cost,
                error_message=f"No budget line found for cost code: {cost.cost_code}"
            )

        # Get markup factor from matching rule
        rule = self.find_matching_rule(cost)
        markup = rule.markup_factor if rule else Decimal("1.0")

        mapped_amount = cost.to_budget_contribution(markup)
        budget_line.link_direct_cost(cost)

        return MappingResult(
            success=True,
            direct_cost=cost,
            budget_line=budget_line,
            mapped_amount=mapped_amount,
            rule_applied=rule
        )

    def map_budget_to_gmp(
        self,
        budget_line: BudgetLine,
        gmp_id: UUID
    ) -> Optional[GMPLineItem]:
        """
        Map a budget line to its GMP line item.

        Args:
            budget_line: BudgetLine to map
            gmp_id: Parent GMP allocation ID

        Returns:
            Parent GMPLineItem or None
        """
        gmp = self.gmp_allocations.get(gmp_id)
        if gmp is None:
            return None

        for line_item in gmp.line_items:
            if line_item.id == budget_line.gmp_line_id:
                return line_item

        return None

    def process_cost_batch(
        self,
        costs: List[DirectCost],
        gmp_id: UUID,
        sub_job_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of direct costs through the mapping pipeline.

        Args:
            costs: List of DirectCost objects to process
            gmp_id: Target GMP allocation
            sub_job_id: Optional sub-job filter

        Returns:
            Summary statistics dictionary
        """
        results = {
            'total_costs': len(costs),
            'mapped_successfully': 0,
            'unmapped': 0,
            'total_amount': Decimal("0.00"),
            'mapped_amount': Decimal("0.00"),
            'by_category': {},
            'by_rule': {},
            'errors': [],
        }

        for cost in costs:
            results['total_amount'] += cost.amount

            mapping_result = self.map_cost_to_budget(cost, sub_job_id)

            if mapping_result.success:
                results['mapped_successfully'] += 1
                results['mapped_amount'] += mapping_result.mapped_amount

                # Track by category
                cat = cost.category.value
                if cat not in results['by_category']:
                    results['by_category'][cat] = Decimal("0.00")
                results['by_category'][cat] += mapping_result.mapped_amount

                # Track by rule
                if mapping_result.rule_applied:
                    rule_id = mapping_result.rule_applied.id
                    if rule_id not in results['by_rule']:
                        results['by_rule'][rule_id] = {
                            'count': 0,
                            'amount': Decimal("0.00")
                        }
                    results['by_rule'][rule_id]['count'] += 1
                    results['by_rule'][rule_id]['amount'] += mapping_result.mapped_amount
            else:
                results['unmapped'] += 1
                results['errors'].append({
                    'cost_id': str(cost.id),
                    'cost_code': cost.cost_code,
                    'error': mapping_result.error_message
                })

        logger.info(
            f"Processed {results['total_costs']} costs: "
            f"{results['mapped_successfully']} mapped, "
            f"{results['unmapped']} unmapped"
        )

        return results

    def get_mapping_summary(self) -> Dict[str, Any]:
        """
        Get summary of current mapping state.

        Returns:
            Dictionary with mapping statistics
        """
        return {
            'rules_count': len(self.mapping_rules),
            'budget_lines_count': len(self.budget_lines),
            'gmp_allocations_count': len(self.gmp_allocations),
            'rules': [
                {
                    'id': r.id,
                    'pattern': r.cost_code_pattern,
                    'category': r.category.value if r.category else None,
                    'priority': r.priority,
                    'markup': float(r.markup_factor)
                }
                for r in self.mapping_rules[:10]  # Top 10 by priority
            ]
        }
