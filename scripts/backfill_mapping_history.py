#!/usr/bin/env python3
"""
Backfill script for Enhanced Direct Cost → Budget Mapping.

Populates mapping_feedback and budget_match_stats tables from existing
direct_to_budget mappings. Run this once after applying the migration.

Usage:
    python scripts/backfill_mapping_history.py [--dry-run]
"""
import argparse
import re
import sys
from datetime import datetime
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, '.')

from sqlalchemy import func
from app.models import (
    SessionLocal, DirectToBudget, MappingFeedback, BudgetMatchStats
)


def normalize_vendor(vendor: str) -> str:
    """Normalize vendor name for consistent matching."""
    if not vendor:
        return ''
    # Lowercase, remove common suffixes, strip whitespace
    normalized = vendor.lower().strip()
    suffixes = [' inc', ' llc', ' corp', ' co', ' ltd', ' company', '.']
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def extract_name_prefix(name: str, max_length: int = 20) -> str:
    """Extract normalized prefix from name for pattern matching."""
    if not name:
        return ''
    # Take first N chars, lowercase
    prefix = name[:max_length].lower().strip()
    return prefix


def backfill_mapping_feedback(db, dry_run: bool = False):
    """
    Populate mapping_feedback from existing direct_to_budget records.
    Creates one feedback entry per unique (vendor_normalized, name_prefix, budget_code).
    """
    print("\n[1/2] Backfilling mapping_feedback table...")

    # Get all existing mappings
    mappings = db.query(DirectToBudget).all()
    print(f"  Found {len(mappings)} existing DirectToBudget records")

    # Aggregate by pattern
    patterns = defaultdict(list)
    for m in mappings:
        vendor_norm = normalize_vendor(m.vendor_normalized or '')
        name_prefix = extract_name_prefix(m.name or '')

        if not vendor_norm and not name_prefix:
            continue

        key = (vendor_norm, name_prefix, m.budget_code)
        patterns[key].append(m)

    print(f"  Identified {len(patterns)} unique patterns")

    # Create feedback records
    created = 0
    skipped = 0

    for (vendor_norm, name_prefix, budget_code), records in patterns.items():
        # Check if pattern already exists
        existing = db.query(MappingFeedback).filter(
            MappingFeedback.vendor_normalized == vendor_norm,
            MappingFeedback.name_prefix == name_prefix,
            MappingFeedback.budget_code == budget_code
        ).first()

        if existing:
            skipped += 1
            continue

        # Use the earliest record's timestamp
        earliest = min(records, key=lambda r: r.created_at or datetime.utcnow())

        feedback = MappingFeedback(
            vendor_normalized=vendor_norm,
            name_prefix=name_prefix,
            budget_code=budget_code,
            was_override=False,  # Historical records are treated as non-overrides
            confidence_at_suggestion=earliest.confidence,
            user_id='backfill',
            created_at=earliest.created_at or datetime.utcnow()
        )

        if not dry_run:
            db.add(feedback)
        created += 1

    if not dry_run:
        db.commit()

    print(f"  Created: {created}, Skipped (existing): {skipped}")
    return created


def backfill_budget_match_stats(db, dry_run: bool = False):
    """
    Populate budget_match_stats with aggregate counts from direct_to_budget.
    """
    print("\n[2/2] Backfilling budget_match_stats table...")

    # Count mappings per budget code
    counts = db.query(
        DirectToBudget.budget_code,
        func.count(DirectToBudget.id).label('count')
    ).group_by(DirectToBudget.budget_code).all()

    print(f"  Found {len(counts)} distinct budget codes with mappings")

    created = 0
    updated = 0

    for budget_code, count in counts:
        if not budget_code:
            continue

        existing = db.query(BudgetMatchStats).filter(
            BudgetMatchStats.budget_code == budget_code
        ).first()

        if existing:
            if not dry_run:
                existing.total_matches = count
                existing.last_updated = datetime.utcnow()
            updated += 1
        else:
            stats = BudgetMatchStats(
                budget_code=budget_code,
                total_matches=count,
                override_count=0,
                trust_score=1.0,
                last_updated=datetime.utcnow()
            )
            if not dry_run:
                db.add(stats)
            created += 1

    if not dry_run:
        db.commit()

    print(f"  Created: {created}, Updated: {updated}")
    return created + updated


def update_vendor_normalized(db, dry_run: bool = False):
    """
    Backfill vendor_normalized column in direct_to_budget for existing records.
    This requires the original direct costs data, so we'll mark it for manual update.
    """
    print("\n[Note] vendor_normalized backfill:")

    # Count records missing vendor_normalized
    missing = db.query(DirectToBudget).filter(
        DirectToBudget.vendor_normalized.is_(None)
    ).count()

    if missing > 0:
        print(f"  {missing} records have NULL vendor_normalized")
        print("  These will be populated automatically during the next mapping save.")
    else:
        print("  All records have vendor_normalized populated.")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill mapping history tables from existing data'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 60)

    print("Backfill Mapping History")
    print("=" * 60)
    print(f"Started at: {datetime.utcnow().isoformat()}")

    db = SessionLocal()

    try:
        feedback_count = backfill_mapping_feedback(db, args.dry_run)
        stats_count = backfill_budget_match_stats(db, args.dry_run)
        update_vendor_normalized(db, args.dry_run)

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  mapping_feedback records created: {feedback_count}")
        print(f"  budget_match_stats records processed: {stats_count}")

        if args.dry_run:
            print("\n[DRY RUN] No changes were made to the database")
        else:
            print("\nBackfill completed successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()
