"""
CLI Module - Command-line interface for GMP Recon App.

Provides management commands for:
- Cutover operations
- Data migration
- Model training
- System monitoring
"""

from .cutover_commands import cutover, register_commands

__all__ = ['cutover', 'register_commands']
