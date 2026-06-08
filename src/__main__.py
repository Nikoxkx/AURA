"""
AURA - Direct execution entry point.

Usage:
    python -m src run          # Run single cycle
    python -m src daemon       # Run as daemon
    python -m src dashboard    # Start API dashboard
    python -m src status       # Show status
"""

from src.agent.cli import main

if __name__ == "__main__":
    main()
