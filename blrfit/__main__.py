"""Allow ``python -m blrfit`` without relying on a shell's executable path."""
from .cli import main

raise SystemExit(main())
