Legacy utilities were consolidated into `PINN/src`.

- `utils/experiment_logger.py` is now a compatibility shim that re-exports
  architecture-search helpers from `src/experiment_logging.py`.
- New development should import from `src.experiment_logging` directly.

