# MLOps boundary

This folder contains only operations that happen after local model training:

- `tracking.py` optionally sends the completed local run to DagsHub through MLflow. Explicit production registration publishes a canonical protected-artifact manifest and writes its completion marker last.
- `integrity.py` owns the shared canonical JSON, path-safety, schema, hashing, inventory, and completion-marker rules. It does not provide a registry or artifact store.
- `registry.py` independently validates one exact numeric registered model version and its complete protected artifact inventory before model loading. It rejects aliases such as `latest` or `champion` and explicitly labels pre-manifest versions as legacy.
- `deployment.py` packages that validated version for inference and verifies the package at startup.
- `__main__.py` exposes the registry and deployment commands.

The normal flow is:

```text
DataIngestion -> ModelTrainer -> local artifacts -> optional ExperimentTracker
                                                -> exact registry version
                                                -> deployment package -> Modal
```

Nothing in this folder fits or refits a model. DagsHub MLflow remains the only remote source of truth, and the numeric registered-model version remains authoritative. Training remains in `src/components/model_trainer.py`, and the training pipeline only coordinates the existing components.
