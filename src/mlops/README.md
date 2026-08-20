# MLOps boundary

This folder contains only operations that happen after local model training:

- `tracking.py` optionally sends the completed local run to DagsHub through MLflow. It also performs explicit production registration when `ENABLE_MODEL_REGISTRATION=true`.
- `registry.py` validates one exact numeric registered model version. It rejects aliases such as `latest` or `champion`.
- `deployment.py` packages that validated version for inference and verifies the package at startup.
- `__main__.py` exposes the registry and deployment commands.

The normal flow is:

```text
DataIngestion -> ModelTrainer -> local artifacts -> optional ExperimentTracker
                                                -> exact registry version
                                                -> deployment package -> Modal
```

Nothing in this folder fits or refits a model. Training remains in `src/components/model_trainer.py`, and the training pipeline only coordinates the existing components.
