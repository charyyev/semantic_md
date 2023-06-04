"""
All important comments are in base_trainer.py
The other models simply overwrite some functions, typically
1) building the model according to the specified way
2) running the training (e.g. with multiple losses / prediction tasks)
3) checking data_flag compatibility

The more specific models are basically just adaptations for the different model
architectures
"""