# for methods that require selective optimization, ie contains on/off hypers, learning rate hypers, and such

OPTIMIZABLE_HYPERPARAMETERS = {
    "lu_merge": ["alpha", "theta"],
    "orth_pro": ["alpha"],
    "clyb_merge": ["alpha"],

    # ... other methods and their optimizable hyperparameters
}