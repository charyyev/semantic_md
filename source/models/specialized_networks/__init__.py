"""
This folder is used to make small changes to the network in category 2.
Why is this structure necessary? We want it to be independent of the actual model used
for the regression. To adjust to the variations of category 2, we only need to amke
slight adjustments (such as changing the number of in/output channels), which we do in
this package for organizational clarity.

Ideally start by reading model_utils to understand what each function does, before
diving into specific changes.
"""
