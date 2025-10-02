"""
This file is responsible for 
visualizing feature importances
to the end user.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_names = np.array(feature_names)
    
    mask = importances > 0.02

    importances = importances[mask]
    feature_names = feature_names[mask]

    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_features = feature_names[indices]

    fig, ax = plt.subplots()
    ax.barh(sorted_features, sorted_importances)
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()

    return fig