from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
import plotly.express as px

def compute_metrics(y_true, y_pred, mode='binary', y_prob=None):
    """
    Compute metrics for a classification problem.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    mode : str, default='binary'
        The type of classification problem.
    y_prob : array-like of shape (n_samples,), default=None
        Predicted probabilities.
        
    Returns
    -------
    full_metrics : dict
        Dictionary containing the computed metrics.        
    """
    full_metrics={}
    full_metrics['precision'] = precision_score(y_true, y_pred, average=mode)
    full_metrics['recall'] = recall_score(y_true, y_pred, average=mode)
    full_metrics['f1_score'] = f1_score(y_true, y_pred, average=mode)
    full_metrics['accuracy'] = accuracy_score(y_true, y_pred)
    if y_prob is None:
        full_metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
    else: 
        full_metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    # Confusion matrix
    full_metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    # Classification report
    full_metrics['classification_report'] = classification_report(y_true, y_pred)
    
    def plot_confusion_matrix(metrics):
        conf_matrix = metrics['confusion_matrix']
        conf_matrix_fig = px.imshow(conf_matrix, 
                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=['Not Transported', 'Transported'],
                                    y=['Not Transported', 'Transported'],
                                    title="Confusion Matrix")

        conf_matrix_fig.update_layout(coloraxis_showscale=False)
        
        return conf_matrix_fig
    
    full_metrics['confusion_matrix_fig'] = plot_confusion_matrix(full_metrics)
    
    return full_metrics
