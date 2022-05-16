import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from mlxtend.plotting import plot_confusion_matrix


def skew_plot(df, col_name):
    """_summary_

    Args:
        df (_type_): _description_
        col_name (_type_): _description_
    """
    plt.figure()
    plt.subplot(2, 2, 1)
    # plt.hist(df2.AGE, bins=40)
    sns.histplot(data=df, x=col_name)
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x=col_name)
    plt.show()

    return


def mod_sum(df_test_Y, y_pred, y_pred_prob):
    """_summary_

    Args:
        df_test_Y (_type_): _description_
        y_pred (_type_): _description_
        y_pred_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    # to plot and understand confusion matrix
    cm = confusion_matrix(df_test_Y, y_pred)
    plot_confusion_matrix(cm)
    plt.gcf().set_dpi(100)
    plt.show()

    # Model Evaluation
    ac_sc = accuracy_score(df_test_Y, y_pred)
    rc_sc = recall_score(df_test_Y, y_pred, average="weighted")
    pr_sc = precision_score(df_test_Y, y_pred, average="weighted")
    f1_sc = f1_score(df_test_Y, y_pred, average="micro")
    roc_sc = roc_auc_score(df_test_Y, y_pred_prob)

    print(
        "Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, ROC-AUC {:.2f}".format(
            ac_sc, rc_sc, pr_sc, f1_sc, roc_sc
        )
    )
    print(classification_report(df_test_Y, y_pred))
    return ac_sc
