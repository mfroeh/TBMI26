import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    corr = 0
    for truth, pred in zip(LPred, LTrue):
        if truth == pred:
            corr += 1
    acc = corr / len(LPred)

    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    classes = np.unique(LTrue)
    nClasses = len(classes)

    # cM[i, j]: How many times we predicted i, when it was j
    cM = np.zeros((nClasses, nClasses), dtype=np.int32)
    for pred, truth in zip(LPred, LTrue):
        cM[pred, truth] += 1

    # ============================================
    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    total = 0
    for i in range(cM.shape[0]):
        for j in range(cM.shape[1]):
            total += cM[i, j]
    acc = sum(np.diag(cM)) / total

    # ============================================
    return acc