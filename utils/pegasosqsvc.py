# # from qiskit_machine_learning.algorithms import PegasosQSVC
# # from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# # import numpy as np


# # def run_PegasosQSVC(training_input, train_labels, test_input, test_labels, quantum_kernel, C=1.0, num_steps=1000, seed=1024):
# #     """Fit and evaluate a Pegasos Quantum Support Vector Classifier.

# #     Args:
# #         training_input (np.ndarray): training sample features.
# #         train_labels (np.ndarray): training sample labels.
# #         test_input (np.ndarray): testing sample features.
# #         test_labels (np.ndarray): testing sample labels.
# #         quantum_kernel (QuantumKernel): the quantum kernel to use.
# #         C (float): regularization parameter.
# #         num_steps (int): number of iterations for Pegasos.
# #         seed (int): random seed.

# #     Returns:
# #         model (PegasosQSVC): fitted PegasosQSVC object.
# #         auc (float): ROC AUC score.
# #         f1 (float): F1 score.
# #         acc (float): Accuracy score.
# #     """
# #     # Create the PegasosQSVC
# #     model = PegasosQSVC(quantum_kernel=quantum_kernel, C=C, num_steps=num_steps, seed=seed)

# #     # Fit the model
# #     model.fit(training_input, train_labels)

# #     # Predictions
# #     predicted = model.predict(test_input)

# #     # PegasosQSVC does not support probability outputs directly → use decision_function instead
# #     try:
# #         decision_scores = model.decision_function(test_input)
# #         if decision_scores.ndim == 1:  # binary classification
# #             predicted_proba = np.vstack([1 - decision_scores, decision_scores]).T
# #         else:  # multi-class
# #             predicted_proba = decision_scores
# #     except:
# #         predicted_proba = None

# #     # F1 score
# #     f1 = f1_score(test_labels, predicted, average="weighted")

# #     # AUC
# #     auc = None
# #     if predicted_proba is not None:
# #         try:
# #             auc = roc_auc_score(test_labels, predicted_proba, multi_class="ovo", average="weighted")
# #         except:
# #             try:
# #                 auc = roc_auc_score(test_labels, predicted_proba[:, 1])
# #             except:
# #                 auc = None

# #     # Accuracy
# #     acc_score = accuracy_score(test_labels, predicted)

# #     return model, auc, f1, acc_score
# from qiskit_machine_learning.algorithms import PegasosQSVC
# from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# def run_PegasosQSVC(training_input, train_labels, test_input, test_labels, quantum_kernel, C=1.0, num_steps=1000, seed=1024):
#     # Create the model
#     model = PegasosQSVC(quantum_kernel=quantum_kernel, C=C, num_steps=num_steps, seed=seed)
#     model.fit(training_input, train_labels)

#     # Predictions
#     predicted = model.predict(test_input)

#     # Decision scores for ROC AUC
#     try:
#         decision_scores = model.decision_function(test_input)
#     except:
#         decision_scores = None

#     # F1 score
#     f1 = f1_score(test_labels, predicted, average="weighted")

#     # ROC AUC
#     auc = None
#     if decision_scores is not None:
#         try:
#             if decision_scores.ndim == 1:  # binary
#                 auc = roc_auc_score(test_labels, decision_scores)
#             else:  # multi-class
#                 auc = roc_auc_score(test_labels, decision_scores, multi_class="ovo", average="weighted")
#         except:
#             auc = None

#     # Accuracy
#     acc_score = accuracy_score(test_labels, predicted)

#     return model, auc, f1, acc_score

from qiskit_machine_learning.algorithms import PegasosQSVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
from scipy.special import expit  # For probability conversion if needed

def run_PegasosQSVC(training_input, train_labels, test_input, test_labels, quantum_kernel, C=1.0, num_steps=1000, seed=1024):
    """Fit and evaluate a Pegasos Quantum Support Vector Classifier.

    Args:
        training_input (np.ndarray): training sample features.
        train_labels (np.ndarray): training sample labels.
        test_input (np.ndarray): testing sample features.
        test_labels (np.ndarray): testing sample labels.
        quantum_kernel (QuantumKernel): the quantum kernel to use.
        C (float): regularization parameter.
        num_steps (int): number of iterations for Pegasos.
        seed (int): random seed.

    Returns:
        model (PegasosQSVC): fitted PegasosQSVC object.
        auc (float): ROC AUC score.
        f1 (float): F1 score.
        acc_score (float): Accuracy score.
    """
    # Input validation
    if not (len(training_input) == len(train_labels) and len(test_input) == len(test_labels)):
        raise ValueError("Mismatch in number of samples and labels")

    # Create and fit the model
    model = PegasosQSVC(quantum_kernel=quantum_kernel, C=C, num_steps=num_steps, seed=seed)
    model.fit(training_input, train_labels)

    # Predictions
    predicted = model.predict(test_input)

    # Get probabilities using predict_proba (preferred for ROC AUC)
    predicted_proba = None
    try:
        predicted_proba = model.predict_proba(test_input)
        if predicted_proba.shape[1] != 2:  # Ensure binary classification
            raise ValueError("predict_proba returned unexpected shape for binary classification")
    except Exception as e:
        print(f"Error in predict_proba: {e}")

    # F1 score
    f1 = f1_score(test_labels, predicted, average="weighted")

    # ROC AUC
    auc = None
    if predicted_proba is not None:
        try:
            auc = roc_auc_score(test_labels, predicted_proba[:, 0])  # Probability of class 1
        except Exception as e:
            print(f"Error calculating ROC AUC: {e}")
   

    # # Accuracy
    acc_score = accuracy_score(test_labels, predicted)

    return model, auc, f1, acc_score  # Fixed typo from acc_sc to acc_score