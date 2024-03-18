import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (array-like, optional): List of labels to index the matrix.
            If not provided, labels will be inferred from the input data.
    Returns:
        array-like: Confusion matrix.
    """
    if labels is None:
        labels = np.unique(y_true)
    
    num_labels = len(labels)
    conf_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        true_index = np.where(labels == true_label)[0][0]
        pred_index = np.where(labels == pred_label)[0][0]
        conf_matrix[true_index, pred_index] += 1
    
    return conf_matrix

def main():
    args_handler = ArgsHandler('Display some graph about a model', [
        ArgsObject('model_path', 'Path to the model to load.')
    ], [
        OptionObject('help', 'Show this help message.', name='h', expected_type=bool, default=False, check_function=display_helper),
    ], """""")

    try:
        user_input = args_handler.parse_args()
        args_handler.check_args(user_input)
    except SystemExit:
        return
    except Exception as e:
        print(e)
        return
    
    try:
        with open(user_input['args'][0], 'rb') as file:
            model = pickle.load(file)
            validation_real = model['validation_real']
            validation_pred = model['validation_pred']
            model = model['model']
            fig = plt.figure(figsize=(9, 4), num='Data window')

            if not (len(validation_real) == 0 or len(validation_real) != len(validation_pred)):
                fig.add_subplot(1, 2, 1)

            # display error history
            for house in model.keys():
                sns.lineplot(x=range(len(model[house].error_history)), y=model[house].error_history, legend='full', label=house)
            plt.title('Error history')

            # display confusion matrix if validation data is available
            if not (len(validation_real) == 0 or len(validation_real) != len(validation_pred)):
                fig.add_subplot(1, 2, 2)
                conf_matrix = confusion_matrix(validation_real, validation_pred)
                sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
                plt.xticks(ticks=np.arange(len(np.unique(validation_real))) + 0.5, labels=np.unique(validation_real))
                plt.xlabel('Predicted House')
                plt.yticks(ticks=np.arange(len(np.unique(validation_real))) + 0.5, labels=np.unique(validation_real))
                plt.ylabel('True House')
                plt.title('Confusion Matrix')

    except Exception as e:
        print(e)
        exit(1)

    try:
        plt.show()
    except KeyboardInterrupt as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()