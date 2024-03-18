import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper


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
            model = model['model']
            fig = plt.figure(figsize=(6, 4), num='Error history')
            for house in model.keys():
                sns.lineplot(x=range(len(model[house].error_history)), y=model[house].error_history, legend='full', label=house)
            plt.title('Error history')
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