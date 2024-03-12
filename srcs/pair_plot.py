from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper
from utils.DataLoader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    args_handler = ArgsHandler('Make an histogram for all features about hogwarts houses', [
        ArgsObject('data_path', 'Path to the dataset.')
    ], [
        OptionObject('help', 'Show this help message.', name='h', expected_type=bool, default=False, check_function=display_helper),
        OptionObject('plot-size', 'Size of the plot', name='s', expected_type=float, default=-1),
        OptionObject('data', """All features that you want to display in the pair plot\n
                            All available features are:
                                Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts,
                                Divination, Muggle Studies, Ancient Runes, History of Magic,
                                Transfiguration, Potions, Care of Magical Creatures, Charms, Flying""", name='d', expected_type=list),
    ], """""")

    try:
        user_input = args_handler.parse_args()
        args_handler.check_args(user_input)
    except SystemExit:
        return
    except Exception as e:
        print(e)
        return

    data = DataLoader(user_input['args'][0])
    if data.data is None:
        return
    print("Data loaded")

    if 'data' not in user_input:
        features = ['Astronomy', 'Herbology']
    elif user_input['data'] == ['*']:
        features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
                               'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                               'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    else:
        features = user_input['data']
        features = [feature.strip() for feature in features]
    for i, feature in enumerate(features):
        if feature not in data.data.columns:
            print(f"The feature {feature} is not in the dataset")
            return
        if feature in features[:i]:
            print(f"The feature {feature} is duplicated")
            return

    if 'plot-size' in user_input and user_input['plot-size'] > 0:
        plot_size = user_input['plot-size']
    else:
        plot_size = 12/len(features)

    sns.pairplot(data.data, x_vars=features, y_vars=features, hue='Hogwarts House', height=plot_size)

    print("Data plotted")
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
