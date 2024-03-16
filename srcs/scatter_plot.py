from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper
from utils.DataLoader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

def create_feature_plot(main_feature, data_to_plot):
    fig = plt.figure(figsize=(8, 6))
    new_data = filter(lambda x: x != main_feature, data_to_plot.columns)
    index_plot = 1
    for column in new_data:
        if (index_plot > 12):
            break
        ax = fig.add_subplot(3, 4, index_plot)
        plt.subplots_adjust(wspace=0.8, hspace=0.8)
        sns.scatterplot(data=data_to_plot, x=main_feature, y=column, ax=ax)
        plt.xlabel(main_feature[:min(len(main_feature), 12)])
        plt.ylabel(column[:min(len(column), 12)])
        index_plot += 1
    plt.suptitle(f"{main_feature} vs all other features")


def main():
    args_handler = ArgsHandler('Make a scatter plot of a features with all others ones', [
        ArgsObject('data_path', 'Path to the dataset.')
    ], [
        OptionObject('help', 'Show this help message.', name='h', expected_type=bool, default=False, check_function=display_helper),
        OptionObject('main-feature', """The main feature to compare with the others.
                        Default value '*' is used to plot all features
                            All available feature are:
                                Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts,
                                Divination, Muggle Studies, Ancient Runes, History of Magic,
                                Transfiguration, Potions, Care of Magical Creatures, Charms, Flying""", name='m', expected_type=str, default='*'),
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

    all_features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
                                'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                                'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

    for all_feature in all_features:
        if all_feature not in data.data.columns:
            print(f"Feature {all_feature} not found in the dataset")
            return

    data_to_plot = data.data[all_features]
    
    if 'main-feature' in user_input:
        if user_input['main-feature'] == '*':
            for row in data_to_plot.columns:
                create_feature_plot(row, data_to_plot)
        else:
            if user_input['main-feature'] not in data_to_plot.columns:
                print("The main feature is not in the dataset")
                return
            create_feature_plot(user_input['main-feature'], data_to_plot)
        
    else:
        while(True):
            main_feature = input("Enter the main feature to compare with the others: ")
            if main_feature == "":
                break
            if main_feature not in data_to_plot.columns:
                print("This feature is not in the dataset")
                continue
            create_feature_plot(main_feature, data_to_plot)
                
            try:
                plt.show()
            except KeyboardInterrupt:
                print("Interrupted by user")
                continue

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
