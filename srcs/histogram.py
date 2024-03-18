from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper
from utils.DataLoader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


def split_by_house(data):
    if 'Hogwarts House' not in data.columns:
        raise ValueError("No 'Hogwarts House' column in the dataset")
    try:
        data = data[['Index','Hogwarts House',
        'Arithmancy',
        'Astronomy',
        'Herbology',
        'Defense Against the Dark Arts',
        'Divination',
        'Muggle Studies',
        'Ancient Runes',
        'History of Magic',
        'Transfiguration',
        'Potions',
        'Care of Magical Creatures',
        'Charms',
        'Flying']]
    except KeyError:
        raise ValueError("Some columns are missing in the dataset")
    g = data[data['Hogwarts House'] == 'Gryffindor']
    r = data[data['Hogwarts House'] == 'Ravenclaw']
    h = data[data['Hogwarts House'] == 'Hufflepuff']
    s = data[data['Hogwarts House'] == 'Slytherin']
    return {'Gryffindor': g, 'Ravenclaw': r, 'Hufflepuff': h, 'Slytherin': s, 'Columns': data.columns[2:]}


def main():
    args_handler = ArgsHandler('Make an histogram for all features about hogwarts houses', [
        ArgsObject('data_path', 'Path to the dataset.')
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

    data = DataLoader(user_input['args'][0])
    if data.data is None:
        return
    print("Data loaded")

    try:
        house_data = split_by_house(data.data)
    except ValueError as e:
        print(e)
        return

    fig = plt.figure(figsize=(10, 5))
    for i, column in enumerate(house_data['Columns']):
        if (i >= 15):
            break
        ax = fig.add_subplot(3, 5, i+1)
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        sns.histplot(house_data['Ravenclaw'][column], kde=False, label='Ravenclaw', legend=True)
        sns.histplot(house_data['Hufflepuff'][column], kde=False, label='Hufflepuff', legend=True)
        sns.histplot(house_data['Slytherin'][column], kde=False, label='Slytherin', legend=True)
        sns.histplot(house_data['Gryffindor'][column], kde=False, label='Gryffindor', legend=True)
        plt.title(column)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

    print("Data plotted")
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
