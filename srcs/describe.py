from ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from DataLoader import DataLoader


def main():
    args_handler = ArgsHandler('Read a csv and write a desxcription of it', [
        ArgsObject('data_path', 'Path to the dataset.')
    ], [
        OptionObject('help', 'Show this help message.', name='h', expected_type=bool),
        OptionObject('output', 'The file to save the description', name='o', expected_type=str),
    ], """""")

    try:
        user_input = args_handler.parse_args()
    except Exception as e:
        print(e)
        return
    if "help" in user_input:
        print(args_handler.full_help())
        return

    try:
        args_handler.check_args(user_input)
    except Exception as e:
        print(e)
        return
   
    data = DataLoader(user_input['args'][0])
    print( "         " + "".join([f"{key: >14.8s}" for key in data.computed_data.keys()]))
    print()
    print(f"Count: " + "".join([f"{value.count: >14d}" for value in data.computed_data.values()]))
    print(f"Mean:  " + "".join([f"{value.mean: >14.4f}" for value in data.computed_data.values()]))
    print(f"Std:   " + "".join([f"{value.std: >14.4f}" for value in data.computed_data.values()]))
    print(f"Min:   " + "".join([f"{value.min: >14.4f}" for value in data.computed_data.values()]))
    print(f"25%:   " + "".join([f"{value.q1: >14.4f}" for value in data.computed_data.values()]))
    print(f"50%:   " + "".join([f"{value.median: >14.4f}" for value in data.computed_data.values()]))
    print(f"75%:   " + "".join([f"{value.q3: >14.4f}" for value in data.computed_data.values()]))
    print(f"Max:   " + "".join([f"{value.max: >14.4f}" for value in data.computed_data.values()]))


if __name__ == "__main__":
    main()