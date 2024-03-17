import pandas as pd
import pickle

from utils.LogisticRegression import LogisticRegression
from utils.DataLoader import DataLoader
from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper
from utils.utils import formate_data, predict_class


def main():
	args_handler = ArgsHandler('Train a model on a dataset for DSLR project the model predict the hogwarts house', [
		ArgsObject('data_path', 'Path to the dataset.')
	], [
		OptionObject('help', 'Show this help message.', name='h', expected_type=bool, default=False, check_function=display_helper),
		OptionObject('epochs', 'Number of epochs for the training.', name='e', expected_type=int, default=1000),
		OptionObject('learning_rate', 'Learning rate for the training.', name='l', expected_type=float, default=0.01),
		OptionObject('save', 'Save the model in a file.', name='s', expected_type=str, default='model.pkl'),
		OptionObject('features', 'Features to use for the training.', name='f', expected_type=list, default=['Astronomy', 'Herbology']),
		OptionObject('validation-percent', 'Percentage of the dataset to use for the validation.', name='p', expected_type=float, default=0.2),
		OptionObject('verbose', 'Show the progress of the training.', name='v', expected_type=bool, default=False)
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
	
	if user_input['verbose']:
		print("Data loaded")

	data_train = data.data.sample(frac=1 - user_input['validation-percent'])
	data_validation = data.data.drop(data_train.index)

	try:
		x_values, y_value = formate_data(data_train, user_input['features'])

		slytherin = y_value.apply(lambda x: "1" if str(x) == "Slytherin" else "0")
		gryffindor = y_value.apply(lambda x: "1" if str(x) == "Gryffindor" else "0")
		ravenclaw = y_value.apply(lambda x: "1" if str(x) == "Ravenclaw" else "0")
		hufflepuff = y_value.apply(lambda x: "1" if str(x) == "Hufflepuff" else "0")
		
		gryffindor = pd.concat([gryffindor, x_values], axis=1)
		slytherin = pd.concat([slytherin, x_values], axis=1)
		hufflepuff = pd.concat([hufflepuff, x_values], axis=1)
		ravenclaw = pd.concat([ravenclaw, x_values], axis=1)
	except ValueError as e:
		print(e)
		return

	learning_rate = user_input['learning_rate']
	epochs = user_input['epochs']

	GryffindorModel = LogisticRegression(gryffindor, learning_rate)
	SlytherinModel = LogisticRegression(slytherin, learning_rate)
	HufflepuffModel = LogisticRegression(hufflepuff, learning_rate)
	RavenclawModel = LogisticRegression(ravenclaw, learning_rate)

	GryffindorModel.train(epochs)
	SlytherinModel.train(epochs)
	HufflepuffModel.train(epochs)
	RavenclawModel.train(epochs)

	saved_model = {
		'model': {
			"Gryffindor": GryffindorModel,
			"Slytherin": SlytherinModel,
			"Hufflepuff": HufflepuffModel,
			"Ravenclaw": RavenclawModel
			},
		"features": user_input['features']
	}

	with open(user_input['save'], 'wb') as file:
		pickle.dump(saved_model, file)

	if user_input['verbose']:
		print(f"Model saved in {user_input['save']}")

	x_values, y_value = formate_data(data_validation, user_input['features'])

	RED = "\33[31m"
	GREEN = "\33[32m"
	END = "\33[0m"

	count = 0
	if user_input['verbose']:
		print("Validation:")
		prediction = x_values.apply(lambda x: predict_class(x, saved_model), axis=1)
		for i in range(len(x_values)):
			count += 1 if prediction.iloc[i] == y_value.iloc[i] else 0
			print(f"	{GREEN if prediction.iloc[i] == y_value.iloc[i] else RED}Predicted: {prediction.iloc[i]} - {y_value.iloc[i]} :Real{END}")

		print(f"Accuracy: {count / len(x_values) * 100:.2f}%")


if __name__ == "__main__":
	main()
