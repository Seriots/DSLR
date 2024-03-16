import pandas as pd
import pickle

from utils.LogisticRegression import LogisticRegression
from utils.DataLoader import DataLoader
from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper


def check_data(func):
	"""Check if the data is correct before training the model
		- Check if the Hogwarts House is in the dataset
		- Check if the features are in the dataset
		- Check if there is data to use for the training
		- Remove the data with missing values"""
	def wrapper(data, features):
		if 'Hogwarts House' not in data.columns:
			raise ValueError("Hogwarts House not found in the dataset")
		if len(features) == 0:
			raise ValueError("No features to use for the training")
		for feature in features:
			if feature not in data.columns:
				raise ValueError(f"Feature {feature} not found in the dataset")
		data = data.dropna(subset=features + ['Hogwarts House'])
		if len(data) == 0:
			raise ValueError("No data to use for the training")
		return func(data, features)
	return wrapper


@check_data
def formate_data(data, features):
	"""Formate the data for the training. Normalize the features."""
	x_values = data[features]
	y_value = data["Hogwarts House"]
	x_values = x_values.apply(lambda x: (x - x.mean()) / x.std())

	slytherin = y_value.apply(lambda x: "1" if str(x) == "Slytherin" else "0")
	gryffindor = y_value.apply(lambda x: "1" if str(x) == "Gryffindor" else "0")
	ravenclaw = y_value.apply(lambda x: "1" if str(x) == "Ravenclaw" else "0")
	hufflepuff = y_value.apply(lambda x: "1" if str(x) == "Hufflepuff" else "0")
	
	gryffindor = pd.concat([gryffindor, x_values], axis=1)
	slytherin = pd.concat([slytherin, x_values], axis=1)
	hufflepuff = pd.concat([hufflepuff, x_values], axis=1)
	ravenclaw = pd.concat([ravenclaw, x_values], axis=1)

	return gryffindor, slytherin, hufflepuff, ravenclaw

def main():
	args_handler = ArgsHandler('Train a model on a dataset for DSLR project the model predict the hogwarts house', [
		ArgsObject('data_path', 'Path to the dataset.')
	], [
		OptionObject('help', 'Show this help message.', name='h', expected_type=bool, default=False, check_function=display_helper),
		OptionObject('epochs', 'Number of epochs for the training.', name='e', expected_type=int, default=1000),
		OptionObject('learning_rate', 'Learning rate for the training.', name='l', expected_type=float, default=0.01),
		OptionObject('save', 'Save the model in a file.', name='s', expected_type=str, default='model.pkl'),
		OptionObject('features', 'Features to use for the training.', name='f', expected_type=list, default=['Astronomy', 'Herbology']),
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
	try:
		gryffindor, slytherin, hufflepuff, ravenclaw = formate_data(data.data, user_input['features'])
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
		"Gryffindor": GryffindorModel,
		"Slytherin": SlytherinModel,
		"Hufflepuff": HufflepuffModel,
		"Ravenclaw": RavenclawModel
	}

	with open(user_input['save'], 'wb') as file:
		pickle.dump(saved_model, file)


if __name__ == "__main__":
	main()