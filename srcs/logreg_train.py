import pandas as pd
import pickle

from utils.LogisticRegression import LogisticRegression
from utils.DataLoader import DataLoader
from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper
from utils.utils import formate_data, predict_class

def default_features(args_handler, user_input):
	if 'features' not in user_input or user_input['features'] is None:
		user_input['features'] = ['Herbology', 'Astronomy']
	return user_input

def check_mode(args_handler, user_input):
	if 'stochastic-mode' in user_input and 'batch-mode' in user_input and user_input['stochastic-mode'] and user_input['batch-mode'] > 0:
		raise ValueError("You can't use the stochastic and the batch mode at the same time")
	return user_input

def main():
	args_handler = ArgsHandler('Train a model on a dataset for DSLR project the model predict the hogwarts house', [
		ArgsObject('data_path', 'Path to the dataset.')
	], [
		OptionObject('help', 'Show this help message.', name='h', expected_type=bool, default=False, check_function=display_helper),
		OptionObject('epochs', 'Number of epochs for the training.', name='e', expected_type=int, default=1000),
		OptionObject('learning_rate', 'Learning rate for the training.', name='l', expected_type=float, default=0.01),
		OptionObject('save', 'Save the model in a file.', name='s', expected_type=str, default='model.pkl'),
		OptionObject('features', 'Features to use for the training.', name='f', expected_type=list, default=None, check_function=default_features),
		OptionObject('validation-percent', 'Percentage of the dataset to use for the validation.', name='p', expected_type=float, default=0.2),
		OptionObject('verbose', 'Show the progress of the training.', name='v', expected_type=bool, default=False),
		OptionObject('stochastic-mode', 'Use the stochastic mode for the training.', expected_type=bool, default=False, check_function=check_mode),
		OptionObject('batch-mode', 'use the batch mode for the training, with a batch of the size given', expected_type=int, default=0, check_function=check_mode)
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
		x_values, y_value = formate_data(data_train, user_input['features'], True)

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
	mode = ["stochastic"] if user_input['stochastic-mode'] else ["mini-batch", user_input['batch-mode']] if user_input['batch-mode'] > 0 else ["batch"]

	GryffindorModel = LogisticRegression(gryffindor, learning_rate)
	SlytherinModel = LogisticRegression(slytherin, learning_rate)
	HufflepuffModel = LogisticRegression(hufflepuff, learning_rate)
	RavenclawModel = LogisticRegression(ravenclaw, learning_rate)

	GryffindorModel.train(epochs, mode)
	SlytherinModel.train(epochs, mode)
	HufflepuffModel.train(epochs, mode)
	RavenclawModel.train(epochs, mode)

	saved_model = {
		'model': {
			"Ravenclaw": RavenclawModel,
			"Gryffindor": GryffindorModel,
			"Slytherin": SlytherinModel,
			"Hufflepuff": HufflepuffModel
			},
		"features": user_input['features']
	}

	try:
		with open(user_input['save'], 'wb') as file:
			pickle.dump(saved_model, file)
	except Exception as e:
		print(e)
		return

	if user_input['verbose']:
		print(f"Model saved in {user_input['save']}")

	if user_input['verbose'] and data_validation.shape[0] >= 10:
		x_values, y_value = formate_data(data_validation, user_input['features'], True)

		RED = "\33[31m"
		GREEN = "\33[32m"
		END = "\33[0m"

		count = 0
		print("Validation:")
		prediction = x_values.apply(lambda x: predict_class(x, saved_model['model']), axis=1)
		for i in range(len(x_values)):
			count += 1 if prediction.iloc[i] == y_value.iloc[i] else 0
			print(f"	{GREEN if prediction.iloc[i] == y_value.iloc[i] else RED}Predicted: {prediction.iloc[i]} - {y_value.iloc[i]} :Real{END}")

		print(f"Accuracy: {count / len(x_values) * 100:.2f}%")


if __name__ == "__main__":
	main()
