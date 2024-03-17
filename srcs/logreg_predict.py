import pickle
import pandas as pd

from utils.DataLoader import DataLoader
from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject, display_helper
from utils.utils import formate_data, predict_class


def main():
	args_handler = ArgsHandler('Train a model on a dataset for DSLR project the model predict the hogwarts house', [
		ArgsObject('data_path', 'Path to the dataset.')
	], [
		OptionObject('help', 'Show this help message.', name='h', expected_type=bool, default=False, check_function=display_helper),
		OptionObject('save', 'File where the model is saved', name='s', expected_type=str, default='model.pkl'),
		OptionObject('output', 'File where the prediction is saved', name='o', expected_type=str, default='houses.csv')
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

	with open(user_input['save'], 'rb') as file:
		try:
			model = pickle.load(file)
			feature = model['features']
			model = model['model']

			x_values, _ = formate_data(data.data, feature)

			predictions: pd.Series = x_values.apply(lambda x: predict_class(x, model), axis=1)
			predictions = predictions.to_frame()
			predictions.columns = ['Hogwarts House']
			predictions.to_csv(user_input['output'], header=True, index=True, index_label='Index', )
		except Exception as e:
			print(e)
			exit(1)


if __name__ == "__main__":
	main()