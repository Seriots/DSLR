import pandas as pd


def check_data(func):
	"""Check if the data is correct before training the model
		- Check if the Hogwarts House is in the dataset
		- Check if the features are in the dataset
		- Check if there is data to use for the training
		- Remove the data with missing values"""
	def wrapper(data, features, removeNan):
		if 'Hogwarts House' not in data.columns:
			raise ValueError("Hogwarts House not found in the dataset")
		if len(features) == 0:
			raise ValueError("No features to use for the training")
		for feature in features:
			if feature not in data.columns:
				raise ValueError(f"Feature {feature} not found in the dataset")
		if removeNan:
			data = data.dropna(subset=features)
		if len(data) == 0:
			raise ValueError("No data to use for the training")
		return func(data, features, removeNan)
	return wrapper


@check_data
def formate_data(data: pd.DataFrame, features, removeNan=True):
	"""Formate the data for the training. Normalize the features."""
	x_values = data[features]
	y_value = data["Hogwarts House"]
	x_values = x_values.apply(lambda x: (x - x.mean()) / x.std())

	return x_values, y_value


def predict_class(x, models):
	"""Predict the class of the data using the models"""
	predictions = {}
	for house, model in models.items():
		predictions[house] = model.predict(x)
	return max(predictions, key=predictions.get)
