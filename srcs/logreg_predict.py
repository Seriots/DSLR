import pickle

with open('model.pkl', 'rb') as file:
	try:
		model = pickle.load(file)
	except Exception as e:
		print(e)
		exit(1)

print(model)