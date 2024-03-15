import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.LogisticRegression import LogisticRegression

data = pd.read_csv('data/dataset_train.csv')

used_features = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Care of Magical Creatures']

data = data.dropna(subset=used_features)

x_values = data[used_features]
y_value = data['Hogwarts House']
x_values = x_values.apply(lambda x: (x - x.mean()) / x.std())

slytherin = y_value.apply(lambda x: '1' if str(x) == 'Slytherin' else '0')
    
gryffindor = y_value.apply(lambda x: '1' if str(x) == 'Gryffindor' else '0')
ravenclaw = y_value.apply(lambda x: '1' if str(x) == 'Ravenclaw' else '0')
hufflepuff = y_value.apply(lambda x: '1' if str(x) == 'Hufflepuff' else '0')
slytherin = pd.concat([slytherin, x_values], axis=1)
gryffindor = pd.concat([gryffindor, x_values], axis=1)
ravenclaw = pd.concat([ravenclaw, x_values], axis=1)
hufflepuff = pd.concat([hufflepuff, x_values], axis=1)


SlytherinData = LogisticRegression(slytherin, 0.01)
GryffindorData = LogisticRegression(gryffindor, 0.01)
RavenclawData = LogisticRegression(ravenclaw, 0.01)
HufflepuffData = LogisticRegression(hufflepuff, 0.01)

SlytherinData.train(5000)
GryffindorData.train(5000)
RavenclawData.train(5000)
HufflepuffData.train(5000)


RED = '\33[31m'
GREEN = '\33[32m'
END = '\33[0m'

y_value = [v for v in y_value]

bad = 0
for i in range(len(x_values)):
    data_to_predict = x_values.iloc[i].values
    prediction = {"Slytherin" : SlytherinData.predict(data_to_predict), "Gryffindor": GryffindorData.predict(data_to_predict), "Ravenclaw": RavenclawData.predict(data_to_predict), "Hufflepuff": HufflepuffData.predict(data_to_predict)}

    bad += 1 if max(prediction, key=prediction.get) != y_value[i] else 0
    print(f"Prediction {max(prediction, key=prediction.get)}:{y_value[i]} Real {GREEN + 'GOOD' + END if max(prediction, key=prediction.get) == y_value[i] else RED + 'BAD' + END}")

print(f"Bad predictions: {bad}/{len(x_values)} = {bad / len(x_values) * 100:.2f}%")

    #print(f"Slytherin = {prediction[0] * 100:.2f}%, Gryffindor = {prediction[1] * 100:.2f}%, Ravenclaw = {prediction[2] * 100:.2f}%, Hufflepuff = {prediction[3] * 100:.2f}%")
#sns.scatterplot(x='Astronomy', y='Herbology', hue='Hogwarts House', data=data)
#x = np.linspace(-2, 2, 100)
#y = - (computedData.weights[0] * x + computedData.bias) / computedData.weights[1]
#plt.plot(x, y)
#plt.show()

#with open('data/test_normalized.csv', 'w') as f:
#    data.to_csv(f, index=False)

