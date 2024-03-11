from ArgCheckDecorator import checkCSV
from dataclasses import dataclass

import pandas


@dataclass
class ColumnData:
    count: float
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float


class DataLoader:
    def __init__(self, data_path):
        self.data = None
        self.by_column = None
        self.computed_data = None

        self.load_data(data_path)

    def count(self, column):
        return len(column)

    def mean(self, column):
        return sum(column) / len(column)

    def std(self, column):
        return (sum([(x - self.mean(column))**2 for x in column]) / len(column))**0.5

    def min(self, column):
        min = column[0]
        for i in column:
            if i < min:
                min = i
        return min

    def q1(self, column):
        return float(sorted(column)[len(column) // 4])

    def median(self, column):
        return (sorted(column)[len(column) // 2] + sorted(column)[(len(column) - 1) // 2]) / 2

    def q3(self, column):
        return (float(sorted(column)[int(len(column) / 4 * 3)]) + float(sorted(column)[int(len(column) / 4 * 3) - 1])) / 2

    def max(self, column):
        max = column[0]
        for i in column:
            if i > max:
                max = i
        return max

    def compute_data(self):
        computed_data = {}
        for key, value in self.by_column.items():
            try:
                _ = float(value[0])
                filtered_column = self.data[key].dropna()
                computed_data[key] = ColumnData(
                    self.count(filtered_column),
                    self.mean(filtered_column),
                    self.std(filtered_column),
                    self.min(filtered_column),
                    self.q1(filtered_column),
                    self.median(filtered_column),
                    self.q3(filtered_column),
                    self.max(filtered_column)
                )
            except ValueError:
                continue
        return computed_data

    @checkCSV
    def load_data(self, data_path):
        try:
            self.data = pandas.read_csv(data_path)
            if (self.data.columns[0] == "Index"):
                self.data.drop(self.data.columns[0], axis=1, inplace=True)

            self.by_column = {}
            for i in range(len(self.data.columns)):
                
                self.by_column[self.data.columns[i]] = self.data[self.data.columns[i]]
                

            self.computed_data = self.compute_data()
        except Exception as e:
            print(f"Error: {e}")
