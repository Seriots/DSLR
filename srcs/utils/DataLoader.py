from dataclasses import dataclass

import pandas


def checkCSV(func):
    """Decorator to handle path error

    Args:
        func (fn): function to be decorated
    """
    def wrapper(*args, **kwargs):
        """Wrapper function to handle path error

        Args:
            path (str): The path to test

        Raises:
            ValueError: If the path is null or if its not a CSV file

        Returns:
            Exception: Handle all errors when opening a file
        """
        try:
            if len(args) != 2:
                raise ValueError("Invalid number of arguments")
            path = args[1]
            if not path.endswith(".csv"):
                raise ValueError("File is not a CSV")
            if not path:
                raise ValueError("File path is empty")
        except ValueError as e:
            print(f"Error in name file: {e}")
            return None
        return func(*args, **kwargs)
    return wrapper


def sort_columns(column: pandas.DataFrame, size: int):
    if size == 0:
        return column
    c = [c for c in column]
    for i, value in enumerate(c):
        if value == c[-1]:
            return c
        min = value
        index = i
        swap_index = i
        for j, v in enumerate(c[i+1:], start=i+1):

            if min > v:
                min = v
                swap_index = j
        tmp = c[index]
        c[index] = c[swap_index]
        c[swap_index] = tmp


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

    def count(self, column: pandas.DataFrame):
        count = 0
        for _ in column:
            count += 1
        return count

    def sum(self, column: pandas.DataFrame):
        sum = 0
        for i in column:
            sum += i
        return sum

    def mean(self, column: pandas.DataFrame):
        return self.sum(column) / self.count(column)

    def std(self, column: pandas.DataFrame):
        return (self.sum([(x - self.mean(column))**2 for x in column]) / (self.count(column) - 1))**0.5

    def min(self, column: pandas.DataFrame):
        min = column[0]
        for i in column:
            if i < min:
                min = i
        return min

    def quantile(self, column: pandas.DataFrame, percentage):
        if percentage < 0:
            return self.min(column)
        if percentage > 1:
            return self.max(column)
        base = percentage * (self.count(column) - 1)
        int_base = int(base)
        float_base = base - int_base
        return column[int_base] + (column[int_base + 1] - column[int_base]) * float_base

    def max(self, column: pandas.DataFrame):
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
                filtered_column: pandas.DataFrame = self.data[key].dropna()
                filtered_column = sort_columns(filtered_column, self.count(filtered_column))
                computed_data[key] = ColumnData(
                    self.count(filtered_column),
                    self.mean(filtered_column),
                    self.std(filtered_column),
                    self.min(filtered_column),
                    self.quantile(filtered_column, 0.25),
                    self.quantile(filtered_column, 0.50),
                    self.quantile(filtered_column, 0.75),
                    self.max(filtered_column)
                )
            except ValueError:
                continue
        self.computed_data = computed_data
        return computed_data

    @checkCSV
    def load_data(self, data_path):
        try:
            self.data = pandas.read_csv(data_path)
            start = 0
            if (self.data.columns[0] == "Index"):
                start = 1

            self.by_column = {}
            for i in range(start, len(self.data.columns)):

                self.by_column[self.data.columns[i]] = self.data[self.data.columns[i]]

        except Exception as e:
            print(f"Error: {e}")
