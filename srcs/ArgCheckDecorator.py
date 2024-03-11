
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
