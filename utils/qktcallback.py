class QKTCallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        """Initialise the class with a data array."""
        self._data = [[] for i in range(5)]

    def callback(
        self,
        n_func_iter,
        weights=None,
        func_value=None,
        step_size=None,
        step_accepted=None,
    ):
        """
        Args
            n_func_iter (int): number of function evaluations.
            weights (np.ndarray): the current weights.
            func_value (float): the function value.
            step_size (float): the step size.
            step_accepted (bool): whether the step was accepted.

        Returns:
            None
        """
        self._data[0].append(n_func_iter)
        self._data[1].append(weights)
        self._data[2].append(func_value)
        self._data[3].append(step_size)
        self._data[4].append(step_accepted)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]
