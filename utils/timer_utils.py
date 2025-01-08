# This code is taken from 
# https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/utils/timer_utils.py

import time
from typing import Type
from collections import defaultdict


class Timer:
    '''
    Class to measure time for different parts of the code and compute average execution times
    '''
    def __init__(self) -> None:
        """
        Initialize the timer object.
        """
        self._reset()

    def tick(self, key: str) -> None:
        """
        This method start the timer for a particular key. 
        It records the current time as the start time for the specified key.

        Args:
            key (str): key of the task

        Raises:
            ValueError: If key is alread initialized.
        """
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key: str) -> None:
        """
        This method stops the timer for a particular key. 
        It calculates the elapsed time since the corresponding tick-call.

        Args:
            key (str): key of the task

        Raises:
            ValueError: If key was not initialized yet
        """
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def context(self, key):
        """
        This method returns a context manager object "_TimerContextManager".
        It allows timing passages of code using the "with"-statement.

        Args:
            key (str): Key to get the timing passing for.

        Returns:
            _type_: _TimeContextManager object
        """
        return _TimerContextManager(self, key)

    def get_average_times(self, reset=True):
        """
        This method computes the average execution times for each key in the timer's records.
        It returns a dictionary where each key is mapped to its corresponding average execution time.

        Args:
            reset (bool, optional): If this is set to True, resets the timers.

        Returns:
            dictionary of average execution times.
        """
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self._reset()
        return ret
    
    def _reset(self):
        """
        This private method resets the timer's internal state. 
        It clears the counts, times and start_times dictionaries.
        """
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}


class _TimerContextManager:
    '''
    Helper class that allows timer to time passages of code using the with statement (with timer.context("key"): ...)
    '''
    def __init__(self, timer: "Timer", key: str):
        """
        This initializes the context manager with a reference to the timer object via key.

        Args:
            timer (Timer): Name of the timer.
            key (str): For identification of the key.
        """
        self.timer = timer
        self.key = key

    def __enter__(self):
        """
        This method is called when entering the context managed by the with statement. 
        It starts the timer for the specified key.
        """
        self.timer.tick(self.key)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        This method is called when exiting the context managed by the with statement. 
        It stops the timer for the specified key.
        """
        self.timer.tock(self.key)
