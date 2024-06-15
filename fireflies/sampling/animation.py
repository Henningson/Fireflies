import torch
import fireflies.sampling.base as base
import random


class AnimationSampler(base.Sampler):
    def __init__(
        self,
        min_integer_train: int,
        max_integer_train: int,
        min_integer_eval: int,
        max_integer_eval: int,
        eval_step_size: int = 1,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> None:
        """
        Will generate samples from the integer interval given by [min_integer, ..., max_integer) similar to how range() is defined in python.
        Assuming we have a set of train and eval objs that were loaded in the Mesh class using load_train_objs() and load_eval_objs().
        """
        super(AnimationSampler, self).__init__(min_integer_train, max_integer_train, eval_step_size, device)
        self._min_integer_train = min_integer_train
        self._max_integer_train = max_integer_train
        self._min_integer_eval = min_integer_eval
        self._max_integer_eval = max_integer_eval
        self._current_step = min_integer_eval

    def sample_eval(self) -> int:
        sample = self._current_step
        self._current_step += self._eval_step_size

        if self._current_step > self._max_integer_eval:
            self._current_step = self._min_integer_eval

        return sample

    def sample_train(self) -> int:
        return random.randint(self._min_integer_train, self._max_integer_train - 1)
    
    def set_train_interval(self, min_integer_train: int, max_integer_train: int) -> None:
        self._min_integer_train = min_integer_train
        self._max_integer_train = max_integer_train

    def set_eval_interval(self, min_integer_eval: int, max_integer_eval: int) -> None:
        self._min_integer_eval = min_integer_eval
        self._max_integer_eval = max_integer_eval