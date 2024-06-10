import warnings
import functools


def RotationAssignmentWarning(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", Warning)  # turn off filter
        warnings.warn(
            "This object should generally not have a transformation assignment via {}.".format(
                func.__name__
            ),
            category=Warning,
            stacklevel=2,
        )
        warnings.simplefilter("default", Warning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def RelativeAssignmentWarning(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", Warning)  # turn off filter
        warnings.warn(
            "This object should generally not have a parent/child assignment via {}.".format(
                func.__name__
            ),
            category=Warning,
            stacklevel=2,
        )
        warnings.simplefilter("default", Warning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def TranslationAssignmentWarning(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", Warning)  # turn off filter
        warnings.warn(
            "This object should generally not have a translation assignment via {}.".format(
                func.__name__
            ),
            category=Warning,
            stacklevel=2,
        )
        warnings.simplefilter("default", Warning)  # reset filter
        return new_func(*args, **kwargs)


def WorldAssignmentWarning(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", Warning)  # turn off filter
        warnings.warn(
            "This object should generally not have a to-world matrix via {}.".format(
                func.__name__
            ),
            category=Warning,
            stacklevel=2,
        )
        warnings.simplefilter("default", Warning)  # reset filter
        return new_func(*args, **kwargs)
