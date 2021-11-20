import logging
from time import perf_counter
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()

        time = (end - start) / 60

        info = (
            f"Execution time of {func.__name__} function is "
            + str(round(time, 3))
            + "min"
        )

        logging.info(info)
        print(info)
        return result

    return wrapper
