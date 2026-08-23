import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}] line [{line_number}]: {str(error)}"


class RecommenderException(Exception):
    """Wraps any exception with the originating file and line number."""

    def __init__(self, error_message: Exception, error_detail: sys) -> None:
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
