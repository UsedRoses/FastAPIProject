import csv
from typing import Callable, List

from starlette.responses import StreamingResponse


class Echo:
    """An object that implements just the write method of a file-like interface."""
    def write(self, value):
        return value


def generate_streaming_data(fetch_data_fn: Callable, fields: List[str], process_data_fn: Callable):
    """
    A generator to stream data from any data source in batches.

    :param fetch_data_fn: Function to fetch data from the data source.
    :param fields: List of field names to include in the output.
    :param process_data_fn: A function to process the data into the desired output format (e.g. CSV rows).
    :param batch_size: Number of records to fetch per query.
    """
    while True:
        # Fetch data in batches
        batch, is_done = fetch_data_fn()

        if not batch:
            break

        # Process and yield the data
        for record in batch:
            yield process_data_fn(record, fields)

        if is_done:
            break


def stream_file_response(fetch_data_fn: Callable, fields: List[str], process_data_fn: Callable, file_name: str,
                         content_type: str = 'text/csv'):
    """
    Return a streaming HTTP response with data formatted as CSV or other file types.

    :param fetch_data_fn: 获取数据源的函数.
    :param fields: 要从数据源中获取那些字段.
    :param process_data_fn: 如何从数据源取出数据的函数.
    :param file_name: 返回的文件名.
    :param content_type: The content type (CSV or other).
    :return: StreamingHttpResponse with data.
    """

    def generate_csv_data():
        """Wrapper function to handle data and write it as CSV."""
        pseudo_buffer = Echo()
        writer = csv.writer(pseudo_buffer)
        # Write header row
        yield writer.writerow(fields)

        # Yield data rows
        for row in generate_streaming_data(fetch_data_fn, fields, process_data_fn):
            yield writer.writerow(row)

    response = StreamingResponse(generate_csv_data(), media_type="text/event-stream")
    response['Cache-Control'] = 'no-cache'
    return response
