"""Static file server with HTTP range support, for the url driver tests.

Run as ``python range_server.py <folder>``: it prints the port on the first line, then serves
until killed. It runs in its own process because GDAL (through rasterio) keeps Python's GIL
while it waits for an HTTP answer, so a server thread in the test process would never reply.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    """Serves files, answering a single ``Range: bytes=a-b`` with 206 Partial Content."""

    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return None
        size = os.path.getsize(path)
        handle = open(path, "rb")  # noqa: SIM115 - closed by the base class
        wanted = self.headers.get("Range", "")
        if wanted.startswith("bytes="):
            first, _, last = wanted[len("bytes=") :].split(",")[0].partition("-")
            if first == "":
                start, end = max(0, size - int(last)), size - 1
            else:
                start, end = int(first), min(int(last), size - 1) if last else size - 1
            if start >= size:
                handle.close()
                self.send_error(416)
                return None
            handle.seek(start)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            length = end - start + 1
        else:
            self.send_response(200)
            length = size
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self._remaining = length
        return handle

    def copyfile(self, source, outputfile) -> None:
        remaining = self._remaining
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=sys.argv[1]))
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
