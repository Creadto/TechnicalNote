from http.server import HTTPServer


class HttpServer:
    def __init__(self, ip: str, port: int, listener, handler):
        self.NewMessage = None
        self.listener = listener
        self.ip = ip
        self.port = port
        self.handler = handler
        self.serve_start()

    def serve_start(self):
        httpd = HTTPServer((self.ip, self.port), self.handler)
        httpd.serve_forever()
