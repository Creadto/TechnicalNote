from processor.http_server import HttpProvider


def main(ip, port):
    provider = HttpProvider(ip=ip, port=port)
    provider.serve()


if __name__ == '__main__':
    main('192.168.219.108', 3000)
