from serve.http_server import HttpProvider


def main(ip, port):
    provider = HttpProvider(ip=ip, port=port)
    provider.serve()


if __name__ == '__main__':
    # import socket
    # my_ip = socket.gethostbyname(socket.gethostname())
    # print(my_ip)
    main("192.168.219.149", 3000)