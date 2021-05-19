import socket

with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_UDP) as server_sock:
    # server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    server_sock.bind(("", 0))
    while True:
        data, (client_ip, client_port) = server_sock.recvfrom(2 << 15)
        # if client_ip == "10.70.70.65":
        print(client_ip, client_port, data)
