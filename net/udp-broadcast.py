import socket

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_sock:
    client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    client_sock.sendto(b"yoyo", ("<broadcast>", 56789))

print("success")
