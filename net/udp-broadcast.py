import socket

iface = "wlp3s0"
with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW) as client_sock:
    # client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    client_sock.bind((iface, 0))
    client_sock.send(b"magic\n\nyo bro")

print("success")
