import socket

iface = "lo"

s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.IPPROTO_RAW)
s.bind((iface, 0))
x = s.recvfrom(1024)
print(x)


# with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#     s.bind(("", 59478))
#     while True:
#         data, client_addr = s.recvfrom(1024)
#         print(f"{client_addr} - {data}")
#         s.sendto(b"magic\n\nyo bra", client_addr)
