import socket
import struct
import selectors

import net


def main():
    iface = "enp6s0"
    # iface = "wlp3s0"
    with socket.socket(
        socket.AF_PACKET, socket.SOCK_RAW, socket.htons(net.ETH_P_IP)
    ) as server_socket:
        server_socket.bind((iface, 0))
        with selectors.DefaultSelector() as selector:
            selector.register(server_socket.fileno(), selectors.EVENT_READ)
            while True:
                ready = selector.select()
                if ready:
                    frame = server_socket.recv(net.ETH_FRAME_LEN)
                    header = frame[: net.ETH_HLEN]
                    dst, src, proto = struct.unpack("!6s6sH", header)
                    payload = frame[net.ETH_HLEN :]
                    dst = net.bytes_to_eui48(dst)
                    src = net.bytes_to_eui48(src)
                    if src == "d4:3b:04:21:b2:ef":
                        print(
                            f"dst: {dst}, "
                            f"src: {src}, "
                            f"type: {hex(proto)}, "
                            f"payload: {payload[:4]}..."
                        )
                    # print(payload)


if __name__ == "__main__":
    main()
