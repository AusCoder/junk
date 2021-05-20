import socket
import binascii
import struct

import net


def main():
    dst_addr = "ff:ff:ff:ff:ff:ff"
    interface = "wlp2s0"

    with socket.socket(socket.AF_PACKET, socket.SOCK_RAW) as client_socket:
        client_socket.bind((interface, 0))

        # Ethernet header
        src_mac = net.get_hardware_address(interface)
        dst_mac = net.eui48_to_bytes(dst_addr)
        eth_header = struct.pack("!6s6sH", dst_mac, src_mac, net.ETH_P_IP)

        # IP header
        source_ip = "0.0.0.0"
        dest_ip = "255.255.255.255"

        # ip header fields
        ip_ihl = 5
        ip_ver = 4
        ip_tos = 0
        ip_tot_len = 0  # kernel will fill the correct total length
        ip_id = 54321  # Id of this packet
        ip_frag_off = 0
        ip_ttl = 255
        ip_proto = socket.IPPROTO_UDP
        ip_check = 0  # kernel will fill the correct checksum
        ip_saddr = socket.inet_aton(
            source_ip
        )  # Spoof the source ip address if you want to
        ip_daddr = socket.inet_aton(dest_ip)

        ip_ihl_ver = (ip_ver << 4) + ip_ihl

        # the ! in the pack format string means network order
        ip_header = struct.pack(
            "!BBHHHBBH4s4s",
            ip_ihl_ver,
            ip_tos,
            ip_tot_len,
            ip_id,
            ip_frag_off,
            ip_ttl,
            ip_proto,
            ip_check,
            ip_saddr,
            ip_daddr,
        )

        user_data = b"yoyo"

        # UDP header
        udp_src_port = 56788
        udp_dst_port = 56789
        udp_len = 8 + len(user_data)
        udp_chk = 0

        udp_header = struct.pack("!HHHH", udp_src_port, udp_dst_port, udp_len, udp_chk)

        packet = eth_header + ip_header + udp_header + user_data
        # packet = eth_header + user_data
        client_socket.sendall(packet)

        # eth_frame = eth_header + b"yoyo"
        # client_socket.sendall(eth_frame)

        print("success")


if __name__ == "__main__":
    main()
