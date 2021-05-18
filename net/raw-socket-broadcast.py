import socket
import binascii

iface = "lo"

src_mac = "00:00:00:00:00:00"
dst_mac = "ff:ff:ff:ff:ff:ff"


def enc_mac(mac_str: str) -> bytes:
    e = b"".join(binascii.unhexlify(x) for x in mac_str.split(":"))
    assert len(e) == 6
    return e


payload = b"yoyo"
# checksum = "\x1a\x2b\x3c\x4d"
ethertype = b"\x08\x01"

packet = enc_mac(dst_mac) + enc_mac(src_mac) + ethertype + payload

s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.IPPROTO_RAW)
s.sendto(packet, (iface, 0))
