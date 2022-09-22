import socket
import sys
import struct

SEND_BUF_SIZE = 256

RECV_BUF_SIZE = 256

Communication_Count:int = 0

receive_count:int = 0


def start_tcp_server(ip, port):
    # create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)

    # bind port
    print("starting listen on ip %s, port %s" % server_address)
    sock.bind(server_address)

    # get the old receive and send buffer size
    s_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    s_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print("socket send buffer size[old] is %d" % s_send_buffer_size)
    print("socket receive buffer size[old] is %d" % s_recv_buffer_size)

    # set a new buffer size
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)

    # get the new buffer size
    s_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    s_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print("socket send buffer size[new] is %d" % s_send_buffer_size)
    print("socket receive buffer size[new] is %d" % s_recv_buffer_size)

    # start listening, allow only one connection
    try:
        sock.listen(1)
    except socket.error:
        print("fail to listen on port %s" % e)
        sys.exit(1)
    while True:
        print("waiting for connection")
        client, addr = sock.accept()
        print("having a connection")
        break
    msg = 'welcome to tcp server' + "\r\n"
    receive_count = 0
    receive_count += 1
    while True:
        print("\r\n")
        msg = client.recv(16384)
        msg_de = msg.decode('utf-8')
        print("recv len is : [%d]" % len(msg_de))
        print("###############################")
        print(msg_de)
        print("###############################")

        if msg_de == 'disconnect': break

        msg = ("hello, client, i got your msg %d times, now i will send back to you " % receive_count)
        client.send(msg.encode('utf-8'))
        receive_count += 1
        print("send len is : [%d]" % len(msg))

    print("finish test, close connect")
    client.close()
    sock.close()
    print(" close client connect ")


if __name__ == '__main__':
    start_tcp_server('10.10.103.127', 29500)
