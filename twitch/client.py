import os
import socket
import queue

server = 'irc.chat.twitch.tv'
port = 6667
nickname = 'kscaletest'
token = os.environ["STOMPYLIVE_TOKEN"]
channel = '#kscaletest'

message_queue = queue.Queue()

def init():
    sock = socket.socket()
    
    sock.connect((server, port))
    sock.send(f"PASS {token}\n".encode('utf-8'))
    sock.send(f"NICK {nickname}\n".encode('utf-8'))
    sock.send(f"JOIN {channel}\n".encode('utf-8'))
    
    try:
        while True:
            resp = sock.recv(2048).decode('utf-8')
        
            if resp.startswith('PING'):
                sock.send("PONG\n".encode('utf-8'))
            
            elif len(resp) > 0:
                message_queue.put(resp)

    except KeyboardInterrupt:
        sock.close()
        exit()
