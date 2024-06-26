import os
import socket
import queue
import re

server = 'irc.chat.twitch.tv'
port = 6667
nickname = 'kscaletest'
token = os.environ["STOMPYLIVE_TOKEN"]
channel = '#kscaletest'

message_queue = queue.Queue()

def parse_message(raw_message):
    # This regex pattern matches the message part of a PRIVMSG
    pattern = r'^:.+!.+@.+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.+)$'
    match = re.match(pattern, raw_message)
    if match:
        return match.group(1)
    return None

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
                message = parse_message(resp)
                if message:
                    message_queue.put(message)

    except KeyboardInterrupt:
        sock.close()
        exit()
