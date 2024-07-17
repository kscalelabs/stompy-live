"""Helper module for connecting to Twitch."""

import os
import queue
import re
import socket
import sys
import time

server = "irc.chat.twitch.tv"
port = 6667
nickname = "kscaletest"
token = os.environ["STOMPYLIVE_TOKEN"]
channel = "#kscaletest"

message_queue = queue.Queue()


def parse_message(raw_message: str) -> str | None:
    # This regex pattern matches the message part of a PRIVMSG
    pattern = r"^:.+!.+@.+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.+)$"
    match = re.match(pattern, raw_message)
    if match:
        return match.group(1)
    return None


def init() -> None:
    sock = socket.socket()

    sock.connect((server, port))
    sock.send(f"PASS {token}\n".encode())
    sock.send(f"NICK {nickname}\n".encode())
    sock.send(f"JOIN {channel}\n".encode())

    last_ping = time.time()

    try:
        while True:
            if time.time() - last_ping > 240:
                sock.send(b"PING\n")
                last_ping = time.time()
            resp = sock.recv(2048).decode("utf-8")

            if resp.startswith("PING"):
                sock.send(b"PONG\n")

            elif len(resp) > 0:
                message = parse_message(resp)
                if message:
                    message_queue.put(message)

    except KeyboardInterrupt:
        sock.close()
        sys.exit()
