from twitch.client import message_queue, init
from threading import Thread
import time, queue

def main():
    print("Starting Twitch IRC client")
    
    # Initializes Twitch IRC thread
    irc_thread = Thread(target=init)
    irc_thread.daemon = True  # This allows the thread to exit when the main program does
    irc_thread.start()

    print("Starting main program loop")

    while True:
        try:
            message = message_queue.get(block=False)
            print(message)
        except queue.Empty:
            time.sleep(1)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
