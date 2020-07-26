"""Question: Design a chat server.

Requirements:
- Multiple users connected
- Conversation order must be consistent for all users
- Messages have timestamps

Qestion:
- Does the message timestamp have to match the users exact timestamp?
- Should messages be ordered by the user timestamp or the server timestamp?
    - If server timestamp: single source of time truth, order is deterministic dependent on server
    - If user timestamp: on a slow connection, a user might send a message but
        it might appear a ways back in the history.
"""
import time
from queue import Queue, Empty
from threading import Thread
from typing import List


class Message:
    def __init__(self, client_id: str, content: str, timestamp: float) -> None:
        self.client_id = client_id
        self.content = content
        self.timestamp = timestamp

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class ClientController:
    def __init__(self, client_id: str) -> None:
        self._message_queue = Queue()

    def add_message(self, message: Message) -> None:
        self._message_queue.put(message)

    def get_messages(self) -> List[Message]:
        messages = []
        while True:
            try:
                # Bad loop
                messages.append(self._message_queue.get_nowait())
            except Empty:
                break
        return messages


class MessageDispatcher:
    def __init__(self) -> None:
        self._client_controllers = []

    def add_client_controller(self, client_controller: ClientController) -> None:
        self._client_controllers.append(client_controller)

    def dispatch_message(self, message: Message) -> None:
        """Dispatches a new message to all clients.
        """
        for client_controller in self._client_controllers:
            client_controller.add_message(message)


class Controller:
    """Queues messages, runs a thread to dispatch messages.
    """

    def __init__(self, message_dispatcher):
        """
        TODO: think about maxsize of queue
        """
        self._message_dispatcher = message_dispatcher
        self._queue = Queue()
        self._is_running = True
        self._thread = Thread(target=self.run_dispatch)
        self._thread.start()

    def stop(self):
        self._is_running = False

    def receive_message(self, client_id: str, content: str):
        """Receive a message, mark timestamp, add to a queue

        Queue ensures the order is determined by when the message
        was received in this function
        """
        message = Message(client_id, content, time.time())
        self._queue.put(message)

    def run_dispatch(self):
        while self._is_running:
            try:
                message = self._queue.get(timeout=0.05)
            except Empty:
                continue
            self._message_dispatcher.dispatch_message(message)


if __name__ == "__main__":
    client_controller1 = ClientController("1")
    client_controller2 = ClientController("2")
    client_controller3 = ClientController("3")
    message_dispatcher = MessageDispatcher()
    message_dispatcher.add_client_controller(client_controller1)
    message_dispatcher.add_client_controller(client_controller2)
    message_dispatcher.add_client_controller(client_controller3)
    controller = Controller(message_dispatcher)

    for i in range(10):
        time.sleep(0.5)
        controller.receive_message(str(i % 3), f"message {i}")

        if i % 3 == 0:
            print(client_controller1.get_messages())
            print(client_controller2.get_messages())
            print(client_controller3.get_messages())

    time.sleep(0.1)
    controller.stop()
