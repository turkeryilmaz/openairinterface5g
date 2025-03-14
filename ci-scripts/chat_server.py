#
# For the usage of this script, refer to ../doc/episys/README_RUN_CHAT.md
#

import time
import socket
import threading
import tkinter as tk
from tkinter import scrolledtext, simpledialog

class ChatServer:
    """Server object to provide multicast service between ChatClient(s)."""
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = simpledialog.askstring(title="Server", prompt="Enter Server IP address:", initialvalue=host)
        self.port = port
        self.clients = {}  # Use a dictionary to store client sockets and usernames
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()

        self.root = tk.Tk()
        self.root.title("Chat Server")

        self.shutdown_event = threading.Event()
        self.root.bind("<Control-c>", lambda event: self.stop_server())

        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        threading.Thread(target=self.accept_connections).start()
        self.root.mainloop()

    def log(self, message):
        """Display (log) message in log_text."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def accept_connections(self):
        """Accepting connection from client(s)."""
        self.log(f"Server listening on {self.host}:{self.port}")
        th_list = []
        while True:
            try:
                self.server_socket.settimeout(5)
                client_socket, client_address = self.server_socket.accept()
                self.log(f"Accepted connection from {client_address}")
                self.clients[client_socket] = client_address  # Store client address with socket
                th = threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True)
                th.start()
                th_list.append(th)
            except Exception as e:
                if self.shutdown_event.is_set():
                    break
                time.sleep(2)
        for th in th_list:
            th.join()

    def handle_client(self, client_socket):
        """Main handler function to forward message or file to other clients(s). Based on header field value, it handles each type."""
        while not self.shutdown_event.is_set():
            try:
                # First, receive the message header (type and size)
                header = b''
                while True:  # Loop to receive the complete header
                    chunk = client_socket.recv(1)
                    if not chunk:
                        raise Exception("Client disconnected.")
                    header += chunk
                    if header.endswith(b'\n'):  # Check for the header delimiter
                        break
                header = header.decode('utf-8')[:-1]  # Remove the trailing '\n'
                msg_type, msg_size = header.split('|', maxsplit=1)

                if msg_type == "text":
                    msg_size = int(msg_size)
                    message = client_socket.recv(msg_size).decode('utf-8')
                    self.broadcast(f"{message}", client_socket)
                elif msg_type == "file":
                    filename, file_size = msg_size.split('|')
                    file_size = int(file_size)
                    # Receive the file data from the client

                    file_data = b''
                    bytes_received = 0
                    while bytes_received < file_size:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            break
                        file_data += chunk
                        bytes_received += len(chunk)
                    # Broadcast the file to other clients
                    self.broadcast_file(file_data, filename, file_size, client_socket)

            except Exception as e:
                self.log(f"Error handling client: {e}")
                username = self.clients.pop(client_socket)
                peername = client_socket.getpeername()
                client_socket.close()
                self.broadcast(f"{username} has left the chat!")
                self.log(f"Connection closed from {peername}")
                if len(self.clients) == 0:
                    break

    def broadcast(self, message, sender_socket=None):
        """Broadcasting message."""
        for client in self.clients:
            if client != sender_socket:
                try:
                    # Send text message header (type and size)
                    header = f"text|{len(message)}\n".encode('utf-8')
                    client.sendall(header)  # Use sendall() to ensure complete header is sent
                    client.sendall(message.encode('utf-8'))
                except Exception as e:
                    self.log(f"Error broadcasting message: {e}")


    def broadcast_file(self, file_data, filename, file_size, sender_socket):
        """Broadcasting file."""
        sender_address = self.clients[sender_socket]
        for client in self.clients:
            if client != sender_socket:
                try:
                    # Send file header (type, filename, and file size)
                    header = f"file|{filename}|{file_size}\n".encode('utf-8')
                    client.sendall(header)
                    client.sendall(file_data)
                    self.log(f"File from {sender_address} sent to {self.clients[client]}")
                except Exception as e:
                    self.log(f"Error broadcasting file: {e}")

    def stop_server(self):
        """Stops the server gracefully."""
        try:
            self.shutdown_event.set()
            self.log("Stopping server...")
            # Close all client sockets
            for client in list(self.clients.keys()):
                client.shutdown(socket.SHUT_RDWR)
                client.close()
            self.clients.clear()
            # Close the server socket
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            self.log("Server stopped.")
            self.root.destroy()  # Close the GUI window
        except Exception as e:
            self.log(f"Error stopping server: {e}")


if __name__ == "__main__":
    server = ChatServer()