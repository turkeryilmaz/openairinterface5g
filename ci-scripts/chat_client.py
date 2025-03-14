#
# For the usage of this script, refer to ../doc/episys/README_RUN_CHAT.md
#

import os
import socket
import threading
import tkinter as tk
from tkinter import scrolledtext, simpledialog, filedialog, messagebox

class ChatClient:
    """Client Object to provide Communication with the other ChatClient(s) via ChatServer."""
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = simpledialog.askstring(title="Client", prompt="Enter Server IP address:", initialvalue=host)
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.root = tk.Tk()
        self.root.title("Chat Client")

        self.root.bind("<Control-c>", lambda event: self.disconnect_from_server())

        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.input_area = tk.Entry(input_frame)
        self.input_area.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_area.bind("<Return>", self.send_message)

        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)

        self.file_button = tk.Button(input_frame, text="File", command=self.send_file)
        self.file_button.pack(side=tk.LEFT)

        self.connect()
        self.root.mainloop()

    def connect(self):
        """Connecting to server."""
        try:
            self.client_socket.connect((self.host, self.port))
            self.username = simpledialog.askstring("Username", "Enter your username:")
            if not self.username:
                self.root.destroy()
                return
            threading.Thread(target=self.receive_messages, daemon=True).start()
        except Exception as e:
            self.chat_area.config(state=tk.NORMAL)
            self.chat_area.insert(tk.END, f"Failed to connect: {e}\n")
            self.chat_area.config(state=tk.DISABLED)

    def receive_messages(self):
        """Handles header first. Based on the header field value, it process recv."""
        while True:
            try:
                # First, receive the message header (type and size)
                header = b''
                while True:  # Loop to receive the complete header
                    chunk = self.client_socket.recv(1)
                    if not chunk:
                        raise ConnectionResetError("Client disconnected.")
                    header += chunk
                    if header.endswith(b'\n'):  # Check for the header delimiter
                        break
                header = header.decode('utf-8')[:-1]  # Remove the trailing '\n'
                msg_type, msg_size = header.split('|', maxsplit=1)

                if msg_type == "text":
                    msg_size = int(msg_size)
                    message = self.client_socket.recv(msg_size).decode('utf-8')
                    self.display_message(message)
                elif msg_type == "file":
                    filename, file_size = msg_size.split('|')
                    file_size = int(file_size)
                    # Ask the user if they want to receive the file
                    if messagebox.askyesno(
                            f"Incoming File: {filename}",
                            "Do you want to receive this file?"):
                        self.receive_file(file_size, filename)
                    else:
                        # Ignore the file
                        self.client_socket.recv(file_size)
                        self.display_message("Incoming file ignored.")
            except Exception as e:
                self.chat_area.config(state=tk.NORMAL)
                self.chat_area.insert(tk.END, f"Disconnected from server: {e}\n")
                self.chat_area.config(state=tk.DISABLED)
                break

    def send_message(self, event=None):
        """Sending message with header."""
        message = self.input_area.get()
        if message:
            try:
                # Send text message header (type and size)
                user_msg = f"{self.username}: {message}"
                disp_msg = f"You: {message}"
                header = f"text|{len(user_msg)}\n".encode('utf-8')
                self.client_socket.sendall(header)
                self.display_message(disp_msg, 'user')
                self.client_socket.sendall(user_msg.encode('utf-8'))
                self.input_area.delete(0, tk.END)
            except Exception as e:
                self.chat_area.config(state=tk.NORMAL)
                self.chat_area.insert(tk.END, f"Error sending message: {e}\n")
                self.chat_area.config(state=tk.DISABLED)

    def send_file(self):
        """Sending file with header."""
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                # Send file header (type, filename, and file size)
                file_size = os.path.getsize(file_path)
                filename = os.path.basename(file_path)  # Extract filename
                header = f"file|{filename}|{file_size}\n".encode('utf-8')
                self.client_socket.sendall(header)

                # Send the file data
                with open(file_path, 'rb') as f:
                    while True:
                        bytes_read = f.read(4096)
                        if not bytes_read:
                            break
                        self.client_socket.sendall(bytes_read)
                self.display_message(f"File '{os.path.basename(file_path)}' sent successfully.")
            except Exception as e:
                self.display_message(f"Error sending file: {e}")

    def receive_file(self, file_size, filename):
        """Receiving file. It requires file name to save."""
        try:
            save_path = filedialog.asksaveasfilename(initialfile=filename)
            if not save_path:
                return

            with open(save_path, 'wb') as f:
                bytes_received = 0
                while bytes_received < file_size:
                    bytes_read = self.client_socket.recv(4096)
                    if not bytes_read:
                        break
                    f.write(bytes_read)
                    bytes_received += len(bytes_read)
            self.display_message(f"File saved to '{os.path.basename(save_path)}'.")
        except Exception as e:
            self.display_message(f"Error receiving file: {e}")

    def display_message(self, message, tag=''):
        """Display message in chat_area."""
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message + "\n", tag)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def disconnect_from_server(self):
        """Handles the Ctrl+C event to gracefully exit the application."""
        try:
            self.display_message("Disconnecting from server...")
            # Close the client socket
            if self.client_socket:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
                self.client_socket = None
            self.display_message("Disconnected from server.")
            self.root.destroy()  # Close the GUI window
        except Exception as e:
            self.display_message(f"Error disconnecting from server: {e}")


if __name__ == "__main__":
    client = ChatClient()