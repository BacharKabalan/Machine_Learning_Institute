if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <path_to_private_key> <username> <host> <port>"
    exit 1
fi

# Assign arguments to variables
PRIVATE_KEY="$1"
USERNAME="$2"
HOST="$3"
PORT="$4"
scp -i "$PRIVATE_KEY" -P "$PORT" /home/bash1989/.ssh/id_ed25519.pub "$USERNAME"@"$HOST":~/.ssh/
scp -i "$PRIVATE_KEY" -P "$PORT" /home/bash1989/.ssh/id_ed25519 "$USERNAME"@"$HOST":~/.ssh/
scp -i "$PRIVATE_KEY" -P "$PORT" ~/founders_and_coders/founders_and_coders/setup/server_setup_script.sh "$USERNAME"@"$HOST":/workspace/
scp -i "$PRIVATE_KEY" -P "$PORT" ~/founders_and_coders/founders_and_coders/.gitignore "$USERNAME"@"$HOST":/workspace/Machine_Learning_Institute/