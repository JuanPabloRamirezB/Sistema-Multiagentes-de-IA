#!/bin/sh

# Start the Ollama server in the background
ollama serve &

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434 > /dev/null; do
    sleep 1
done
echo "Ollama server is now running."

# Check if the model exists and pull it if it doesn't
ollama list | grep -q 'llama3.2'
if [ $? -ne 0 ]; then
    echo "Model llama3.2 not found. Downloading..."
    ollama pull llama3.2
else
    echo "Model llama3.2 already exists. Skipping download."
fi

# Bring the background server process to the foreground
wait %1