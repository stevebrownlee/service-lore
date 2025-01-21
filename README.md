
# Lore - Natural Language Learning Assistant

## Description

Lore is an AI-powered learning assistant service that uses the Mistral language model to provide natural language responses to student questions. It is designed to enhance the learning experience by providing contextual explanations while strictly avoiding code generation.

## Dependencies and Virtual Environment

1. [Install Poetry](https://python-poetry.org/) for creating/managing virtual environment.
2. Run `poetry self add poetry-plugin-dotenv@latest` to have Poetry support sourcing the `.env` file you need.

Then install dependencies and create the shell.

```bash
poetry install
poetry shell
```

## Mistral Model Overview

Lore utilizes the Mistral language model, a state-of-the-art Large Language Model (LLM) specifically fine-tuned for educational contexts. Key features:

- Optimized for natural language explanations
- Built-in safeguards against code generation
- Context-aware responses based on learning materials
- Maintains consistent educational tone

## System Architecture

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Valkey as Valkey Message Bus
    participant Lore
    participant Buffer

    Client->>+API: POST /helprequest {question}
    API->>API: Create HelpRequest record
    API->>Valkey: Publish to 'student_question'
    API-->>-Client: Return requestId

    Client->>+API: GET /answers/{requestId} (SSE)
    API->>API: Initialize EventSource stream

    Valkey->>+Lore: Receive question
    Note over Lore: Begin model inference

    loop Until response complete
        Lore->>Buffer: Add generated text chunk
        alt Buffer at window size
            Buffer-->>Lore: Wait for acks
        else Buffer has space
            Lore->>Valkey: Publish chunk to 'lore_response_{requestId}'
            Valkey->>API: Deliver chunk
            API->>Client: Stream chunk via SSE

            Client->>+API: POST /answers/{requestId}/ack
            API->>Valkey: Publish to 'lore_ack'
            Valkey->>Lore: Deliver ack
            Lore->>Buffer: Remove acknowledged chunk
            Note over Buffer: Space freed for next chunk
            API-->>-Client: Ack confirmed
        end
    end

    Lore->>Valkey: Publish final chunk (is_final=true)
    Valkey->>API: Deliver final chunk
    API->>Client: Stream final chunk
    Client->>API: Final acknowledgment
    Client->>Client: Close EventSource

    Note over Client,Lore: Stream complete
```

## Configuration
Create a `.env` file with the following variables, for you to test locally:

```
HUGGING_FACE_HUB_TOKEN=<<token>>
VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_PASSWORD=<<password>>
```
