
# Lore - Natural Language Learning Assistant

## Description

Lore is an AI-powered learning assistant service that uses the Mistral language model to provide natural language responses to student questions. It is designed to enhance the learning experience by providing contextual explanations while strictly avoiding code generation.

## Dependencies and Virtual Environment

[Install Poetry](https://python-poetry.org/) for creating/managing virtual environment.

Then install dependencies and create the shell.

```bash
poetry install
poetry shell
```

Required packages:
- poetry
- mistralai
- valkey
- pydantic

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
Create a `.env` file with the following variables:
```
MISTRAL_API_KEY=your_api_key
VALKEY_URL=your_valkey_url
REDIS_URL=your_redis_url
```

## Usage
The service listens for questions on the Valkey `student_question` channel and responds with natural language explanations. All responses are filtered to prevent code generation.
