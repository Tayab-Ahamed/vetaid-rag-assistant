# VetAid | Veterinary First-Aid RAG Assistant

VetAid is a Retrieval-Augmented Generation project that provides calm, step-by-step first-aid guidance for pet emergencies. It uses a local veterinary knowledge base, semantic retrieval with ChromaDB, and Groq LLaMA 3.1 for grounded responses.

## What is included

- A premium Streamlit interface with dark/light theme transition
- Chat-style response area with cleaner information hierarchy
- Conversation memory for follow-up questions in the same case
- Inline answer citations plus retrieved evidence blocks
- Multi-file built-in veterinary dataset loading from `data/`
- Optional upload area for adding extra reference files on top of the built-in library
- Species-aware retrieval focus with `Auto`, `Dog`, `Cat`, and `Other`
- Persistent ChromaDB indexing scoped to the active dataset set

## Key project improvements

- Follow-up chat memory so the same pet case can continue naturally
- Inline citations and expandable retrieved evidence for answer transparency
- Curated official-source dataset expansion for stronger emergency coverage
- Local chat auto-save, restore, and transcript download for demos and review
- Urgency labeling and follow-up prompts for a more complete chatbot experience

## Project structure

```text
vet_rag_project/
|-- app.py
|-- rag_pipeline.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- vet_data.txt
|   |-- critical_emergencies.txt
|   `-- common_urgent_symptoms.txt
|   `-- official_vet_emergency_reference.txt
|-- chroma_db/
|-- logs/
`-- uploads/
```

## Knowledge base

The app no longer depends on one dataset file only.

Built-in files:

- `data/vet_data.txt`: core veterinary first-aid reference
- `data/critical_emergencies.txt`: bloat/GDV, urinary blockage, shock, anaphylaxis, trauma transport
- `data/common_urgent_symptoms.txt`: vomiting red flags, breathing distress, eye emergencies, bite wounds, seizure escalation
- `data/official_vet_emergency_reference.txt`: curated official-source supplement based on Merck Veterinary Manual, ASPCA Animal Poison Control, VCA, and Red Cross guidance

The app loads every supported `.txt` or `.pdf` file in the `data/` folder automatically.

## Upload area

The manual upload area is optional.

Use it when you want to add:

- a clinic-specific emergency protocol
- a custom PDF handout
- another veterinary reference document for a demo

It is not required for normal use or for project submission. Uploaded files are added on top of the built-in library, not used as a replacement.

## Pet focus selector

The `Dog / Cat / Other` style selector was improved into:

- `Auto`
- `Dog`
- `Cat`
- `Other`

`Auto` is the default and is enough for most questions. The species choice is only helpful when retrieval should be nudged for species-specific emergencies, such as cat urinary blockage or dog bloat.

## Running the app

Create and activate your virtual environment, install dependencies, set `GROQ_API_KEY` in `.env`, then run:

```bash
streamlit run app.py
```

First run builds embeddings and the Chroma vector store for the active dataset set. Later runs reuse the matching persisted store.

## Demo notes

- Use `Reload knowledge library` after changing the dataset so the vector store reflects new files.
- Use `New chat` to clear the active case without changing the underlying knowledge base.
- Use `Download chat transcript` to keep a markdown record of the conversation and supporting references.

## Notes

- The app is for emergency first-aid guidance only.
- It is not a substitute for direct veterinary care.
- If a pet is collapsing, choking, unable to breathe, having repeated seizures, or unable to urinate, escalate to a veterinarian immediately.
