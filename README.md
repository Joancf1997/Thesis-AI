# ARDI - AI Agent for News Media Data Exploration

> **Master's Thesis Project**  
> Free University of Bozen-Bolzano (UNIBZ)  
> Master's Degree in Data Science  
> Developing an AI Agent for Data Exploration in the News Media Industry

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Context](#project-context)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Agent Workflow](#agent-workflow)
- [Data Sources](#data-sources)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**ARDI** (Analytical Reasoning for Data Insights) is an intelligent AI agent designed to assist news editors, analysts, and content strategists in understanding and interpreting reader behavior with digital news content. The system leverages Large Language Models (LLMs) to provide natural language interfaces for complex data analytics tasks in the news media industry.

This project demonstrates how AI agents can bridge the gap between complex data analytics and non-technical users, enabling news organizations to make data-driven editorial decisions through conversational interfaces.

---

## 🌍 Project Context

This thesis addresses a critical challenge in the news media industry: **making audience analytics accessible to editorial teams**. Traditional analytics tools require technical expertise, creating a barrier between data insights and editorial decision-making.

### Business Problem

News organizations collect vast amounts of reader interaction data but struggle to:
- Understand what topics attract specific user segments
- Analyze when and how readers consume news
- Identify which articles or topics drive engagement
- Make these insights accessible to non-technical editorial staff

### Solution

ARDI provides a conversational AI interface that:
- Translates natural language questions into analytical workflows
- Executes multi-step data analysis tasks autonomously
- Generates human-readable insights from complex data
- Operates within the context of a German regional news organization

---

## 🏗️ Architecture

ARDI implements an **agentic AI architecture** with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (FastAPI REST API)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      ARDI Agent Core                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Planner    │→ │   Executor   │→ │  Responder   │      │
│  │  (LangChain) │  │  (LangGraph) │  │    (LLM)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Analytical Tools Layer                    │
│  • User Segmentation    • News Topics    • Article Data     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Data Sources (Pickle/JSON)                │
│  • user_segments_viz.pkl  • news_topics.pkl                 │
│  • news_viz2.json                                            │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Task Planning Module** (`planner.py`)
   - Converts user questions into structured analytical plans
   - Validates task dependencies and tool availability
   - Uses LLM with structured output for plan generation

2. **Task Execution Module** (`executor.py`)
   - Executes analytical tasks in dependency order
   - Manages data flow between tasks
   - Supports dynamic plan updates based on intermediate results

3. **Response Generation Module** (`responder.py`)
   - Synthesizes analytical results into natural language
   - Provides context-aware explanations
   - Formats insights for editorial decision-making

4. **Workflow Orchestration** (`workflow.py`)
   - Implements state machine using LangGraph
   - Manages execution flow and error handling
   - Provides checkpointing for conversation continuity

---

## ✨ Key Features

### 🤖 Intelligent Query Planning
- Automatically decomposes complex questions into analytical steps
- Identifies required data sources and tools
- Validates plan feasibility before execution

### 🔄 Multi-Step Execution
- Executes tasks with dependency resolution
- Chains multiple analytical operations
- Supports dynamic replanning based on intermediate results

### 📊 Comprehensive Analytics Tools
- **15+ specialized analytical functions** covering:
  - User segment analysis (demographics, behavior, engagement)
  - Topic modeling and transitions
  - Temporal activity patterns
  - Regional consumption analysis
  - Article performance metrics

### 💬 Natural Language Interface
- Conversational query input
- Context-aware responses
- Journalistic tone adapted for newsroom environments

### 🗄️ Persistent Conversation Management
- Thread-based conversation history
- PostgreSQL database for state persistence
- Run and step tracking for audit trails

### 📈 Evaluation Framework
- Dataset-based evaluation system
- Tool usage accuracy metrics
- Performance tracking across queries

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.x** - Primary programming language
- **LangChain** - LLM orchestration framework
- **LangGraph** - State machine and workflow management
- **FastAPI** - REST API framework
- **PostgreSQL** - Relational database for persistence
- **SQLAlchemy** - ORM for database operations

### AI/ML Components
- **OpenAI GPT** - Large Language Model (configurable)
- **Structured Output** - Pydantic models for type-safe LLM responses
- **Prompt Engineering** - Context-specific system prompts

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Pickle** - Serialized data storage

### Development Tools
- **Uvicorn** - ASGI server
- **python-dotenv** - Environment configuration
- **Pydantic** - Data validation

---

## 📁 Project Structure

```
Thesis-AI/
├── API/                          # REST API layer
│   ├── api.py                   # Main API endpoints
│   └── SystemAPI.py             # System-level endpoints
│
├── Assistant/                    # Core agent implementation
│   ├── ARDI.py                  # Main agent class
│   ├── ARDIChat.py              # Chat interface wrapper
│   └── agent_core/              # Agent workflow components
│       ├── planner.py           # Task planning logic
│       ├── executor.py          # Task execution engine
│       ├── responder.py         # Response generation
│       └── workflow.py          # LangGraph workflow
│
├── config/                       # Configuration files
│   ├── settings.yaml            # LLM and system settings
│   └── prompts/                 # System prompts
│       ├── 0.business_context.txt
│       ├── 1.data_sources_context.txt
│       ├── 2.tools_planning.txt
│       ├── 2.plan_update.txt
│       ├── 3.direct_response.txt
│       └── 4.response_stage.txt
│
├── crud/                         # Database operations
│   ├── login.py
│   ├── message.py
│   ├── run.py
│   ├── step.py
│   ├── thread.py
│   ├── tool.py
│   └── users.py
│
├── db/                           # Database setup
│   ├── base.py                  # Base models
│   ├── create_db.py             # Database initialization
│   ├── insert_dataset.py        # Dataset insertion
│   └── session.py               # Session management
│
├── models/                       # SQLAlchemy models
│   ├── datasetEvaluation.py
│   ├── message.py
│   ├── run.py
│   ├── step.py
│   ├── thread.py
│   ├── toolCall.py
│   └── user.py
│
├── utils/                        # Utility modules
│   ├── tools.py                 # Analytical tools implementation
│   └── utils.py                 # Helper functions
│
├── data/                         # Data sources
│   ├── user_segments_viz.pkl    # User segmentation data
│   ├── news_topics.pkl          # Topic modeling data
│   └── news_viz2.json           # Raw article data
│
├── datasets/                     # Evaluation datasets
│   └── evaluation_dataset.json
│
├── logs/                         # Application logs
│   └── system.log
│
├── playground.ipynb              # Development notebook
├── ToolsDev.ipynb               # Tools development notebook
├── .env                         # Environment variables
└── README.md                    # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL 12 or higher
- OpenAI API key (or compatible LLM provider)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Joancf1997/Thesis-AI.git
cd Thesis-AI
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

*Note: If `requirements.txt` is not present, install the following packages:*
```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv \
            langchain langchain-openai langgraph pandas numpy pydantic
```

### Step 4: Set Up PostgreSQL Database
```bash
# Create database
createdb news_AI

# Or using psql
psql -U postgres
CREATE DATABASE news_AI;
\q
```

### Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```env
PROJECT_NAME=ARDI AI Assistant
API_V1_STR=/api/v1
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=news_AI
DATABASE_URL=postgresql://your_username:your_password@localhost/news_AI
OPENAI_API_KEY=your_openai_api_key
```

### Step 6: Initialize Database
```bash
python db/create_db.py
```

### Step 7: Load Evaluation Dataset (Optional)
```bash
python db/insert_dataset.py
```

---

## ⚙️ Configuration

### LLM Configuration
Edit `config/settings.yaml` to configure the language model:

```yaml
llm:
  provider: openai          # Options: openai, ollama
  model_name: gpt-4         # Model identifier
  temperature: 0            # 0-1, controls randomness
  max_tokens: 2048          # Maximum response length
```

### Prompt Engineering
System prompts are located in `config/prompts/`:
- `0.business_context.txt` - Business domain and role definition
- `1.data_sources_context.txt` - Available data sources description
- `2.tools_planning.txt` - Tool descriptions for planning
- `2.plan_update.txt` - Dynamic plan update instructions
- `3.direct_response.txt` - Direct response (no tools) template
- `4.response_stage.txt` - Final response generation template

---

## 💻 Usage

### Starting the API Server
```bash
uvicorn API.api:app --reload --root-path /API
```

#### 1. Create a User
```bash
curl -X POST "http://localhost:8000/users/create" \
  -H "Content-Type: application/json" \
  -d '{"username": "editor1", "password": "secure_password"}'
```

#### 2. Login
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "editor1", "password": "secure_password"}'
```

#### 3. Create a Conversation Thread
```bash
curl -X POST "http://localhost:8000/chat/newThread" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your-user-uuid",
    "name": "Segment Analysis Session"
  }'
```

#### 4. Ask a Question
```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which regions are most engaged for the political debates segment?",
    "thread_id": "your-thread-uuid"
  }'
```

### Example Questions

**User Segment Analysis:**
- "Describe the reading behavior of segment 5"
- "Which regions are most active for the political debates segment?"
- "What time of day do readers in the sports segment engage most?"

**Topic Analysis:**
- "What are the most common topic transitions for segment 3?"
- "Predict the next likely topic after 'Politik' for segment 7"

**Article Performance:**
- "Show me the top articles read between 8am and 10am by segment 2"
- "Which articles have the highest engagement in the morning?"

**Temporal Patterns:**
- "When are readers most active during the day for segment 4?"
- "Compare morning vs evening reading patterns"

---

## 🔌 API Endpoints

### Authentication
- `POST /auth/login` - User authentication
- `POST /users/create` - Create new user
- `GET /users` - List all users

### Chat Interface
- `POST /chat/newThread` - Create new conversation thread
- `POST /chat/ask` - Submit question to agent
- `GET /chat/history/{thread_id}` - Retrieve conversation history
- `PUT /chat/thread/{thread_id}/rename` - Rename thread
- `DELETE /chat/thread/{thread_id}` - Delete thread

### Thread Management
- `GET /threads?user_id={uuid}` - Get user's threads

### System Endpoints
- `GET /UserSegments` - List all user segments
- `GET /UserSegment/{id}` - Get segment details

### Evaluation
- `POST /dataset_evaluation` - Run evaluation on dataset
- `GET /dataset_evaluations` - Retrieve evaluation results

---

## 🔄 Agent Workflow

ARDI follows a multi-stage workflow implemented as a state machine:

### 1. Task Planning
```
User Question → LLM Planning → Structured Plan (JSON)
```
- Analyzes user intent
- Identifies required tools
- Creates dependency graph
- Outputs structured task list

### 2. Plan Validation
```
Structured Plan → Validation → [Valid] → Execution
                             → [Invalid] → Replan
```
- Checks task IDs uniqueness
- Validates tool existence
- Verifies dependency integrity
- Ensures proper argument types

### 3. Plan Execution
```
Task 1 → Task 2 → ... → Task N
  ↓        ↓              ↓
Output 1  Output 2    Output N
```
- Resolves dependencies
- Executes tools sequentially
- Manages data flow between tasks
- Supports dynamic replanning

### 4. Response Generation
```
Execution Outputs → LLM Synthesis → Natural Language Response
```
- Synthesizes results
- Generates insights
- Formats for editorial context
- Provides actionable recommendations

### State Machine Diagram
```
[START] → [task_planning] → [validate_plan] → {validation_router}
                                                      ↓
                                    ┌─────────────────┼─────────────────┐
                                    ↓                 ↓                 ↓
                            [direct_response]   [run_plan]      [task_planning]
                                    ↓                 ↓                 ↑
                                  [END]    [generate_response]         │
                                                      ↓                 │
                                                    [END]               │
                                                                        │
                                    (invalid plan) ─────────────────────┘
```

---

## 📊 Data Sources

### 1. User Segmentation Data (`user_segments_viz.pkl`)
Behavioral clusters of news readers including:
- Segment descriptions and titles
- User type distribution (frequent/non-frequent)
- Regional consumption patterns
- Engagement metrics (scroll depth, time, words per minute)
- Topic transition probabilities
- Representative articles

### 2. News Topics Data (`news_topics.pkl`)
Topic modeling results including:
- Topic titles and descriptions
- High/low representative documents
- Topic clusters

### 3. Raw Articles Data (`news_viz2.json`)
Article metadata including:
- Article IDs and titles
- Teaser text
- Publication dates
- Topic clusters
- Engagement metrics

---

## 📈 Evaluation

### Evaluation Framework
The system includes an evaluation framework to assess agent performance:

#### Metrics
- **Tool Selection Accuracy**: Ratio of correctly selected tools
- **Task Completion Rate**: Percentage of successfully completed queries
- **Response Quality**: Human evaluation of generated insights

#### Running Evaluation
```bash
curl -X POST "http://localhost:8000/dataset_evaluation" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your-user-uuid",
    "name": "Evaluation Run 1"
  }'
```

#### Viewing Results
```bash
curl -X GET "http://localhost:8000/dataset_evaluations"
```

### Evaluation Dataset
Located in `datasets/evaluation_dataset.json`, containing:
- User queries
- Expected tool selections
- Ground truth answers

---

## 🧪 Development

### Jupyter Notebooks
- `playground.ipynb` - Interactive development and testing
- `ToolsDev.ipynb` - Tool development and validation

### Adding New Tools
1. Implement function in `utils/tools.py`
2. Add to `TASK_FUNCS` dictionary
3. Update `config/prompts/2.tools_planning.txt` with tool description
4. Add argument type mapping in `executor.py` (if needed)

### Database Schema
The system tracks:
- **Users** - Authentication and ownership
- **Threads** - Conversation sessions
- **Messages** - User and assistant messages
- **Runs** - Agent execution instances
- **Steps** - Individual workflow stages
- **ToolCalls** - Tool invocations with inputs/outputs
- **DatasetEvaluations** - Evaluation results

---

## 🤝 Contributing

This is a thesis project, but contributions and suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

---

## 📝 License

This project is part of a Master's thesis at the Free University of Bozen-Bolzano.  
Please contact the author for licensing information.

---

## 👤 Author

**Jose Andres**  
Master's Student in Data Science  
Free University of Bozen-Bolzano (UNIBZ)

---

## 🙏 Acknowledgments

- Free University of Bozen-Bolzano for academic support
- Davisd Massimo for the guidance and support during the thesis
- News organization for providing anonymized data

---

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

---

**Last Updated**: January 2026
