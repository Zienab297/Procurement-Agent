# Procurement Agent

A smart, AI-powered procurement assistant that helps you find the best products for your company and automatically generates a detailed report. Built with [CrewAI](https://crewai.com) for multi-agent orchestration and [AgentOps](https://agentops.ai) for agent observability.

---

## Features

- **Multi-Agent Architecture** — Specialized AI agents collaborate to research, evaluate, and recommend products
- **Automated Report Generation** — Outputs a structured procurement report with findings and recommendations
- **OpenRouter LLM Integration** — Flexible LLM backend via [OpenRouter](https://openrouter.ai), giving access to many models
- **Flask Web Interface** — Simple, accessible UI for interacting with the agent crew
- **AgentOps Observability** — Full monitoring and tracing of agent sessions and LLM calls
- **Docker Support** — Containerized deployment via the included `Dockerfile`

---

## Architecture

The project uses **CrewAI** to define a crew of autonomous AI agents, each with a specific role in the procurement workflow. Agents collaborate sequentially, passing context and findings between tasks to arrive at a final recommendation and report.

```
main.py          ← Entry point;
ai_crew.py       ← Defines agents, tasks, and the CrewAI crew
ai-agent-output/ ← Stores generated procurement reports
.env             ← API keys and configuration
```

---

## Prerequisites

- Python 3.10–3.13
- Docker (optional, for containerized deployment)
- API keys for:
  - [OpenRouter](https://openrouter.ai) — for LLM access
  - [AgentOps](https://agentops.ai) — for agent monitoring

## 1. Clone the Repository

```bash
git clone https://github.com/Zienab297/Procurement-Agent.git
cd Procurement-Agent
```

## 2. Set Up Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
AGENTOPS_API_KEY=your_agentops_api_key
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
python main.py
```

---

## Docker Deployment

Build and run the container:

```bash
docker build -t procurement-agent .
docker run -p 5000:5000 --env-file .env procurement-agent
```

---

## Project Structure

```
Procurement-Agent/
├── .github/workflows/     # CI/CD pipeline configuration
├── ai-agent-output/       # Generated procurement reports
├── ai_crew.py             # CrewAI agents, tasks, and crew definition
├── main.py                # Flask app entry point
├── requirements.txt       # Python dependencies
├── DockerFile             # Docker build configuration
├── .env.example           # Environment variable template
├── agentops.log           # AgentOps session logs
└── README.md
```

---

## How It Works

1. **User Input** — You describe what product or category you need to procure via the web UI.
2. **Agent Research** — The CrewAI agents autonomously search, evaluate, and compare options using configured tools.
3. **Report Generation** — A structured report is generated and saved to the `ai-agent-output/` directory.
4. **Monitoring** — All agent actions, LLM calls, and costs are tracked via AgentOps.

---

## Tech Stack

| Component | Technology |
|---|---|
| Multi-Agent Framework | [CrewAI](https://github.com/crewAIInc/crewAI) |
| Web Framework | [Flask](https://flask.palletsprojects.com) |
| LLM Provider | [OpenRouter](https://openrouter.ai) |
| Agent Observability | [AgentOps](https://agentops.ai) |
| Language | Python 3.10+ |
| Containerization | Docker |

---

## Monitoring with AgentOps

This project integrates with [AgentOps](https://agentops.ai) to provide real-time visibility into agent behavior. Once configured with your `AGENTOPS_API_KEY`, you can:

- View session replays of agent runs
- Track LLM token usage and costs
- Debug agent failures and tool calls
- Monitor multi-agent interaction patterns

Logs are also stored locally in `agentops.log`.
