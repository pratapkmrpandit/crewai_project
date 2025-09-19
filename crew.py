import yaml
from crewai import Agent, Task, Crew, Process
from helper import load_env

# Load Gemini key from .env
GEMINI_API_KEY = load_env()

def load_yaml(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_agents(agent_yaml: str):
    data = load_yaml(agent_yaml)
    agents = {}
    for a in data["agents"]:
        agents[a["name"]] = Agent(
            role=a["role"],
            goal=a["goal"],
            backstory=a["backstory"],
            verbose=a.get("verbose", False),
            llm="gemini/gemini-2.0-flash"   # ✅ correct LiteLLM syntax
        )
    return agents


def load_tasks(task_yaml: str, agents: dict, city: str = "", famous_things: str = ""):
    data = load_yaml(task_yaml)
    tasks = []
    for t in data["tasks"]:
        description = (
            t["description"]
            .replace("{{city}}", city)
            .replace("{{famous_things}}", famous_things)
        )
        tasks.append(Task(
            description=description,
            expected_output=t["expected_output"],
            agent=agents[t["agent"]]
        ))
    return tasks

def main():
    # Load agents
    agents = load_agents("config/agents.yaml")

    # First: run city task via a temporary Crew
    city_task = Task(
        description="Choose a random city in India. Only return the city name.",
        expected_output="A single city name in India.",
        agent=agents["city_agent"]
    )

    city_crew = Crew(
        agents=[agents["city_agent"]],
        tasks=[city_task],
        process=Process.sequential,
        verbose=True
    )

    city_result = city_crew.kickoff()
    city = city_result.raw.strip()
    print(f"Gemini chose city: {city}")

    # Load famous + explanation tasks dynamically
    tasks = load_tasks("config/tasks.yaml", agents, city, "the two famous things")
    tasks = [tasks[1], tasks[2]]  # skip city_task since we already executed it

    # Now run the full crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    print("\n=== Final Output ===")
    print(result)

if __name__ == "__main__":
    main()
