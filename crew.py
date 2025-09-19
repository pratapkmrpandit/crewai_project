import yaml
from helper import load_env
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen, router
from crewai_tools import SerperDevTool


def load_yaml(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


class CityInfoFlow(Flow):
    def __init__(self):
        super().__init__()
        keys = load_env()
        self.search_tool = SerperDevTool(api_key=keys["serper_api_key"])
        self.agents = self._load_agents("config/agents.yaml")

    def _load_agents(self, agent_yaml: str):
        data = load_yaml(agent_yaml)
        agents = {}
        for a in data["agents"]:
            tools = []
            if a["name"] == "famous_agent" or a["name"] == "explanation_agent":
                tools = [self.search_tool]  # attach SerperDevTool
            agents[a["name"]] = Agent(
                role=a["role"],
                goal=a["goal"],
                backstory=a["backstory"],
                verbose=a.get("verbose", False),
                llm="gemini/gemini-2.0-flash",
                tools=tools
            )
        return agents

    def _load_tasks(self, task_yaml: str, city: str = "", famous_things: str = ""):
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
                agent=self.agents[t["agent"]]
            ))
        return tasks


    @start()
    def pick_city(self):
        """Step 1: Choose a random Indian city."""
        city_task = Task(
            description="Choose a random city in India. Only return the city name.",
            expected_output="A single city name in India.",
            agent=self.agents["city_agent"]
        )
        crew = Crew(
            agents=[self.agents["city_agent"]],
            tasks=[city_task],
            process=Process.sequential,
            verbose=True
        )
        result = crew.kickoff()
        city = result.raw.strip()
        print(f"\nGemini chose city: {city}")
        return city

    @listen(pick_city)
    def find_famous(self, city: str):
        """Step 2: Find two famous things in that city using Serper."""
        tasks = self._load_tasks("config/tasks.yaml", city, "the two famous things")
        famous_task = tasks[1]  # famous_task is the 2nd in YAML
        crew = Crew(
            agents=[self.agents["famous_agent"]],
            tasks=[famous_task],
            process=Process.sequential,
            verbose=True
        )
        result = crew.kickoff()
        famous = result.raw.strip()
        print(f"\nFamous things in {city}: {famous}")
        return {"city": city, "famous_things": famous}

    @listen(find_famous)
    def explain_famous(self, state: dict):
        """Step 3: Generate 5-line explanations for each famous thing."""
        city = state["city"]
        famous_things = state["famous_things"]
        tasks = self._load_tasks("config/tasks.yaml", city, famous_things)
        explanation_task = tasks[2]  # explanation_task is the 3rd in YAML
        crew = Crew(
            agents=[self.agents["explanation_agent"]],
            tasks=[explanation_task],
            process=Process.sequential,
            verbose=True
        )
        result = crew.kickoff()
        print("\n📝 Explanations:")
        print(result)
        return result


if __name__ == "__main__":
    flow = CityInfoFlow()
    output = flow.kickoff()
    print("\nFinal Output\n ")
    print(output)
