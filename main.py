from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from ai_crew import (recommender_agent, search_engine_agent, scraping_agent, procurement_report_agent,
                    recommender_agent_task, search_engine_task, scraping_task, procurement_report_task,
                    no_keywords, company_context)


"""## Running AI Crew"""

rankyx_crew = Crew(
    agents=[
        recommender_agent,
        search_engine_agent,
        scraping_agent,
        procurement_report_agent
        ],

    tasks=[
        recommender_agent_task,
        search_engine_task,
        scraping_task,
        procurement_report_task
        ],

    process=Process.sequential,
    knowledge_sources=[
        company_context
        ]
)

crew_results= rankyx_crew.kickoff(
    inputs={
        'product_name': 'coffee machine for the office',
        'website_list': ['www.amazon.eg', 'www.jumia.eg', 'www.noon.com?egypt-en'],
        'no_keywords': no_keywords,
        'country_name': 'Egypt',
        'language': 'Arabic',
        'score_th': 0.10,
        'top_recommendations_no': 5

        }
)