from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import agentops
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List

from tavily import TavilyClient
from scrapegraph_py import Client

import os
import json

load_dotenv()


agentops.init(
    api_key=os.environ['AGENTOPS_API_KEY'],
    skip_auto_end_session=True # the default is Flase so the agent can stop the session anytime
)

output_dir = './ai-agent-output'
os.makedirs(output_dir, exist_ok=True)

basic_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ['GROQ_API_KEY'],
    temperature=0
)

search_client = TavilyClient(api_key=os.environ['TVLY_API_KEY'])
scrape_client = Client(api_key=os.environ['SCRAPEGRAPH_API_KEY'])


no_keywords = 10

about_company = "Rankyx is a company that provides AI solutions to help websites refine their search and recommendation systems."

company_context = StringKnowledgeSource(
    content=about_company
)

"""# Setup Agent

## 1. Search Queries Generator
"""

class SuggestedSearchQueries(BaseModel):
    queries: List[str] = Field(..., title='Suggested search queries to be passed to the search engine',
                               min_length=1, max_length=no_keywords)


recommender_agent = Agent(
    role="Search Queries Recommendation Agent",
    goal='\n'.join(['To provide a list of suggested search querie to be passed to the search engine.',
            'The queries must be varied and looking for a specific items']),
    backstory='The agent is designed to help in looking for products by providing a list of suggested search queries to be passed to the search engine based on the context provided',
    llm=basic_llm,
    verbose=True
)

recommender_agent_task = Task(
    description='\n'.join([
        'Rankyx is looking to but {product_name} at the best price (value for price strategy)',
        'The company targets any of those websites to buy from: {website_list}',
        'The company wants to reach all available products on the internet to be compared later in another stage.',
        'The stores must sell the product in {country_name}',
        'Generate at maximum {no_keywords} queries.',
        'The search keywords must be in {language} language'
        'The search query must reach an ecommerece webpage for product, and not just listing page.'
    ]),

    expected_output='A JSON object containing a list of suggested search queries.',
    output_json=SuggestedSearchQueries,

    output_file=os.path.join(output_dir, 'recommender_agent_output.json'),

    agent=recommender_agent,
)

"""## 2. Search Agent"""

class SignleSearchResult(BaseModel):
    title: str
    url: str = Field(..., title='URL of the search result')
    score: float
    search_query: str

class AllSearchResults(BaseModel):
    result: List[SignleSearchResult]

@tool
def search_engine_tool(query: str):
  # context to the crewai
    """The function is used to find current infoo about any query related pages using the search engine"""
    return search_client.search(query)

search_engine_agent = Agent(
    role="Search Engine Agent",
    goal='To search for products based on the suggested search query',
    backstory='The agent is designed to help in searching for products based on the suggested search query',
    llm=basic_llm,
    verbose=True,
    tools=[search_engine_tool]
)

search_engine_task = Task(
    description='\n'.join([
        'The task is to search for products based on the suggested search queries',
        'You have to collect results from multiple search queries',
        'The search query must reach an ecommerece webpage for product',
        'Ignore any susbisious links or not an ecommerce website link',
        'Ignore any search results with confidence score less than ({score_th})',
        'The search results will be used to compare prices of products from different websites.',
    ]),
    expected_output='A JSON object containing a list of search results.',
    output_json=AllSearchResults,
    output_file=os.path.join(output_dir, 'search_engine_output(2).json'),
    agent=search_engine_agent
)

"""## 3. Details of Product Webpage"""

class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str

class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title='URL of the product page') # The ... to ensure that this is a required field
    product_title: str = Field(..., title='Title of the product')
    product_image_url: str = Field(..., title="The url of the product image")
    product_url: str = Field(..., title="The url of the product")
    product_current_price: float = Field(..., title="The current price of the product")
    product_original_price: float = Field(title="The original price of the product before discount. Set to None if no discount", default=None)
    product_discount_percentage: float = Field(title="The discount percentage of the product. Set to None if no discount", default=None)

    product_specs: List[ProductSpec] = Field(..., title="The specifications of the product. Focus on the most important specs to compare.", min_items=1, max_items=5)

    agent_recommendation_rank: int = Field(..., title="The rank of the product to be considered in the final procurement report. (out of 5, Higher is Better) in the recommendation list ordering from the best to the worst")
    agent_recommendation_notes: List[str]  = Field(..., title="A set of notes why would you recommend or not recommend this product to the company, compared to other products.")


class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]


@tool
def web_scrpaing_tool(page_url: str):
  """An AI tool to help an agent scrape a web page

    Example:
    web_scraping_tool(
      page_url='',
    )
  """

  details = scrape_client.smartscraper(
      website_url=page_url,
      user_prompt='Extract ```json\n' + SingleExtractedProduct.schema_json() + '```\n From the webpage'
  )
  return {
      'page_url': page_url,
      'details': details
  }

scraping_agent = Agent(
    role='web scraping agent',
    goal="To extract details from any website",
    backstory='The agent is designed to help in extracting details from any website. These details will be used to decide which best products to buy.',
    llm=basic_llm,
    tools=[web_scrpaing_tool],
    verbose=True

)

scraping_task = Task(
    description="\n".join([
        "The task is to extract product details from any ecommerce store page url.",
        "The task has to collect results from multiple pages urls.",
        "Collect the best {top_recommendations_no} products from the search results.",
    ]),
    expected_output="A JSON object containing products details",
    output_json=AllExtractedProducts,
    output_file=os.path.join(output_dir, 'scraping_output(3).json'),
    agent=scraping_agent
)

"""## 4. Report Designer Agent


"""

procurement_report_agent = Agent(
    role="Procurement Report Author Agent",
    goal="To generate a professional HTML page for the procurement report",
    backstory="The agent is designed to assist in generating a professional HTML page for the procurement report after looking into a list of products.",
    llm=basic_llm,
    verbose=True
)

procurement_report_task = Task(
    description="\n".join([
        "The task is to generate a professional HTML page for the procurement report.",
        "You have to use Bootstrap CSS framework for a better UI.",
        "Use the provided context about the company to make a specialized report.",
        "The report will include the search results and prices of products from different websites.",
        "The report should be structured with the following sections:",
        "1. Executive Summary: A brief overview of the procurement process and key findings.",
        "2. Introduction: An introduction to the purpose and scope of the report.",
        "3. Methodology: A description of the methods used to gather and compare prices.",
        "4. Findings: Detailed comparison of prices from different websites, including tables and charts.",
        "5. Analysis: An analysis of the findings, highlighting any significant trends or observations.",
        "6. Recommendations: Suggestions for procurement based on the analysis.",
        "7. Conclusion: A summary of the report and final thoughts.",
        "8. Appendices: Any additional information, such as raw data or supplementary materials.",
    ]),

    expected_output="A professional HTML page for the procurement report.",
    output_file=os.path.join(output_dir, "step_4_procurement_report.html"),
    agent=procurement_report_agent,
)
