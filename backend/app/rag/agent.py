from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from .retriever import PgVectorRetriever, create_retrieval_tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from ..mlflow_logger import start_run, end_run
import time
import json
import os
import tempfile


class MedicalAgent:
    """
    An agentic RAG agent that can reason about queries, decompose complex questions,
    and iteratively retrieve information using tools.
    """

    def __init__(self, model_name="gpt-4o-mini", temperature=0, top_k=10):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.retriever = PgVectorRetriever(
            embeddings=self.embeddings, top_k=top_k)

        # Create the retrieval tool
        self.search_tool = create_retrieval_tool(self.retriever)
        self.tools = [self.search_tool]

        # Create the agent with custom system prompt for true agentic behavior
        system_prompt = """You are an expert medical guideline assistant with access to WHO medical documents.

                            You work autonomously to answer medical questions by reasoning step-by-step and deciding what actions to take.

                            You have access to this tool:
                            - search_medical_documents: Search WHO medical documents and guidelines stored in a vector database

                            Your decision-making process:
                            1. Analyze the question to understand what information is needed
                            2. For complex questions, break them into sub-questions that can be searched independently
                            3. Decide whether to search, and if so, what query terms to use
                            4. Evaluate search results - are they sufficient? Do you need to search again with different terms?
                            5. Synthesize information from multiple searches if needed
                            6. Provide a comprehensive answer based only on retrieved documents

                            Key principles:
                            - Think critically about whether a search is needed and what to search for
                            - For multi-part questions, search each part separately (e.g., "symptoms AND treatment" → search "symptoms" then "treatment")
                            - If initial results are insufficient, try alternative search terms or more specific queries
                            - Only answer based on information found in the documents
                            - If information is not found after multiple attempts, clearly state that
                            - Cite document sources when possible
                            - Reason autonomously - decide your own approach to solving the problem"""

        # Create agent using LangGraph (create_agent returns a CompiledStateGraph)
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=True
        )

    def run(self, query: str) -> str:
        """
        Run the agentic RAG flow:
        1. Agent reasons about the query
        2. Agent decides what tools to use
        3. Agent may make multiple retrieval calls
        4. Agent synthesizes the final answer
        """
        print(f"\n[Agentic RAG] Processing query: {query}")
        print("=" * 60)

        try:
            # Invoke the agent graph - it will handle reasoning, tool calls, and synthesis
            # The agent is a CompiledStateGraph, so we invoke it with the input
            # Allow up to 10 steps of reasoning
            config = {"recursion_limit": 10}

            # LangGraph expects messages in a specific format
            start_time = time.time()
            start_run(
                name="rag_inference",
                params={
                    "model": self.llm,
                    "temperature": self.llm.temperature,
                    "top_k": self.retriever.top_k,
                    "query": query,
                },
            )
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=query)]}, config=config
            )

            # Extract the final answer from the result
            # LangGraph returns a dict with "messages" key containing the conversation
            if isinstance(result, dict):
                messages = result.get("messages", [])
                if messages:
                    # Get the last message (should be the agent's final response)
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'content'):
                        answer = last_msg.content
                    elif isinstance(last_msg, dict):
                        answer = last_msg.get('content', '')
                    else:
                        answer = str(last_msg)
                else:
                    answer = result.get("output", str(result))
            else:
                answer = str(result)

            # Build and save a simple retrieval/answer trace as JSON
            trace = {
                "query": query,
                "answer": answer,
            }
            trace_path = os.path.join(
                tempfile.gettempdir(),
                f"rag_retrieval_trace_{int(time.time())}.json",
            )
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

            print("=" * 60)
            print(f"[Agentic RAG] Final Answer: {answer[:200]}...")

            # Log metrics and artifacts
            latency = time.time() - start_time
            metrics = {"latency": latency}
            artifacts = {"retrieval_trace": trace_path}
            end_run(metrics=metrics, artifacts=artifacts)

            return answer

        except Exception as e:
            error_msg = f"Error in agentic RAG: {str(e)}"
            print(f"[Agentic RAG] Error: {error_msg}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error while processing your query: {error_msg}"
