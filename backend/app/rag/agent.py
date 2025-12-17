from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from .retriever import PgVectorRetriever, create_retrieval_tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from ..mlflow_logger import start_run, end_run
from .prompts import system_prompt, PROMPT_VERSION
import time
import json
import os
import tempfile


# Approximate pricing for gpt-4o-mini (USD per 1K tokens).
# Adjust according to OpenAI pricing changes.
INPUT_COST_PER_1K = 0.00015
OUTPUT_COST_PER_1K = 0.00060


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
                    "prompt_version": PROMPT_VERSION,
                },
            )
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=query)]}, config=config
            )

            # Extract the final answer from the result
            # LangGraph returns a dict with "messages" key containing the conversation
            messages = []
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

            # Try to get accurate token usage from LangChain message metadata
            input_tokens = 0
            output_tokens = 0
            for msg in messages or []:
                usage = getattr(msg, "usage_metadata", None)
                if isinstance(usage, dict):
                    input_tokens += int(usage.get("input_tokens", 0) or 0)
                    output_tokens += int(usage.get("output_tokens", 0) or 0)

            # Fallback: very rough estimate based on word counts if no usage metadata
            if input_tokens == 0:
                input_tokens = len(str(query).split())
            if output_tokens == 0:
                output_tokens = len(str(answer).split())

            estimated_cost_usd = (
                (input_tokens / 1000.0) * INPUT_COST_PER_1K
                + (output_tokens / 1000.0) * OUTPUT_COST_PER_1K
            )

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
            # Tool / retrieval statistics
            retrieval_count = getattr(self.retriever, "call_count", 0)
            distances = getattr(self.retriever, "distances", []) or []
            avg_chunk_distance = sum(distances) / \
                len(distances) if distances else 0.0

            # Very rough token proxy for answer length (still log separately)
            answer_length_tokens = len(str(answer).split())

            metrics = {
                "latency": latency,
                "retrieval_count": float(retrieval_count),
                "avg_chunk_distance": float(avg_chunk_distance),
                "answer_length_tokens": float(answer_length_tokens),
                "input_tokens": float(input_tokens),
                "output_tokens": float(output_tokens),
                "estimated_cost_usd": float(estimated_cost_usd),
            }
            artifacts = {"retrieval_trace": trace_path}
            end_run(metrics=metrics, artifacts=artifacts)

            return answer

        except Exception as e:
            error_msg = f"Error in agentic RAG: {str(e)}"
            print(f"[Agentic RAG] Error: {error_msg}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error while processing your query: {error_msg}"
