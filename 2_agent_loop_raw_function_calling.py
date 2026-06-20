from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 3
MODEL = "qwen3:1.7b"

# --- Tools(LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> str:
    """Lookup the price of a product in the catalog."""
    print(f"  >>>Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers : bronze, silver, gold
    """
    print(
        f"  >>>Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Lookup the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name example 'laptop', 'headphones', 'keyboard'",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers : bronze, silver, gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The originalprice of the product",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply. Available tiers : bronze, silver, gold",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# summary
# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ----Agent Loop----


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    # llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool"
                "and a discount tool. \n\n"
                "STRICT Rules- You must follow these strictly:\n"
                "1. Never guess or assume any product price"
                "You must call get_product_price tool first to get the real price of a product.\n"
                "2. Only call apply_discount AFTER you have received"
                "a price from get_product_price. Pass the exact price"
                "returned by get_product_price - do NOT pass a made-up number \n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply_discount tool \n"
                "4. if the user does not  specify a discount tier, "
                "ask the which tier to use -do NOT assume one "
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()

        response = ollama_chat_traced(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer

        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content} ")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in available tools.")

        # Difference 7:Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)
        print(f"    [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Error: Max iterations without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "What is the price of the laptop after applying a gold discount?"
    )
