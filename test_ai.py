
from ai_engine import ask_ai

print("Testing Claude AI connection...")

try:
    question = "What is artificial intelligence?"

    answer, latency = ask_ai(question)

    print("\nQuestion:", question)
    print("\nClaude:", answer)
    print(f"\nResponse time: {latency:.2f} seconds")

    print("\nClaude connected successfully!")

except Exception as e:
    print("\nError:", e)
