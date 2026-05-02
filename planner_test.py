from planner import generate_plan
import json

while True:
    user_input = input("\nEnter your idea (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        break

    result = generate_plan(user_input)

    print("\n===== GENERATED PLAN =====")
    print(json.dumps(result, indent=2))