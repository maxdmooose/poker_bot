import subprocess
import json

BOT_A = "bots/maximus_bot/CFR.py"
BOT_B = "bots/maximus_bot/CFR4.py"
HANDS = "25"
ROUNDS = 30

BOT_A_NAME = "CFR"
BOT_B_NAME = "bot2"

total_A = 0
total_B = 0

for i in range(ROUNDS):
    cmd = [
        "python3",
        "sandbox/match.py",
        BOT_A,
        BOT_B,
        "--hands",
        HANDS,
        "--json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        data = json.loads(result.stdout)

        delta_A = data["chip_delta"][BOT_A_NAME]
        delta_B = data["chip_delta"][BOT_B_NAME]

        total_A += delta_A
        total_B += delta_B

        print(
            f"Round {i+1}: "
            f"{BOT_A_NAME}={delta_A:+}, {BOT_B_NAME}={delta_B:+} | "
            f"Cumulative: {BOT_A_NAME}={total_A:+}, {BOT_B_NAME}={total_B:+}"
        )

    except Exception as e:
        print(f"Error on round {i+1}")
        print("Raw output:", result.stdout)
        print("Exception:", e)

print("\nFinal cumulative chip deltas:")
print(f"{BOT_A_NAME}: {total_A:+}")
print(f"{BOT_B_NAME}: {total_B:+}")
