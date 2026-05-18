import copy
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_AGENT_DIR = "agent"

GENERATIONS = 100000
GAMES_PER_SIDE = 2

MUTATION_RATE = 0.35
MUTATION_SCALE = 0.08

ACCEPT_IF_SCORE_AT_LEAST = 3

ROOT = Path(__file__).resolve().parent

BASE_COPY = ROOT / "agent_base"
MUTANT_COPY = ROOT / "agent_mutant"

PYTHON_EXECUTABLE = sys.executable

print("\nUsing Python:")
print(PYTHON_EXECUTABLE)


PARAM_REGEX = re.compile(
    r"^([A-Z_]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
    re.MULTILINE,
)


def delete_dir(path):
    if path.exists():
        shutil.rmtree(path)

def copy_agent(src_name, dst_path):
    src_path = ROOT / src_name

    delete_dir(dst_path)

    shutil.copytree(
        src_path,
        dst_path,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".git",
            ".venv",
        ),
    )

def load_constants_text():
    constants_path = ROOT / BASE_AGENT_DIR / "constants.py"

    with open(constants_path, "r") as f:
        return f.read()

def save_constants_text(agent_dir, text):
    constants_path = agent_dir / "constants.py"

    with open(constants_path, "w") as f:
        f.write(text)

def parse_parameters(text):
    params = {}

    for match in PARAM_REGEX.finditer(text):
        name = match.group(1)
        value = float(match.group(2))
        params[name] = value

    return params
def mutate_parameters(params):

    new_params = copy.deepcopy(params)

    changed = []

    for key in new_params:

        if random.random() < MUTATION_RATE:

            old = new_params[key]

            # integer parameters 
            if (
                key.endswith("_END")
                or key == "AB_MAX_DEPTH"
                or key == "MAX_ROLLOUT_DEPTH"
            ):

                scale = random.uniform(0.8, 1.2)

                new_value = int(round(old * scale))

                if new_value < 1:
                    new_value = 1
            # float parameters
            else:

                scale = random.uniform(
                    1.0 - MUTATION_SCALE,
                    1.0 + MUTATION_SCALE,
                )

                new_value = old * scale

            new_params[key] = new_value

            changed.append((key, old, new_value))

    return new_params, changed


def apply_parameters(text, params):
    def replacer(match):
        name = match.group(1)

        if name not in params:
            return match.group(0)

        value = params[name]

        if isinstance(value, int):
            return f"{name} = {value}"

        return f"{name} = {value}"

    return PARAM_REGEX.sub(replacer, text)


def run_match(red_agent, blue_agent):

    cmd = [
        PYTHON_EXECUTABLE,
        "-m",
        "referee",
        red_agent,
        blue_agent,
    ]

    print("\nRUNNING:")
    print(" ".join(cmd))

    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    duration = time.time() - start

    print(f"\nMATCH TOOK {duration:.2f} SECONDS")

    stdout = result.stdout
    stderr = result.stderr

    # crash
    if result.returncode != 0:
        print("\nMATCH CRASHED")
        return "CRASH"

    # detect winner
    if "winner is RED" in stdout:
        return "RED"

    if "winner is BLUE" in stdout:
        return "BLUE"

    return "DRAW"

# initialise base parameters

base_text = load_constants_text()

base_params = parse_parameters(base_text)

print("\nLoaded parameters:")

for k, v in base_params.items():
    print(f"{k} = {v}")

# evolution loop

generation = 1

while generation <= GENERATIONS:

    print("\n============================================================")
    print(f"GENERATION {generation}")
    print("============================================================")


    copy_agent(BASE_AGENT_DIR, BASE_COPY)
    copy_agent(BASE_AGENT_DIR, MUTANT_COPY)


    mutant_params, changes = mutate_parameters(base_params)

    if not changes:
        print("\nNo mutations generated")
        generation += 1
        continue

    print("\nMutations:")

    for name, old, new in changes:
        print(f"\n{name}: {old} -> {new}")

    mutant_text = apply_parameters(base_text, mutant_params)

    save_constants_text(MUTANT_COPY, mutant_text)


    score = 0
    losses = 0

    # red mutant
    print("\nMutant as RED")

    for i in range(GAMES_PER_SIDE):

        winner = run_match("agent_mutant", "agent_base")

        if winner == "RED":
            print(f"\nGame {i + 1}: WIN")
            score += 1
        else:
            print(f"\nGame {i + 1}: LOSS")
            losses += 1

        # reject early
        if losses >= 2:
            print("\nStopping early (2 losses reached)")
            break

    # blue mutant
    if score < 3 and losses < 2:

        print("\nMutant as BLUE")

        for i in range(GAMES_PER_SIDE):

            winner = run_match("agent_base", "agent_mutant")

            if winner == "BLUE":
                print(f"\nGame {i + 1}: WIN")
                score += 1
            else:
                print(f"\nGame {i + 1}: LOSS")
                losses += 1

            # accept early
            if score >= 3:
                print("\nStopping early (3 wins reached)")
                break

            # reject early
            if losses >= 2:
                print("\nStopping early (2 losses reached)")
                break

    total_games = GAMES_PER_SIDE * 2

    print(f"\nScore: {score}/{total_games}")

    if score >= ACCEPT_IF_SCORE_AT_LEAST:

        print("\nMutant accepted")

        base_params = mutant_params
        base_text = mutant_text

        save_constants_text(ROOT / BASE_AGENT_DIR, mutant_text)

    else:
        print("\nMutant rejected")

    generation += 1