from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class TraceState(TypedDict, total=False):
    trace: Annotated[list[str], operator.add]


class ComplimentState(TraceState, total=False):
    name: str
    result: str


def compliment_node(state: ComplimentState) -> dict[str, Any]:
    name = str(state.get("name", "Learner"))
    return {
        "result": f"{name}, you're doing an amazing job learning LangGraph.",
        "trace": ["compliment: generated one-node response"],
    }


compliment_builder = StateGraph(ComplimentState)
compliment_builder.add_node("compliment", compliment_node)
compliment_builder.add_edge(START, "compliment")
compliment_builder.add_edge("compliment", END)
COMPLIMENT_GRAPH = compliment_builder.compile()


class SequenceState(TraceState, total=False):
    name: str
    age: int
    skills: list[str]
    result: str


def personalize(state: SequenceState) -> dict[str, Any]:
    return {
        "result": f"{state.get('name', 'Learner')}, welcome to the system!",
        "trace": ["personalize: greeted the user"],
    }


def describe_age(state: SequenceState) -> dict[str, Any]:
    result = state.get("result", "")
    return {
        "result": f"{result} You are {state.get('age', 0)} years old.",
        "trace": ["age_description: appended age"],
    }


def describe_skills(state: SequenceState) -> dict[str, Any]:
    skills = ", ".join(state.get("skills", [])) or "curiosity"
    return {
        "result": f"{state.get('result', '')} You have skills in: {skills}.",
        "trace": ["skills_description: appended skills"],
    }


sequence_builder = StateGraph(SequenceState)
sequence_builder.add_node("personalize", personalize)
sequence_builder.add_node("age_description", describe_age)
sequence_builder.add_node("skills_description", describe_skills)
sequence_builder.add_edge(START, "personalize")
sequence_builder.add_edge("personalize", "age_description")
sequence_builder.add_edge("age_description", "skills_description")
sequence_builder.add_edge("skills_description", END)
SEQUENCE_GRAPH = sequence_builder.compile()


class MultiInputState(TraceState, total=False):
    name: str
    values: list[int]
    operation: str
    result: str
    value: int


def process_values(state: MultiInputState) -> dict[str, Any]:
    values = [int(value) for value in state.get("values", [])]
    operation = state.get("operation", "+")
    if operation in {"*", "product", "multiply"}:
        value = 1
        for item in values:
            value *= item
        label = "product"
    else:
        value = sum(values)
        label = "sum"
    return {
        "value": value,
        "result": f"Hi {state.get('name', 'Learner')}, your {label} is {value}.",
        "trace": [f"processor: reduced {len(values)} values with {label}"],
    }


multi_builder = StateGraph(MultiInputState)
multi_builder.add_node("processor", process_values)
multi_builder.add_edge(START, "processor")
multi_builder.add_edge("processor", END)
MULTI_INPUT_GRAPH = multi_builder.compile()


class RouterState(TraceState, total=False):
    number1: int
    number2: int
    operation: str
    result: int


def route_operation(state: RouterState) -> str:
    return "addition" if state.get("operation") == "+" else "subtraction"


def add_numbers(state: RouterState) -> dict[str, Any]:
    result = int(state.get("number1", 0)) + int(state.get("number2", 0))
    return {"result": result, "trace": [f"add_node: computed {result}"]}


def subtract_numbers(state: RouterState) -> dict[str, Any]:
    result = int(state.get("number1", 0)) - int(state.get("number2", 0))
    return {"result": result, "trace": [f"subtract_node: computed {result}"]}


router_builder = StateGraph(RouterState)
router_builder.add_node("router", lambda _state: {"trace": ["router: inspected operation"]})
router_builder.add_node("add_node", add_numbers)
router_builder.add_node("subtract_node", subtract_numbers)
router_builder.add_edge(START, "router")
router_builder.add_conditional_edges(
    "router", route_operation, {"addition": "add_node", "subtraction": "subtract_node"}
)
router_builder.add_edge("add_node", END)
router_builder.add_edge("subtract_node", END)
ROUTER_GRAPH = router_builder.compile()


class DualRouterState(TraceState, total=False):
    number1: int
    number2: int
    operation: str
    number3: int
    number4: int
    operation2: str
    result: int
    result2: int


def route_first(state: DualRouterState) -> str:
    return "addition" if state.get("operation") == "+" else "subtraction"


def route_second(state: DualRouterState) -> str:
    return "addition" if state.get("operation2") == "+" else "subtraction"


def first_add(state: DualRouterState) -> dict[str, Any]:
    value = int(state.get("number1", 0)) + int(state.get("number2", 0))
    return {"result": value, "trace": [f"phase_one.add: {value}"]}


def first_subtract(state: DualRouterState) -> dict[str, Any]:
    value = int(state.get("number1", 0)) - int(state.get("number2", 0))
    return {"result": value, "trace": [f"phase_one.subtract: {value}"]}


def second_add(state: DualRouterState) -> dict[str, Any]:
    value = int(state.get("number3", 0)) + int(state.get("number4", 0))
    return {"result2": value, "trace": [f"phase_two.add: {value}"]}


def second_subtract(state: DualRouterState) -> dict[str, Any]:
    value = int(state.get("number3", 0)) - int(state.get("number4", 0))
    return {"result2": value, "trace": [f"phase_two.subtract: {value}"]}


dual_builder = StateGraph(DualRouterState)
dual_builder.add_node("router_one", lambda _state: {"trace": ["router_one: chose phase one"]})
dual_builder.add_node("add_one", first_add)
dual_builder.add_node("subtract_one", first_subtract)
dual_builder.add_node("router_two", lambda _state: {"trace": ["router_two: chose phase two"]})
dual_builder.add_node("add_two", second_add)
dual_builder.add_node("subtract_two", second_subtract)
dual_builder.add_edge(START, "router_one")
dual_builder.add_conditional_edges(
    "router_one", route_first, {"addition": "add_one", "subtraction": "subtract_one"}
)
dual_builder.add_edge("add_one", "router_two")
dual_builder.add_edge("subtract_one", "router_two")
dual_builder.add_conditional_edges(
    "router_two", route_second, {"addition": "add_two", "subtraction": "subtract_two"}
)
dual_builder.add_edge("add_two", END)
dual_builder.add_edge("subtract_two", END)
DUAL_ROUTER_GRAPH = dual_builder.compile()


class LoopState(TraceState, total=False):
    name: str
    numbers: list[int]
    counter: int
    limit: int
    greeting: str


def loop_greeting(state: LoopState) -> dict[str, Any]:
    return {
        "greeting": f"Hi there, {state.get('name', 'Learner')}",
        "trace": ["greeting: initialized checkpointed loop"],
    }


def loop_number(state: LoopState) -> dict[str, Any]:
    counter = int(state.get("counter", 0)) + 1
    numbers = list(state.get("numbers", []))
    number = (counter * 7 + 3) % 11
    numbers.append(number)
    return {
        "counter": counter,
        "numbers": numbers,
        "trace": [f"random: iteration {counter} produced {number}"],
    }


def continue_loop(state: LoopState) -> str:
    return "loop" if int(state.get("counter", 0)) < int(state.get("limit", 5)) else "exit"


loop_builder = StateGraph(LoopState)
loop_builder.add_node("greeting", loop_greeting)
loop_builder.add_node("random", loop_number)
loop_builder.add_edge(START, "greeting")
loop_builder.add_edge("greeting", "random")
loop_builder.add_conditional_edges("random", continue_loop, {"loop": "random", "exit": END})
LOOP_MEMORY = MemorySaver()
LOOP_GRAPH = loop_builder.compile(checkpointer=LOOP_MEMORY)


class GameState(TraceState, total=False):
    player_name: str
    target_number: int
    guesses: list[int]
    attempts: int
    hint: str
    lower_bound: int
    upper_bound: int


def setup_game(state: GameState) -> dict[str, Any]:
    player = str(state.get("player_name", "Player"))
    target = int(state.get("target_number") or (sum(map(ord, player)) % 20 + 1))
    return {
        "target_number": max(1, min(20, target)),
        "guesses": [],
        "attempts": 0,
        "hint": "Game started. Guess a number between 1 and 20.",
        "lower_bound": 1,
        "upper_bound": 20,
        "trace": [f"setup: started game for {player}"],
    }


def make_guess(state: GameState) -> dict[str, Any]:
    low = int(state.get("lower_bound", 1))
    high = int(state.get("upper_bound", 20))
    guess = (low + high) // 2
    guesses = [*state.get("guesses", []), guess]
    attempts = int(state.get("attempts", 0)) + 1
    return {"guesses": guesses, "attempts": attempts, "trace": [f"guess: attempt {attempts} chose {guess}"]}


def give_hint(state: GameState) -> dict[str, Any]:
    guess = state.get("guesses", [0])[-1]
    target = int(state.get("target_number", 1))
    if guess < target:
        hint = f"{guess} is too low. Try higher."
        update = {"lower_bound": guess + 1}
    elif guess > target:
        hint = f"{guess} is too high. Try lower."
        update = {"upper_bound": guess - 1}
    else:
        hint = f"Correct! Found {target} in {state.get('attempts', 0)} attempts."
        update = {}
    return {**update, "hint": hint, "trace": [f"hint: {hint}"]}


def continue_game(state: GameState) -> str:
    solved = state.get("guesses", [None])[-1] == state.get("target_number")
    return "end" if solved or int(state.get("attempts", 0)) >= 7 else "continue"


game_builder = StateGraph(GameState)
game_builder.add_node("setup", setup_game)
game_builder.add_node("guess", make_guess)
game_builder.add_node("hint", give_hint)
game_builder.add_edge(START, "setup")
game_builder.add_edge("setup", "guess")
game_builder.add_edge("guess", "hint")
game_builder.add_conditional_edges("hint", continue_game, {"continue": "guess", "end": END})
GAME_GRAPH = game_builder.compile()


WORKFLOW_CATALOG = [
    {
        "id": "compliment",
        "title": "Single node",
        "concept": "Minimal graph",
        "description": "A single node reads and updates typed state.",
        "nodes": ["compliment"],
        "edges": [["START", "compliment"], ["compliment", "END"]],
        "sample": {"name": "Bob"},
    },
    {
        "id": "sequence",
        "title": "Sequential profile",
        "concept": "Stateful sequence",
        "description": "Three nodes enrich one shared state in a fixed order.",
        "nodes": ["personalize", "age_description", "skills_description"],
        "edges": [["START", "personalize"], ["personalize", "age_description"], ["age_description", "skills_description"], ["skills_description", "END"]],
        "sample": {"name": "Linda", "age": 31, "skills": ["Python", "Machine Learning", "LangGraph"]},
    },
    {
        "id": "multiple-inputs",
        "title": "Multiple inputs",
        "concept": "State reduction",
        "description": "One processor reduces a list with sum or product.",
        "nodes": ["processor"],
        "edges": [["START", "processor"], ["processor", "END"]],
        "sample": {"name": "Jack", "values": [1, 2, 3, 4], "operation": "*"},
    },
    {
        "id": "conditional-router",
        "title": "Conditional router",
        "concept": "Branching",
        "description": "A router selects addition or subtraction from state.",
        "nodes": ["router", "add_node", "subtract_node"],
        "edges": [["START", "router"], ["router", "add_node | subtract_node"], ["operation", "END"]],
        "sample": {"number1": 10, "operation": "-", "number2": 5},
    },
    {
        "id": "dual-router",
        "title": "Two-phase router",
        "concept": "Chained branches",
        "description": "Two independent conditional phases run in sequence.",
        "nodes": ["router_one", "operation_one", "router_two", "operation_two"],
        "edges": [["START", "router_one"], ["router_one", "operation_one"], ["operation_one", "router_two"], ["router_two", "operation_two"], ["operation_two", "END"]],
        "sample": {"number1": 10, "operation": "-", "number2": 5, "number3": 7, "operation2": "+", "number4": 2},
    },
    {
        "id": "checkpoint-loop",
        "title": "Checkpointed loop",
        "concept": "Cycles + memory",
        "description": "A cyclic graph records state under a reusable thread id.",
        "nodes": ["greeting", "random", "MemorySaver"],
        "edges": [["START", "greeting"], ["greeting", "random"], ["random", "random | END"]],
        "sample": {"name": "Vaibhav", "counter": 2, "limit": 5, "thread_id": "demo-thread"},
    },
    {
        "id": "guessing-game",
        "title": "Guessing game",
        "concept": "Bounded agent loop",
        "description": "Guess and hint nodes tighten bounds until success or seven attempts.",
        "nodes": ["setup", "guess", "hint"],
        "edges": [["START", "setup"], ["setup", "guess"], ["guess", "hint"], ["hint", "guess | END"]],
        "sample": {"player_name": "Student", "target_number": 17},
    },
]


def run_workflow(workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if workflow_id == "compliment":
        return dict(COMPLIMENT_GRAPH.invoke({**payload, "trace": []}))
    if workflow_id == "sequence":
        return dict(SEQUENCE_GRAPH.invoke({**payload, "trace": []}))
    if workflow_id == "multiple-inputs":
        return dict(MULTI_INPUT_GRAPH.invoke({**payload, "trace": []}))
    if workflow_id == "conditional-router":
        if payload.get("operation") not in {"+", "-"}:
            raise ValueError("operation must be '+' or '-'")
        return dict(ROUTER_GRAPH.invoke({**payload, "trace": []}))
    if workflow_id == "dual-router":
        if payload.get("operation") not in {"+", "-"} or payload.get("operation2") not in {"+", "-"}:
            raise ValueError("operation and operation2 must be '+' or '-'")
        return dict(DUAL_ROUTER_GRAPH.invoke({**payload, "trace": []}))
    if workflow_id == "checkpoint-loop":
        thread_id = str(payload.get("thread_id", "default-thread"))
        state = {
            "name": payload.get("name", "Learner"),
            "numbers": payload.get("numbers", []),
            "counter": int(payload.get("counter", 0)),
            "limit": max(1, min(12, int(payload.get("limit", 5)))),
            "trace": [],
        }
        return dict(LOOP_GRAPH.invoke(state, config={"configurable": {"thread_id": thread_id}}))
    if workflow_id == "guessing-game":
        return dict(GAME_GRAPH.invoke({**payload, "trace": []}))
    raise KeyError(f"Unknown workflow: {workflow_id}")

