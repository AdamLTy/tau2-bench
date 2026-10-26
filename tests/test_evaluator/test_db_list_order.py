"""DB grading compares order-free lists as multisets (issue #514)."""

from copy import deepcopy
from typing import Callable

import pytest

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.domains.retail.environment import (
    get_environment as get_retail_environment,
)
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

# Airline tasks whose gold book_reservation splits the payment across methods.
MULTI_PAYMENT_TASKS = ["14", "20", "23", "24", "25"]


def _task(task_id: str) -> Task:
    return next(task for task in get_tasks() if task.id == task_id)


def _run_gold(
    task: Task, edit_booking: Callable[[dict], dict] = lambda args: args
) -> tuple[list[Message], Environment]:
    """Execute the gold actions live, editing each book_reservation's arguments."""
    env = get_environment()
    messages: list[Message] = []
    for index, action in enumerate(task.evaluation_criteria.actions):
        arguments = deepcopy(action.arguments)
        if action.name == "book_reservation":
            arguments = edit_booking(arguments)
        call = ToolCall(
            id=str(index),
            name=action.name,
            arguments=arguments,
            requestor=action.requestor,
        )
        messages.append(AssistantMessage(role="assistant", tool_calls=[call]))
        response = env.get_response(call)
        assert not response.error, response.content
        messages.append(response)
    return messages, env


def _db_reward(task: Task, trajectory: list[Message]) -> float:
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
    )
    return reward_info.reward_breakdown["DB"]


def _reverse_payments(arguments: dict) -> dict:
    arguments["payment_methods"].reverse()
    return arguments


def _move_one_dollar(arguments: dict) -> dict:
    """Same cards and total, but $1 moves from the first method to the last."""
    payments = arguments["payment_methods"]
    payments[0]["amount"] -= 1
    payments[-1]["amount"] += 1
    return arguments


def test_gold_trajectory_passes_db_check():
    task = _task("14")
    trajectory, _ = _run_gold(task)
    assert _db_reward(task, trajectory) == 1.0


@pytest.mark.parametrize("task_id", MULTI_PAYMENT_TASKS)
def test_reordered_payment_methods_pass_db_check(task_id: str):
    task = _task(task_id)
    trajectory, _ = _run_gold(task, _reverse_payments)
    assert _db_reward(task, trajectory) == 1.0


def test_different_payment_split_still_fails_db_check():
    task = _task("14")
    trajectory, _ = _run_gold(task, _move_one_dollar)
    assert _db_reward(task, trajectory) == 0.0


def test_check_db_and_get_db_hash_agree_on_reordered_payments():
    _, gold = _run_gold(_task("14"))
    _, reordered = _run_gold(_task("14"), _reverse_payments)
    assert reordered.get_db_hash() == gold.get_db_hash()
    assert reordered.check_db(gold.tools.db)


def test_reservation_flight_order_still_matters():
    _, env = _run_gold(_task("14"))
    db = env.tools.db
    reservation_id = next(
        rid for rid, res in db.reservations.items() if len(res.flights) > 1
    )
    reordered = FlightDB.model_validate(db.model_dump())
    reordered.reservations[reservation_id].flights.reverse()
    assert reordered.get_hash() != db.get_hash()


def test_retail_payment_history_order_still_matters():
    db = get_retail_environment().tools.db
    order_id = "#W3220387"
    history = db.orders[order_id].payment_history
    assert len({entry.model_dump_json() for entry in history}) > 1
    reordered = type(db).model_validate(db.model_dump())
    reordered.orders[order_id].payment_history.reverse()
    assert reordered.get_hash() != db.get_hash()


def test_only_airline_declares_order_insensitive_lists():
    assert FlightDB.order_insensitive_lists() == (
        ("reservations", "*", "payment_history"),
    )
    assert type(get_retail_environment().tools.db).order_insensitive_lists() == ()
