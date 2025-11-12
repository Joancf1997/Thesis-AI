import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from models.toolCall import ToolCall  


def create_tool_call(
    db: Session,
    step_id: uuid.UUID,
    tool_name: str,
    input_data: dict,
    meta: dict | None = None,
) -> ToolCall:
    """
    Creates a new ToolCall row linked to a Step.
    Returns the ToolCall instance.
    """
    tool_call = ToolCall(
        step_id=step_id,
        tool_name=tool_name,
        input=input_data,
        output=None,
        status="success", 
        error_message=None,
        started_at=datetime.utcnow(),
        meta=meta or {},
    )

    db.add(tool_call)
    db.commit()
    db.refresh(tool_call)

    return tool_call



def update_tool_call(
    db: Session,
    tool_call_id: uuid.UUID,
    status: str,
    output_data: dict | None = None,
    error_message: str | None = None
) -> ToolCall:
    """
    Updates a ToolCall with output or error information.
    Sets ended_at automatically.
    """
    tool_call = db.query(ToolCall).filter(ToolCall.id == tool_call_id).first()

    if not tool_call:
        raise ValueError(f"ToolCall with ID {tool_call_id} not found")

    tool_call.status = status              # "success" or "error"
    tool_call.ended_at = datetime.utcnow()

    if output_data is not None:
        tool_call.output = output_data

    if error_message is not None:
        tool_call.error_message = error_message

    db.commit()
    db.refresh(tool_call)

    return tool_call
