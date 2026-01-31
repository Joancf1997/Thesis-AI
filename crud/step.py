from sqlalchemy.orm import Session
from models.step import Step
from datetime import datetime
import uuid


def create_step(
    db: Session,
    run_id: uuid.UUID,
    name: str,
    input_data: dict,
    status: str = "started"
) -> Step:
    """
    Creates a new Step entry in the DB linked to a Run.
    Returns the Step instance.
    """
    step = Step(
        run_id=run_id,
        name=name,
        input=input_data,
        output=None,
        status=status,
        created_at=datetime.utcnow(),
    )

    db.add(step)
    db.commit()
    db.refresh(step)

    return step

def update_step(
    db: Session,
    step_id: uuid.UUID,
    status: str = None,
    output_data: dict = None
) -> Step:
    """
    Updates an existing step: status, output data, or both.
    Returns the updated Step instance.
    """
    step = db.query(Step).filter(Step.id == step_id).first()

    if not step:
        raise ValueError(f"Step with ID {step_id} not found")

    if status is not None:
        step.status = status

    if output_data is not None:
        step.output = output_data

    db.commit()
    db.refresh(step)

    return step

