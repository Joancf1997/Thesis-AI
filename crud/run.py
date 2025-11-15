import uuid
from models.run import Run
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session


def create_run(db: Session, message_id: str, status: str = "queued") -> Run:
    """
    Creates a new Run record linked to a message_id.
    """
    new_run = Run(
        message_id=message_id,
        status=status,
        started_at=datetime.utcnow(),
    )

    db.add(new_run)
    db.commit()
    db.refresh(new_run)

    return new_run

def update_run_message_id(
        db: Session,
        message_id: uuid.UUID,
        new_message_id: uuid.UUID
    ) -> Run:
        
        run = (
            db.query(Run)
            .filter(Run.message_id == message_id)
            .first()
        )

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        print(f"Old id {message_id}")
        print(f"New id {new_message_id}")
        run.message_id = new_message_id

        db.commit()        # session sees the change automatically
        db.refresh(run)

        return run


def end_run(db: Session, run_id: str, status: str = "completed") -> Run:
    """
    Ends a run by setting status and ended_at timestamp.
    """
    run = db.query(Run).filter(Run.id == run_id).first()

    if not run:
        raise ValueError(f"Run with ID {run_id} not found.")

    run.status = status
    run.ended_at = datetime.utcnow()

    db.commit()
    db.refresh(run)

    return run
