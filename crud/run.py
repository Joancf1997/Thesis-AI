from datetime import datetime
from sqlalchemy.orm import Session
from models.run import Run

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
