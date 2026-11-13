"""Worker bootstrap. Domain tasks and persistent orchestration are future work."""

from celery import Celery

from app.config import Settings

settings = Settings()
celery_app = Celery("ada_jobs", broker=settings.celery_broker_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=True,
    worker_prefetch_multiplier=1,
    task_create_missing_queues=False,
)
