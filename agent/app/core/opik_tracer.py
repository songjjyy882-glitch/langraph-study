import os

from app.core.config import settings

# Opik은 환경변수 OPIK_URL_OVERRIDE, OPIK_PROJECT_NAME을 자동으로 읽음
if settings.OPIK and settings.OPIK.URL_OVERRIDE:
    os.environ.setdefault("OPIK_URL_OVERRIDE", settings.OPIK.URL_OVERRIDE)
if settings.OPIK and settings.OPIK.PROJECT:
    os.environ.setdefault("OPIK_PROJECT_NAME", settings.OPIK.PROJECT)

from opik.integrations.langchain import OpikTracer  # noqa: E402

# 싱글턴 Opik 트레이서 — LangChain 콜백으로 주입하여 사용
opik_tracer = OpikTracer(
    project_name=settings.OPIK.PROJECT if settings.OPIK else None,
)
