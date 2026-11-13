"""Dépendances partagées par les routes."""
from fastapi import Request

from backend.app.container import Services


def get_services(request: Request) -> Services:
    return request.app.state.services
