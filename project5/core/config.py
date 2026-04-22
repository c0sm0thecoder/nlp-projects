from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str

    pinecone_api_key: str
    pinecone_index_name: str = "athena-knowledge"
    pinecone_region: str = "us-east-1"

    slack_bot_token: str
    slack_channel_ids: str

    confluence_url: str
    confluence_user: str
    confluence_api_token: str
    confluence_spaces: str = "WIKI,ENG"

    telegram_bot_token: str

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "athena123"

    sync_state_path: str = "sync_state.json"

    @property
    def slack_channel_list(self) -> list[str]:
        return [c.strip() for c in self.slack_channel_ids.split(",") if c.strip()]

    @property
    def confluence_space_list(self) -> list[str]:
        return [s.strip() for s in self.confluence_spaces.split(",") if s.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
