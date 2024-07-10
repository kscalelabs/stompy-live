"""Tests if the assets are correcty looaded and what the actor_id are."""

from stompy_live.utils.scene_builders.ycb_builder import model_db


def main() -> None:
    print("Available model IDs:", model_db())


if __name__ == "__main__":
    main()
