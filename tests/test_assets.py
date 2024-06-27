"""Tests if the assets are correcty looaded and what the actor_id are."""

from mani_skill.utils.building.actors.ycb import model_db


def main():
    print("Available model IDs:", model_db())


if __name__ == "__main__":
    main()
