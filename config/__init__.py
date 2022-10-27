import yaml
from pathlib import Path

path = Path(__file__).parent

config = yaml.safe_load(open((path / 'configuration.yml')))

__all__ = [config]
