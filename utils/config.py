import argparse
import json
import os


def args_and_config():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    # get base.json arguments
    current_path = os.path.abspath(__file__)
    project_root_dir = os.path.dirname(os.path.dirname(current_path))
    # config_path = os.path.join(project_root_dir, 'configs', 'base.json')
    # config_path = os.path.join(project_root_dir, 'configs', 'oliver_local.json')
    config_path = os.path.join(project_root_dir, 'configs', 'euler.json')
    parser.add_argument('--config', type=str, nargs='?', default=config_path)
    # Parse the argument
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    return config
