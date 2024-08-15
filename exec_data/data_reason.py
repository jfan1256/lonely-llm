import yaml

from utils.system import get_configs
from class_reason.reason_gpt import ReasonGPT

if __name__ == '__main__':
    # Get configuration
    configs = yaml.load(open(get_configs() / 'data' / 'data.yaml', 'r'), Loader=yaml.Loader)

    # Initialize ReasonGPT
    reason_gpt = ReasonGPT(data_path=configs['data_path'], output_path=configs['output_path'], prompt=configs['prompt'])

    # Generate reasons
    reason_gpt.reason_gpt()