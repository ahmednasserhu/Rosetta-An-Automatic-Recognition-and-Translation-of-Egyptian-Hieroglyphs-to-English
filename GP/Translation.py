import subprocess

# Define the command
command = [
    'onmt_translate',
    '-model', 'D:/GP/models/translation_step_3000.pt',
    '-src', 'D:/GP/test_eg.txt',
    '-output', 'D:/GP/output.txt',
    '-replace_unk',
    '-gpu', '0'
]

# Execute the command using subprocess
subprocess.run(command, shell=True)