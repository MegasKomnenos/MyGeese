import tarfile

with tarfile.open("my_goose.tar.gz", "w:gz") as tar:
    tar.add('main.py')
    tar.add(f'ddrive/{input()}.txt', arcname='weights.txt')